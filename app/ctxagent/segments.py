"""The Segment: the unit the compaction engine reasons about.

A conversation is a list of Segments, each of which renders to exactly zero or
one OpenAI-format chat messages. Compaction never deletes a Segment inside the
hot zone -- it only rewrites `body` and flips `state`. That guarantees the
assistant(tool_calls) -> tool(tool_call_id) pairing the API requires can never
be broken by a rewrite. Whole-run deletion happens only in deep compaction,
which cuts on validated boundaries.
"""

from __future__ import annotations

import functools
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import tokens

# ---- kinds ----------------------------------------------------------------
K_SYSTEM = "system"
K_USER = "user"
K_ASSISTANT = "assistant_text"
K_TOOL_CALL = "tool_call"        # assistant message carrying tool_calls
K_TOOL_RESULT = "tool_result"    # role=tool message
K_MEDIA = "media"              # native images supplied by tools, role=user
K_SUMMARY = "summary"            # synthetic, produced by compaction
K_RESEARCH = "research_handoff"   # source-linked notes, not user intent
K_TOOL_CHANGE = "tool_change"     # deferred tools surfaced/removed (dynamic_tools), role=system

# ---- states ---------------------------------------------------------------
S_ORIGINAL = "original"
S_OFFLOADED = "offloaded"    # body on disk, handle in context
# Set by the incremental tiers removed since (age masking, dedup, condensation); segments
# of older chats may still carry them.
S_MASKED = "masked"
S_SUPERSEDED = "superseded"
S_CONDENSED = "condensed"
S_FROZEN_NOTE = "frozen"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


#: Body of the synthetic result that closes a call whose execution left no record.
INTERRUPTED_RESULT = ("[interrupted] No result recorded. Execution may not have started, "
                      "or may have produced side effects. Inspect state before retrying.")


def is_interrupted_placeholder(seg: "Segment") -> bool:
    return seg.kind == K_TOOL_RESULT and (seg.body or "").startswith("[interrupted] No result recorded")


def result_owners(segs: list["Segment"]) -> dict[int, int]:
    """Index of each tool result -> index of the call segment it answers (the latest earlier
    assistant(tool_calls) segment carrying its id). Results without such a call are absent."""
    owner_of_id: dict[str, int] = {}
    out: dict[int, int] = {}
    for i, s in enumerate(segs):
        if s.kind == K_TOOL_CALL:
            for tc in s.tool_calls or []:
                owner_of_id[tc.get("id") or ""] = i
        elif s.kind == K_TOOL_RESULT:
            owner = owner_of_id.get(s.tool_call_id or "")
            if owner is not None:
                out[i] = owner
    return out


def pick_results(segs: list["Segment"], indexes: list[int]) -> dict[str, int]:
    """One result per call id among `indexes`: the first recorded one, except that a real
    result wins over an `[interrupted]` placeholder for the same id."""
    picks: dict[str, int] = {}
    for j in indexes:
        cid = segs[j].tool_call_id or ""
        cur = picks.get(cid)
        if cur is None or (is_interrupted_placeholder(segs[cur]) and not is_interrupted_placeholder(segs[j])):
            picks[cid] = j
    return picks


def wire_problems(messages: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """Tool pairing problems in an OpenAI-format request, as (problem, message index).

    The strictest common subset of the provider rules: an assistant message with tool_calls
    is followed by exactly one tool message per call id before any other message (OpenAI:
    "An assistant message with 'tool_calls' must be followed by tool messages responding to
    each 'tool_call_id'"; Anthropic: tool_result blocks first in the next user turn, "each
    tool_use must have a single result"), a tool message answers a call of the assistant
    message right before its run, and call ids are unique within the request."""
    out: list[tuple[str, int]] = []
    pending: set[str] = set()
    current: set[str] = set()
    seen: set[str] = set()
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "tool":
            cid = m.get("tool_call_id") or ""
            if cid not in current:
                out.append(("tool message without a preceding call", i))
            elif cid not in pending:
                out.append(("second tool message for one call", i))
            else:
                pending.discard(cid)
            continue
        if pending:
            out.append(("call not followed by all of its results", i))
            pending = set()
        current = set()
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                cid = tc.get("id") or ""
                if not cid:
                    out.append(("tool call without id", i))
                if cid in seen:
                    out.append(("duplicate tool call id", i))
                seen.add(cid)
                pending.add(cid)
                current.add(cid)
            for tc in m["tool_calls"]:
                try:
                    if not isinstance(json.loads((tc.get("function") or {}).get("arguments") or ""), dict):
                        out.append(("tool call arguments are not a JSON object", i))
                except ValueError:
                    out.append(("tool call arguments are not a JSON object", i))
    if pending:
        out.append(("trailing call without results", len(messages)))
    return out


@functools.lru_cache(maxsize=1024)
def _wire_arguments(raw: str) -> str | None:
    """None when `raw` already is a JSON object; otherwise its JSON-object re-encoding."""
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return None
    except (ValueError, TypeError):
        try:  # raw control characters inside strings (seen from Claude models)
            value = json.loads(raw, strict=False)
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)
        except (ValueError, TypeError):
            value = raw
    if raw == "":
        return "{}"
    return json.dumps({"_unparsed_arguments": value if isinstance(value, str) else raw},
                      ensure_ascii=False)


def _wire_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") or {}
    raw = fn.get("arguments")
    if not isinstance(raw, str):
        raw = "" if raw is None else json.dumps(raw, ensure_ascii=False)
        fixed = _wire_arguments(raw) or raw
    else:
        fixed = _wire_arguments(raw)
        if fixed is None:
            return tc
    return {**tc, "function": {**fn, "arguments": fixed}}


@dataclass
class Segment:
    seq: int
    role: str
    kind: str
    body: str = ""
    id: str = field(default_factory=_new_id)
    created_at: float = field(default_factory=time.time)

    # tool plumbing
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    # normalised identity of the thing observed, e.g. "file:c:/p/x.py".
    # Two segments with the same resource_key observe the same object.
    resource_key: str | None = None
    # true when this segment *changed* the resource (write/edit/command)
    mutates: bool = False
    # how much of the resource this observation captured (see tools.DETAIL_FULL).
    # A later, *shallower* observation must never supersede a deeper one.
    detail: int = 100
    # observation channel (see tools.CH_*): detail only compares within one
    channel: str = "raw"
    observation: dict[str, Any] = field(default_factory=dict)
    images: list[dict[str, Any]] = field(default_factory=list)
    reasoning_details: list[dict[str, Any]] = field(default_factory=list)

    # compaction bookkeeping
    state: str = S_ORIGINAL
    original_body: str | None = None
    original_tokens: int = 0
    blob_id: str | None = None
    frozen: bool = False
    pinned: bool = False
    # Manual pins survive deep compaction; recall pins protect incremental passes.
    pin_scope: str = "manual"
    # A pin set by the `pin` tool may expire: the conversation's tool-batch counter
    # value after which it is dropped (None = until unpinned / forever).
    pin_until_batch: int | None = None
    note: str = ""                 # human-readable reason, shown in the UI

    def __post_init__(self) -> None:
        self._tok_cache: tuple[int, int] | None = None
        if not self.original_tokens:
            self.original_tokens = self.tokens()

    # -- sizing ------------------------------------------------------------
    def tokens(self) -> int:
        """Token cost of this segment, memoised on the body.

        The compaction engine asks for this constantly (every tier, every
        stats call, every hot-zone walk). Re-encoding a 40k-token file body
        each time dominates the runtime, so the count is cached and keyed on
        the body's hash -- which Python memoises on the string object itself,
        making repeat lookups O(1).
        """
        key = hash((self.body, self.state, tuple(i.get("id") for i in self.images), json.dumps(self.reasoning_details, sort_keys=True)))
        cache = getattr(self, "_tok_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        msg = self.to_message()
        n = tokens.count_message(msg) if msg else 0
        self._tok_cache = (key, n)
        return n

    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.tokens())

    # -- rendering ---------------------------------------------------------
    def to_message(self) -> dict[str, Any] | None:
        if self.kind == K_TOOL_CALL:
            msg: dict[str, Any] = {"role": "assistant"}
            msg["content"] = self.body or None
            msg["tool_calls"] = self.tool_calls or []
            if self.reasoning_details:
                msg["reasoning_details"] = self.reasoning_details
            return msg
        if self.kind == K_TOOL_RESULT:
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "content": self.body if self.body else "(empty)",
            }
        if self.kind in (K_SYSTEM,):
            return {"role": "system", "content": self.body}
        if self.kind == K_TOOL_CHANGE:
            # Only the direct Anthropic API renders these (anthropic_api.convert_messages);
            # every other path drops them (llm, anthropic_api without deferred tools).
            return {"role": "system", "content": [
                {"type": change, "tool": {"type": "tool_reference", "name": name}}
                for change, key in (("tool_addition", "add"), ("tool_removal", "remove"))
                for name in self.observation.get(key) or []]}
        if self.kind == K_MEDIA:
            blocks = [{"type": "text", "text": self.body}]
            if self.state == S_ORIGINAL:
                blocks.extend({"type": "image_ref", "image": image} for image in self.images)
            return {"role": "user", "content": blocks}
        if self.kind in (K_USER, K_SUMMARY, K_RESEARCH):
            return {"role": "user", "content": self.body}
        if self.kind == K_ASSISTANT:
            if not self.body.strip():
                return None
            return {"role": "assistant", "content": self.body,
                    **({"reasoning_details": self.reasoning_details} if self.reasoning_details else {})}
        return None

    # -- mutation helpers --------------------------------------------------
    def invalidate(self) -> None:
        self._tok_cache = None

    def rewrite(self, new_body: str, state: str, note: str) -> int:
        """Replace the effective body. Returns tokens freed."""
        if self.frozen or self.pinned:
            return 0
        if self.original_body is None:
            self.original_body = self.body
        before = self.tokens()
        self.body = new_body
        self.state = state
        self.note = note
        return max(0, before - self.tokens())

    def restore(self) -> None:
        if not self.pinned or self.pin_scope != "manual":
            self.pin_scope = "recall"
        if self.original_body is not None:
            self.body = self.original_body
            self.original_body = None
        self.state = S_ORIGINAL
        self.note = ""
        self.pinned = True  # recovered once => stop truncating this path

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seq": self.seq,
            "role": self.role,
            "kind": self.kind,
            "body": self.body,
            "created_at": self.created_at,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "resource_key": self.resource_key,
            "mutates": self.mutates,
            "detail": self.detail,
            "channel": self.channel,
            "observation": self.observation,
            "images": self.images,
            "reasoning_details": self.reasoning_details,
            "state": self.state,
            "original_body": self.original_body,
            "original_tokens": self.original_tokens,
            "blob_id": self.blob_id,
            "frozen": self.frozen,
            "pinned": self.pinned,
            "pin_scope": self.pin_scope,
            "pin_until_batch": self.pin_until_batch,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        seg = cls(
            seq=d["seq"],
            role=d["role"],
            kind=d["kind"],
            body=d.get("body", ""),
            id=d.get("id") or _new_id(),
            created_at=d.get("created_at", time.time()),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
            tool_name=d.get("tool_name"),
            resource_key=d.get("resource_key"),
            mutates=bool(d.get("mutates")),
            detail=int(d.get("detail", 100)),
            channel=d.get("channel", "raw"),
            observation=d.get("observation") or {},
            images=d.get("images") or [],
            reasoning_details=d.get("reasoning_details") or [],
            state=d.get("state", S_ORIGINAL),
            original_body=d.get("original_body"),
            original_tokens=int(d.get("original_tokens") or 0),
            blob_id=d.get("blob_id"),
            frozen=bool(d.get("frozen")),
            pinned=bool(d.get("pinned")),
            pin_scope=d.get("pin_scope", "legacy"),
            pin_until_batch=(int(d["pin_until_batch"]) if d.get("pin_until_batch") is not None else None),
            note=d.get("note", ""),
        )
        return seg

    # -- UI summary --------------------------------------------------------
    def preview(self, n: int = 160) -> str:
        if self.kind == K_TOOL_CALL and self.tool_calls:
            names = ", ".join(
                (tc.get("function") or {}).get("name", "?") for tc in self.tool_calls
            )
            extra = f" {self.body[:60]}" if self.body.strip() else ""
            return f"→ {names}{extra}"
        s = " ".join((self.body or "").split())
        return s[:n] + ("…" if len(s) > n else "")


class Conversation:
    """Ordered segment list + the frozen-prefix invariant."""

    def __init__(self, segments: list[Segment] | None = None, frozen_upto: int = 0):
        self.segments: list[Segment] = segments or []
        # index (exclusive) of the cache-stable prefix. Monotone: only grows,
        # except on a deep compaction which rebuilds the prefix entirely.
        self.frozen_upto: int = frozen_upto
        self._next_seq = (max((s.seq for s in self.segments), default=-1)) + 1
        self.cache_state: dict = {}
        self.compaction_state: dict = {}
        self.archive: dict[str, str] = {}  # segment id -> immutable JSON blob
        from .workspace import DocumentWorkspace
        self.workspace = DocumentWorkspace()
        from .browser_workspace import BrowserWorkspace
        self.browser = BrowserWorkspace()
        from .canvas import Canvas
        self.canvas = Canvas()

    # -- building ----------------------------------------------------------
    def add(self, seg: Segment) -> Segment:
        seg.seq = self._next_seq
        self._next_seq += 1
        self.segments.append(seg)
        return seg

    def new(self, role: str, kind: str, body: str = "", **kw: Any) -> Segment:
        return self.add(Segment(seq=-1, role=role, kind=kind, body=body, **kw))

    # -- querying ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self) -> Iterable[Segment]:
        return iter(self.segments)

    def total_tokens(self) -> int:
        return sum(s.tokens() for s in self.segments) + 3

    def original_tokens(self) -> int:
        return sum(s.original_tokens for s in self.segments) + 3

    def to_messages(self, sources: list[tuple[int, str | None]] | None = None) -> list[dict[str, Any]]:
        """Messages for the API, always in a shape every provider accepts.

        `sources`, when given, receives one (segment index, segment id) per message, in the
        same order: results are emitted with their call, so the message order is not the
        segment order. A placeholder result has the index of its call and no id.

        The stored segments normally satisfy the pairing invariant (see `repair_tool_pairs`),
        but a request is never sent malformed even when they do not:

        * a tool result whose call is no longer in the history (an earlier compaction broke the
          pairing) is sent as a user note: OpenAI rejects the whole request otherwise ("No tool
          call found for function call output"), and every later request with it;
        * the results of a call are emitted directly after it even when a tool wrote another
          segment while its batch was in flight (document_close's handoff note), because OpenAI
          requires the tool messages to follow the call before any other message and Anthropic
          requires tool_result blocks first in the next user turn;
        * a second result for the same call id is dropped ("each tool_use must have a single
          result"); a recorded result wins over an `[interrupted]` placeholder;
        * a call without any result gets a placeholder result;
        * tool-call arguments that are not a JSON object (a call cut off at the output limit) are
          re-encoded as one, since some routes parse them into a structured `input`.
        """
        segs = self.segments
        owners = result_owners(segs)
        attached: dict[int, list[int]] = {}
        for j, i in owners.items():
            attached.setdefault(i, []).append(j)
        out: list[dict[str, Any]] = []
        src: list[tuple[int, str | None]] = sources if sources is not None else []
        for i, s in enumerate(segs):
            if s.kind == K_TOOL_RESULT and i in owners:
                continue  # emitted (or dropped as a duplicate) together with its call
            m = s.to_message()
            if m is None:
                continue
            if s.kind == K_TOOL_CALL:
                calls = s.tool_calls or []
                if calls:
                    m = {**m, "tool_calls": [_wire_call(tc) for tc in calls]}
                out.append(m)
                src.append((i, s.id))
                picks = pick_results(segs, attached.get(i, []))
                for tc in calls:
                    cid = tc.get("id") or ""
                    j = picks.get(cid)
                    if j is None:
                        name = (tc.get("function") or {}).get("name") or "tool"
                        out.append({"role": "tool", "tool_call_id": cid,
                                    "content": f"{INTERRUPTED_RESULT} (tool {name})"})
                        src.append((i, None))
                    else:
                        out.append(segs[j].to_message())
                        src.append((j, segs[j].id))
                continue
            if s.kind == K_TOOL_RESULT:
                content = m.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                m = {"role": "user", "content": f"[Result of the tool call {s.tool_name or 'tool'} "
                                                f"whose call was compacted away]\n{content}"}
            out.append(m)
            src.append((i, s.id))
        return out

    def group_start(self, index: int) -> int:
        """`index` as a boundary that splits no tool group: moved back to the group's call
        when it falls between a call and its results (or the media after them)."""
        segs = self.segments
        k = index
        while 0 < k < len(segs) and segs[k].kind in (K_TOOL_RESULT, K_MEDIA):
            k -= 1
        return k if k < index and segs[k].kind == K_TOOL_CALL else index

    def by_id(self, sid: str) -> Segment | None:
        for s in self.segments:
            if s.id == sid:
                return s
        return None

    def archive_segments(self, segments: list[Segment], session_id: str) -> None:
        from .blobs import put_blob
        for seg in segments:
            if seg.id not in self.archive:
                data = seg.to_dict()
                self.archive[seg.id] = put_blob(session_id, "archive-" + seg.id,
                    json.dumps(data, ensure_ascii=False))

    def archived_by_id(self, sid: str) -> Segment | None:
        from .blobs import get_blob
        handle = self.archive.get(sid)
        if not handle:
            return None
        data = get_blob(handle)
        if data is None:
            raise RuntimeError(f"Archived segment {sid} is missing from disk")
        return Segment.from_dict(json.loads(data))

    def migrate_legacy_recall_pins(self) -> None:
        """Classify old automatic recall pins using recorded calls, once per chat.

        Unknown pins remain permanent. Explicit pin_scope always wins. Archives
        are read only for evidence; no source content is changed by migration.
        """
        if self.compaction_state.get('pin_scopes_migrated'):
            return
        targets = set()
        def inspect(seg):
            for tc in seg.tool_calls or []:
                fn = tc.get('function') or {}
                if fn.get('name') == 'recall':
                    try:
                        targets.add(json.loads(fn.get('arguments') or '{}').get('segment_id'))
                    except (ValueError, AttributeError):
                        pass
        for seg in self.segments:
            inspect(seg)
        for sid in self.archive:
            try:
                old = self.archived_by_id(sid)
                if old:
                    inspect(old)
            except (ValueError, RuntimeError, OSError):
                continue  # unknown provenance never licenses removing a pin
        recalled_results = set()
        for seg in self.segments:
            if seg.pin_scope == 'legacy' and (seg.id in targets or seg.tool_name == 'recall'):
                seg.pin_scope = 'recall'
                recalled_results.add(seg.id)
        for seg in self.segments:
            if seg.kind == K_MEDIA and seg.pin_scope == 'legacy' and any(
                f'Source result: {sid}.' in seg.body for sid in recalled_results
            ):
                seg.pin_scope = 'recall'
        self.compaction_state['pin_scopes_migrated'] = True

    def repair_tool_pairs(self, start: int = 0) -> int:
        """Make every tool batch well-formed in place; never rerun uncertain side effects.

        For each assistant(tool_calls) segment at or after `start`:
          * its results move directly after it, in call order. A tool may write another
            segment while its batch is in flight (document_close appends the user-kind
            handoff note before its own result); such segments now follow the results;
          * a second result for the same call id is removed when it is an `[interrupted]`
            placeholder and otherwise kept as a plain note (a recorded result always wins);
          * a call without any result gets an `[interrupted]` placeholder.

        The old version only looked at the results directly after the call, so a handoff note
        in between made it add a placeholder for a call that HAD a result: two results for one
        id, which Anthropic rejects on every later request ("each tool_use must have a single
        result"). Returns the number of changes; the frozen prefix is cut back only to the
        first changed position, so the cached prefix before it stays valid.
        """
        segs = self.segments
        owners = result_owners(segs)
        attached: dict[int, list[int]] = {}
        for j, i in owners.items():
            attached.setdefault(i, []).append(j)
        rebuilt: list[Segment] = []
        changed: set[int] = set()  # id() of segments whose content changed in place
        dropped = 0
        for i, seg in enumerate(segs):
            if i in owners and owners[i] >= start:
                continue  # placed together with its call below (or dropped as a duplicate)
            rebuilt.append(seg)
            if seg.kind != K_TOOL_CALL or i < start:
                continue
            idxs = attached.get(i, [])
            picks = pick_results(segs, idxs)
            extras = []
            for j in idxs:
                if picks.get(segs[j].tool_call_id or "") == j:
                    continue  # placed below
                if is_interrupted_placeholder(segs[j]):
                    dropped += 1  # a placeholder next to a recorded result carries nothing
                    continue
                dup = segs[j]  # a second REAL result: keep its content, not its pairing
                dup.kind, dup.role = K_USER, "user"
                dup.body = (f"[Second result recorded for tool call {dup.tool_call_id} "
                            f"({dup.tool_name or 'tool'}); the first one is above]\n{dup.body}")
                dup.tool_call_id = None
                dup.invalidate()
                changed.add(id(dup))
                extras.append(dup)
            for tc in seg.tool_calls or []:
                j = picks.get(tc.get("id") or "")
                if j is not None:
                    rebuilt.append(segs[j])
                    continue
                placeholder = Segment(seq=self._next_seq, role="tool", kind=K_TOOL_RESULT,
                                      body=INTERRUPTED_RESULT, tool_call_id=tc.get("id"),
                                      tool_name=(tc.get("function") or {}).get("name"))
                self._next_seq += 1
                changed.add(id(placeholder))
                rebuilt.append(placeholder)
            # Converted duplicates go right after the batch's results. Other segments written
            # while the batch was in flight (handoff notes, media) follow in recorded order.
            rebuilt.extend(extras)
        first = next((k for k, s in enumerate(rebuilt)
                      if k >= len(segs) or s is not segs[k] or id(s) in changed), None)
        if first is None and len(rebuilt) == len(segs):
            return 0
        if first is None:
            first = len(rebuilt)
        moved = sum(1 for k in range(first, min(len(rebuilt), len(segs)))
                    if rebuilt[k] is not segs[k] and id(rebuilt[k]) not in changed)
        changes = len(changed) + dropped + moved
        self.segments = rebuilt
        if first < self.frozen_upto:
            # Only what follows the first changed position leaves the cached prefix. That may
            # be inside a tool group; the request then puts the live views behind the group's
            # results (agent._run_turn), so the cached call stays cached.
            self.frozen_upto = first
            for s in rebuilt[first:]:
                s.frozen = False
        return max(1, changes)

    # -- the cache-stable split -------------------------------------------
    def compute_hot_start(self, hot_zone_tokens: int) -> int:
        """Index of the first segment inside the hot (rewritable) zone.

        Walks backwards accumulating tokens. The result is clamped so it can
        never move *backwards* past an already-frozen boundary: once a segment
        has been sent to the provider in its final form, rewriting it would
        invalidate the cache for everything after it.
        """
        acc = 0
        idx = len(self.segments)
        for i in range(len(self.segments) - 1, -1, -1):
            acc += self.segments[i].tokens()
            if acc > hot_zone_tokens:
                idx = i + 1
                break
            idx = i
        return max(idx, self.frozen_upto)

    def freeze_prefix(self, hot_start: int) -> int:
        """Mark everything before `hot_start` immutable. Returns newly frozen.

        A tool group is frozen whole or not at all. A boundary right after a large call
        (a write_file of a big file outweighs the hot zone by itself) used to freeze the
        call without its results, and the live views that go right behind the frozen
        prefix then sat between the call and its results: Anthropic rejects that request
        ("tool_use ids were found without tool_result blocks immediately after")."""
        hot_start = self.group_start(hot_start)
        n = 0
        for s in self.segments[:hot_start]:
            if not s.frozen:
                s.frozen = True
                n += 1
        self.frozen_upto = max(self.frozen_upto, hot_start)
        return n

    def to_dict(self) -> dict[str, Any]:
        return {
            "frozen_upto": self.frozen_upto,
            "archive": self.archive,
            "workspace": self.workspace.to_dict(),
            "browser": self.browser.to_dict(),
            "canvas": self.canvas.to_dict(),
            "cache_state": self.cache_state,
            "compaction_state": self.compaction_state,
            "next_seq": self._next_seq,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Conversation":
        if not d:
            return cls()
        segs = [Segment.from_dict(x) for x in d.get("segments", [])]
        conv = cls(segs, int(d.get("frozen_upto", 0)))
        conv.archive = dict(d.get("archive") or {})
        from .workspace import DocumentWorkspace
        conv.workspace = DocumentWorkspace.from_dict(d.get("workspace"))
        from .browser_workspace import BrowserWorkspace
        conv.browser = BrowserWorkspace.from_dict(d.get("browser"))
        from .canvas import Canvas
        conv.canvas = Canvas.from_dict(d.get("canvas"))
        conv.cache_state = dict(d.get("cache_state") or {})
        conv.compaction_state = dict(d.get("compaction_state") or {})
        conv._next_seq = max(conv._next_seq, int(d.get("next_seq", 0)))
        return conv

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
