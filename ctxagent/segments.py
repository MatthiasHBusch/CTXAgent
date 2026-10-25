"""The Segment: the unit the compaction engine reasons about.

A conversation is a list of Segments, each of which renders to exactly zero or
one OpenAI-format chat messages. Compaction never deletes a Segment inside the
hot zone -- it only rewrites `body` and flips `state`. That guarantees the
assistant(tool_calls) -> tool(tool_call_id) pairing the API requires can never
be broken by a rewrite. Whole-run deletion happens only in deep compaction,
which cuts on validated boundaries.
"""

from __future__ import annotations

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

# ---- states ---------------------------------------------------------------
S_ORIGINAL = "original"
S_MASKED = "masked"          # age-tiered truncation applied
S_SUPERSEDED = "superseded"  # a newer observation of the same resource exists
S_OFFLOADED = "offloaded"    # body on disk, handle in context
S_CONDENSED = "condensed"    # LLM-condensed in place (significance pass)
S_FROZEN_NOTE = "frozen"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


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
    significance: float = 0.5
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
            "significance": self.significance,
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
            significance=float(d.get("significance", 0.5)),
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

    def to_messages(self) -> list[dict[str, Any]]:
        """Messages for the API. A tool result whose call is no longer in the history (the
        pairing invariant was broken by an earlier compaction) is sent as a user note instead
        of a tool message: OpenAI rejects the whole request otherwise ("No tool call found for
        function call output"), and every later request with it."""
        out = []
        seen: set[str] = set()
        for s in self.segments:
            m = s.to_message()
            if m is None:
                continue
            if s.kind == K_TOOL_CALL:
                for tc in s.tool_calls or []:
                    seen.add(tc.get("id") or "")
            elif s.kind == K_TOOL_RESULT and s.tool_call_id not in seen:
                content = m.get("content")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                m = {"role": "user", "content": f"[Result of the tool call {s.tool_name or 'tool'} "
                                                f"whose call was compacted away]\n{content}"}
            out.append(m)
        return out

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

    def repair_tool_pairs(self) -> int:
        """Close interrupted batches in place; never rerun uncertain side effects."""
        repaired = 0
        i = 0
        while i < len(self.segments):
            seg = self.segments[i]
            if seg.kind != K_TOOL_CALL:
                i += 1
                continue
            end = i + 1
            while end < len(self.segments) and self.segments[end].kind == K_TOOL_RESULT:
                end += 1
            found = {s.tool_call_id for s in self.segments[i + 1:end]}
            missing = [tc for tc in seg.tool_calls or [] if tc.get('id') not in found]
            for tc in missing:
                result = Segment(seq=self._next_seq, role='tool', kind=K_TOOL_RESULT,
                    body='[interrupted] No result recorded. Execution may not have started, '
                         'or may have produced side effects. Inspect state before retrying.',
                    tool_call_id=tc.get('id'), tool_name=(tc.get('function') or {}).get('name'))
                self._next_seq += 1
                self.segments.insert(end, result)
                end += 1
                repaired += 1
            i = end
        if repaired:
            # An old malformed frozen batch may have been repaired.
            self.frozen_upto = 0
            for s in self.segments:
                s.frozen = False
        return repaired

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
        """Mark everything before `hot_start` immutable. Returns newly frozen."""
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
        conv.cache_state = dict(d.get("cache_state") or {})
        conv.compaction_state = dict(d.get("compaction_state") or {})
        conv._next_seq = max(conv._next_seq, int(d.get("next_seq", 0)))
        return conv

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
