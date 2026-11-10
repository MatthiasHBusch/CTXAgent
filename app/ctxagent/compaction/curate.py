"""Agent-curated context (the `curate_context` tool, rewrite_policy='agent').

Only the agent knows which items are still load-bearing: a 3 kB grep that located the bug
stays useful for an hour, a 40 kB build log is dead the moment the agent has read
"BUILD_OK". The agent replaces what it no longer needs verbatim by its own short text:

    curate_context(3, "BUILD_OK, 0 warnings", 7, "Foo::bar is defined at x.cpp:120")

* A reference counts back from the newest item before the call: 1 is the newest tool
  result, tool call or answer, 2 the one before, and so on, in the order the model reads
  them (a call's results follow it). An answer is an assistant message without tool calls;
  text written alongside tool calls belongs to them and is no item of its own (in a live
  check on Opus 5.5 the model counted exactly that way, 4 of 4). User messages, summaries
  and harness messages are not items; neither is the batch that runs the curate call.
* The original goes to a blob, and recall(segment_id) returns it. The visible stub is the
  agent's replacement plus that handle; a tool call's arguments become {"replaced": text,
  "recall_segment_id": id}.
* Items not named stay verbatim; a call without pairs changes nothing.
* The provider re-reads the prompt up to the newest cache point before the oldest replaced
  item and writes the rest again. The next request puts a breakpoint right before that item
  (cache_policy.plan_rungs), so replacing recent items is cheap.
* The review hint (engine._hint) names the cache breakpoints in the history by reference
  and a few words of the item (`hint`): the items behind one breakpoint are re-written
  together, whichever of them is replaced.

The safety net stays automatic (engine.py): deep compaction at deep_trigger_ratio.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..segments import (Conversation, Segment, K_ASSISTANT, K_TOOL_CALL, K_TOOL_RESULT, S_OFFLOADED)
from ..blobs import put_blob

#: Longer replacements are cut here; a replacement is a takeaway, not a second copy.
REPLACEMENT_MAX_CHARS = 4000
HINT = "[harness: context review due]"


@dataclass
class Item:
    seg: Segment
    part: str                 # 'result', 'call' (one tool call's arguments) or 'answer'
    call: int | None = None   # index in seg.tool_calls for part 'call'

    def describe(self, n: int = 70) -> str:
        if self.part == 'call':
            fn = (self.seg.tool_calls[self.call].get('function') or {})
            return f"{fn.get('name') or 'tool'} call ({_args(fn.get('arguments'), n)})"
        if self.part == 'result':
            return f"{self.seg.tool_name or 'tool'} result ({_first_line(self.seg.body, n)})"
        return f"answer ({_first_line(self.seg.body, n)})"


def _first_line(text: str | None, n: int = 70) -> str:
    line = (text or "").strip().split("\n", 1)[0].replace("\r", "")
    return line[:n] + ("..." if len(line) > n else "")


def _args(raw, n: int = 70) -> str:
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        obj = raw
    if isinstance(obj, dict):
        text = " ".join(f"{k}={str(v)[:40]}" for k, v in obj.items())
    else:
        text = str(obj or "")
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


def _running_batch(conv: Conversation) -> int:
    """Index of the newest call segment: the batch that runs this curate call."""
    return max((i for i, s in enumerate(conv.segments) if s.kind == K_TOOL_CALL), default=len(conv.segments))


def _placed(conv: Conversation, running: bool, sources: list | None = None) -> list[tuple[Item, int]]:
    """(item, index of its message in conv.to_messages()), oldest first."""
    cut = _running_batch(conv) if running else len(conv.segments)
    if sources is None:
        sources = []
        conv.to_messages(sources)
    out: list[tuple[Item, int]] = []
    seen: set[str] = set()
    for at, (i, sid) in enumerate(sources):
        if sid is None or i >= cut or sid in seen:
            continue
        seen.add(sid)
        seg = conv.segments[i]
        if seg.kind == K_TOOL_CALL:
            out.extend((Item(seg, 'call', k), at) for k in range(len(seg.tool_calls or [])))
        elif seg.kind == K_TOOL_RESULT:
            out.append((Item(seg, 'result'), at))
        elif seg.kind == K_ASSISTANT and (seg.body or "").strip():
            out.append((Item(seg, 'answer'), at))
    return out


def items(conv: Conversation, running: bool = True) -> list[Item]:
    """The replaceable items, oldest first, in the order the model reads them. The newest
    batch runs the curate call and holds no items; running=False counts it too, the way a
    call in the next response will."""
    return [item for item, _ in _placed(conv, running)]


def breakpoints(conv: Conversation) -> list[tuple[int, Item]] | None:
    """The cache breakpoints the next request sends inside the history, newest first, as
    (reference, item): the item at or right before each rung (cache_policy.plan_rungs),
    counted the way a curate_context call in the next response counts. Replacing an item
    re-writes everything behind the nearest breakpoint before it. None when the provider
    caches without breakpoints or no request was sent yet."""
    from .. import cache_policy, tokens
    state = conv.cache_state or {}
    if state.get('cache_family') not in ('anthropic', 'openai') or state.get('history_hashes') is None:
        return None
    sources: list[tuple[int, str | None]] = []
    history = conv.to_messages(sources)
    # The history only grew since that request (a change resets the review counter, so no
    # hint follows it): the planner sees the known messages and the new ones behind them.
    known = list(state['history_hashes'])[:len(history)]
    plan = cache_policy.plan_rungs(state, known + [''] * (len(history) - len(known)),
                                   [tokens.count_message(m) for m in history],
                                   state.get('views_hash'), int(state.get('views_count') or 0))
    placed = _placed(conv, False, sources)
    out: list[tuple[int, Item]] = []
    for rung in sorted(plan['rungs'], reverse=True):
        before = [k for k, (_, at) in enumerate(placed) if at <= rung]
        if before and (not out or out[-1][0] != len(placed) - before[-1]):
            out.append((len(placed) - before[-1], placed[before[-1]][0]))
    return out


def hint(conv: Conversation) -> str:
    """The review hint. Where the provider caches at breakpoints it says where they lie."""
    found = breakpoints(conv)
    if found is None:
        return HINT
    if not found:
        return HINT[:-1] + "; no cache breakpoint in the history yet]"
    return HINT[:-1] + "; cache breakpoints " + " and ".join(
        f"after item {n} = {item.describe(40)}" for n, item in found) + "]"


def parse_pairs(pairs) -> tuple[list[tuple[int, str]], list[str]]:
    """(reference, replacement) pairs from what the model sent: a flat list
    [ref1, text1, ref2, text2, ...], a list of [ref, text] pairs or {reference, replacement}
    objects, or an object {ref: text}."""
    problems: list[str] = []
    flat: list = []
    if isinstance(pairs, dict):
        for k, v in pairs.items():
            flat += [k, v]
    elif isinstance(pairs, (list, tuple)):
        for p in pairs:
            if isinstance(p, dict):
                flat += [p.get('reference', p.get('ref')), p.get('replacement', p.get('text', ''))]
            elif isinstance(p, (list, tuple)) and len(p) == 2:
                flat += list(p)
            else:
                flat.append(p)
    elif pairs not in (None, ""):
        flat = [pairs]
    out: list[tuple[int, str]] = []
    if len(flat) % 2:
        problems.append(f"{flat[-1]!r} has no replacement")
        flat = flat[:-1]
    for ref, text in zip(flat[0::2], flat[1::2]):
        try:
            n = int(str(ref).strip().lstrip('#'))
        except (TypeError, ValueError):
            problems.append(f"{ref!r} is not a reference (1 = the newest item, 2 the one before, ...)")
            continue
        out.append((n, "" if text is None else str(text).strip()))
    return out, problems


def _stub(seg: Segment, text: str) -> str:
    return (f"[replaced: {text} | " if text else "[replaced | ") + \
        f'recall(segment_id="{seg.id}") restores the original]'


def _original(seg: Segment) -> str:
    """What recall returns: the body, and for a call segment also its calls' arguments."""
    body = seg.original_body if seg.original_body is not None else (seg.body or "")
    if seg.kind != K_TOOL_CALL:
        return body
    calls = seg.observation.get('original_tool_calls') or seg.tool_calls or []
    parts = [body] if body.strip() else []
    for tc in calls:
        fn = tc.get('function') or {}
        parts.append(f"[tool call {fn.get('name') or 'tool'}]\n{fn.get('arguments') or ''}")
    return "\n\n".join(parts)


def replace(item: Item, text: str, session_id: str) -> int:
    """Replace one item by `text`. Returns the tokens freed (negative when it grew)."""
    seg = item.seg
    before = seg.tokens()
    if seg.kind == K_TOOL_CALL and 'original_tool_calls' not in seg.observation:
        seg.observation['original_tool_calls'] = json.loads(json.dumps(seg.tool_calls or []))
    if not seg.blob_id:
        seg.blob_id = put_blob(session_id, seg.id, _original(seg))
    if seg.pinned:   # replacing it overrides the agent's own earlier pin
        seg.pinned = False
        seg.pin_until_batch = None
        seg.pin_scope = "manual"
    if item.part == 'call':
        call = seg.tool_calls[item.call]
        call['function'] = {**(call.get('function') or {}),
                            'arguments': json.dumps({"replaced": text, "recall_segment_id": seg.id},
                                                    ensure_ascii=False)}
    else:
        if seg.original_body is None:
            seg.original_body = seg.body
        seg.body = _stub(seg, text)
    seg.state = S_OFFLOADED
    seg.note = "replaced by agent"
    seg.invalidate()
    return before - seg.tokens()


def curate(conv: Conversation, session_id: str, pairs=None, *, tool_overhead: int = 0) -> dict:
    """Apply the agent's (reference, replacement) pairs. Returns the report the tool shows
    (`lines`) and the compaction log keeps (dropped, freed, problems, token counts)."""
    todo, problems = parse_pairs(pairs)
    report = {"replaced": [], "dropped": [], "problems": problems, "freed": 0}
    if not todo and not problems:
        return report
    found = items(conv)
    before = conv.total_tokens() + tool_overhead
    chosen: dict[int, tuple[Item, str]] = {}
    for n, text in todo:
        if not 1 <= n <= len(found):
            problems.append(f"{n}: there are {len(found)} items (1 = the newest)")
            continue
        if len(text) > REPLACEMENT_MAX_CHARS:
            text = text[:REPLACEMENT_MAX_CHARS] + " [cut]"
        chosen[n] = (found[-n], text)            # a reference named twice: the last text wins
    freed = 0
    for n in sorted(chosen):
        item, text = chosen[n]
        what = item.describe()                   # before the replacement changes it
        gained = replace(item, text, session_id)
        freed += gained
        report["replaced"].append(f"{n}: {what}")
        report["dropped"].append(f"{item.seg.id} -{gained} tok" if gained >= 0 else f"{item.seg.id} +{-gained} tok")
    if chosen:
        st = conv.compaction_state
        st["batches"] = 0
        st.pop("new_tokens", None)
        st.pop("review_due", None)
        st["last_reason"] = "agent review"
        st.setdefault("agent_reviews", []).append({"batch": st.get("batch_clock", 0), "freed": freed,
                                                   "dropped": len(chosen)})
        del st["agent_reviews"][:-50]
    report.update({"freed": freed, "before_tokens": before, "after_tokens": conv.total_tokens() + tool_overhead})
    return report


def report_text(report: dict) -> str:
    if not report["replaced"] and not report["problems"]:
        return "curate_context: nothing replaced (no reference/replacement pairs given)."
    lines = []
    if report["replaced"]:
        lines.append("curate_context replaced:")
        lines += [f"  {r}" for r in report["replaced"]]
        lines.append('recall(segment_id) in a stub restores that original.')
    if report["problems"]:
        lines.append("not replaced: " + "; ".join(report["problems"]))
    return "\n".join(lines)
