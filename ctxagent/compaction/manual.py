"""Agent-directed deep compaction (the `compact_context` tool).

The automatic tiers decide by age and pressure. They cannot know that a task is
DONE: when the conversation moves to a new topic, the whole earlier work (file
reads, logs, test output) is dead weight that every later request pays to read
from the cache (~0.1x input price per token per request). The agent is the one
party that knows where the topic boundary is and what of the old work still
matters, so this tool lets it say so:

    compact_context(summary="...", keep_from="last_user", keep_ids=[...])

* Everything between the system prompt and `keep_from` is ARCHIVED (losslessly:
  `recall(segment_id)` still returns any of it) and replaced by ONE summary
  segment whose text the agent wrote itself. No summariser call, no cost, and the
  author is the party that knows which facts are load-bearing.
* `keep_ids` (and every manual pin) stay verbatim, together with their whole tool
  group (a call never loses its results or vice versa).
* An index of the archived tool results (id, tool, first line) is appended to the
  summary (capped), so a later recall knows its handle.
* The cut never reaches into the tool batch that is running this call.
* `dry_run=true` reports the effect and the cache break-even without changing
  anything.

Cache economics (reported): after the rewrite the retained prefix is written to
the cache once (~1.25x input price) instead of read (~0.1x); every later request
then reads `before - after` fewer tokens. Break-even after
    n = 1.15 * after / (0.1 * (before - after))
requests - typically a handful when a large finished topic goes.
"""
from __future__ import annotations

import time

from ..segments import (Conversation, Segment, K_SYSTEM, K_USER, K_SUMMARY, K_TOOL_CALL,
                        K_TOOL_RESULT, K_MEDIA)
from .deep_plan import groups
from .. import tokens

MIN_SUMMARY_CHARS = 200
INDEX_MAX_ENTRIES = 60


def _cut_index(conv: Conversation, keep_from: str | None) -> tuple[int, str]:
    segs = conv.segments
    last_call = max((i for i, s in enumerate(segs) if s.kind == K_TOOL_CALL), default=len(segs))
    spec = (keep_from or "last_user").strip()
    if spec in ("last_user", ""):
        idx = max((i for i, s in enumerate(segs[:last_call]) if s.kind == K_USER), default=None)
        if idx is None:
            raise ValueError("no user message before the current batch to keep from")
        how = "the last user message"
    elif spec == "current_batch":
        idx, how = last_call, "the current tool batch"
    else:
        idx = next((i for i, s in enumerate(segs) if s.id == spec), None)
        if idx is None:
            raise ValueError(f"keep_from: no active segment {spec!r} (use an id from a recall "
                             "placeholder / context_report, 'last_user' or 'current_batch')")
        how = f"segment {spec}"
    # Never into the running batch; snap back to the start of the tool group it falls in.
    idx = min(idx, last_call)
    while 0 < idx < len(segs) and segs[idx].kind in (K_TOOL_RESULT, K_MEDIA):
        idx -= 1
    return idx, how


def _index_line(s: Segment) -> str:
    first = (s.original_body or s.body or "").strip().split("\n", 1)[0]
    first = first.replace("\r", "")[:110]
    return f"- {s.id}  {s.tool_name or s.kind}  ({s.original_tokens or s.tokens()} tok)  {first}"


def compact(conv: Conversation, session_id: str, *, summary: str, keep_from: str | None = None,
            keep_ids=None, dry_run: bool = False, index: bool = True,
            tool_overhead: int = 0) -> dict:
    summary = (summary or "").strip()
    if not dry_run and len(summary) < MIN_SUMMARY_CHARS:
        raise ValueError(f"summary too short ({len(summary)} chars, minimum {MIN_SUMMARY_CHARS}): "
                         "write the handoff - goals, decisions, changed files, open issues, ids to recall")
    segs = conv.segments
    head = 0
    while head < len(segs) and segs[head].kind == K_SYSTEM:
        head += 1
    cut, how = _cut_index(conv, keep_from)
    if cut <= head:
        raise ValueError("nothing to compact before the cut")

    keep = {str(x).strip() for x in (keep_ids or []) if str(x).strip()}
    region = segs[head:cut]
    kept_groups, victims = [], []
    missing = set(keep)
    for g in groups(region):
        ids = {s.id for s in g}
        missing -= ids
        keep_group = bool(ids & keep) or any(
            s.pinned and s.pin_scope == "manual" and s.kind != K_SUMMARY for s in g)
        (kept_groups if keep_group else victims).append(g)
    victim_segs = [s for g in victims for s in g]
    kept_segs = [s for g in kept_groups for s in g]
    if not victim_segs:
        raise ValueError("every segment before the cut is kept (keep_ids / pins): nothing to compact")

    before = conv.total_tokens() + tool_overhead
    removed = sum(s.tokens() for s in victim_segs)
    body = summary
    if index:
        results = [s for s in victim_segs if s.kind == K_TOOL_RESULT]
        lines = [_index_line(s) for s in results[-INDEX_MAX_ENTRIES:]]
        if lines:
            more = len(results) - len(lines)
            body += ("\n\nArchived tool results (recall(segment_id) restores any of them exactly"
                     + (f"; {more} older ones not listed" if more > 0 else "") + "):\n" + "\n".join(lines))
    wrapped = (f"[Context compacted by the agent at its own request: {len(victim_segs)} segments "
               f"(~{removed} tokens) before {how} archived and replaced by this handoff. "
               "Any archived segment id is still recallable.]\n\n" + body)
    summary_seg = Segment(seq=victim_segs[0].seq, role="user", kind=K_SUMMARY, body=wrapped, pinned=True)
    after = before - removed + summary_seg.tokens()
    saved = before - after
    breakeven = (1.15 * after / (0.1 * saved)) if saved > 0 else None
    report = {
        "dry_run": bool(dry_run), "cut": how, "compacted_segments": len(victim_segs),
        "compacted_tokens": removed, "kept_verbatim_before_cut": len(kept_segs),
        "summary_tokens": summary_seg.tokens(), "before_tokens": before, "after_tokens": after,
        "saved_tokens": saved,
        "cache_breakeven_requests": round(breakeven, 1) if breakeven is not None else None,
        "unknown_keep_ids": sorted(missing),
    }
    if dry_run:
        return report
    if saved <= 0:
        raise ValueError(f"the summary ({summary_seg.tokens()} tok) is not smaller than what it "
                         f"replaces ({removed} tok): nothing gained")
    conv.archive_segments(victim_segs, session_id)
    conv.segments = segs[:head] + [summary_seg] + kept_segs + segs[cut:]
    for s in conv.segments:
        s.frozen = False
    conv.frozen_upto = 0
    conv.freeze_prefix(head + 1)
    st = conv.compaction_state
    st["turns_since_deep"] = 0
    st["reviewed_through_seq"] = summary_seg.seq
    st.setdefault("manual_compactions", []).append({"ts": time.time(), "saved": saved,
                                                   "segments": len(victim_segs)})
    report["summary_id"] = summary_seg.id
    return report
