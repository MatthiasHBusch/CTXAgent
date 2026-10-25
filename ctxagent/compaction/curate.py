"""Agent-curated incremental review (the `curate_context` tool, rewrite_policy='agent').

The automatic due cycle masks tool results by AGE RANK. Age is a poor proxy for
relevance: a 3 kB grep that located the bug stays load-bearing for an hour, a
40 kB build log is dead the moment the agent has read "BUILD_OK". Only the agent
knows which is which. With ``rewrite_policy='agent'`` the engine therefore stops
masking on the interval / new-token / soft-pressure triggers and instead appends
a one-line HINT to the newest tool result. The agent then

    curate_context()                                   # list the unreviewed results
    curate_context(drop=["id1", {"id": "id2", "note": "BUILD_OK, 0 warnings"}])

* `drop`: the full body goes to a blob (lossless; recall restores it) and the
  visible stub becomes the agent's own note plus the result's first line - a
  better stub than any truncation, because the author knows the takeaway.
* Everything not named is KEPT verbatim (``rest='keep'``, default) - nothing is
  removed by accident. ``rest='auto'`` additionally runs the age-rank masking
  over the remaining unpinned results of the reviewed region.
* ``commit`` (default true) promotes the reviewed region to the frozen prefix
  (the same boundary the automatic review would freeze) and resets the
  due-cycle counters, so no hint/auto cycle follows right after.
* ``dry_run`` reports the effect and the cache break-even without changing
  anything: dropping a result invalidates the provider cache from its position
  on, so a small drop far back can cost more than it saves.

Safety nets stay automatic (engine.py): pressure >= deep_trigger_ratio, or no
review for ``agent_review_overdue_batches`` batches, run the legacy cycle.
"""
from __future__ import annotations

import json

from ..segments import Conversation, K_TOOL_CALL, K_TOOL_RESULT, S_OFFLOADED, S_ORIGINAL
from ..blobs import put_blob
from .. import tokens
from . import freeze_review, masking

NOTE_MAX_CHARS = 400
LIST_MAX_ROWS = 80


def _last_call_index(conv: Conversation) -> int:
    return max((i for i, s in enumerate(conv.segments) if s.kind == K_TOOL_CALL), default=len(conv.segments))


def _call_args(conv: Conversation, seg) -> str:
    """Short rendering of the arguments of the call that produced `seg`."""
    for s in conv.segments:
        if s.kind != K_TOOL_CALL:
            continue
        for tc in s.tool_calls or []:
            if tc.get("id") == seg.tool_call_id:
                raw = (tc.get("function") or {}).get("arguments") or ""
                try:
                    obj = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(obj, dict):
                        raw = " ".join(f"{k}={str(v)[:60]}" for k, v in obj.items()
                                       if k not in ("content", "new_string", "old_string", "summary"))
                except (ValueError, TypeError):
                    pass
                return str(raw).replace("\n", " ")[:110]
    return ""


def candidates(conv: Conversation) -> list:
    """Unfrozen tool results the agent has already seen (excludes the running batch)."""
    cut = _last_call_index(conv)
    return [s for i, s in enumerate(conv.segments)
            if i < cut and i >= conv.frozen_upto and s.kind == K_TOOL_RESULT and not s.frozen]


def unreviewed_stats(conv: Conversation) -> tuple[int, int]:
    c = candidates(conv)
    return len(c), sum(s.tokens() for s in c)


def hint_text(conv: Conversation, cfg, batches_pending: int, reason: str) -> str:
    n, tok = unreviewed_stats(conv)
    left = max(0, cfg.agent_review_overdue_batches - batches_pending)
    return (f"[harness: context review due ({reason}) - {n} unfrozen tool results, ~{tok} tok. "
            "At the next boundary (when the current step is done), call curate_context() to list them and "
            "drop what you no longer need, each with a one-line takeaway; unnamed results are kept verbatim. "
            "If the work before this point is finished and the conversation has moved on, "
            "compact_context(summary=...) is the better tool. "
            f"Automatic age-based masking runs if no review happens within {left} batches.]")


def listing(conv: Conversation, cfg, min_tokens: int = 0) -> str:
    rows = []
    cands = candidates(conv)
    shown = [s for s in cands if s.tokens() >= min_tokens]
    review = freeze_review.plan(conv, cfg)
    commit_ids = review["eligible_ids"]
    for s in shown[-LIST_MAX_ROWS:]:
        flags = []
        if s.pinned:
            flags.append("pinned")
        if s.state != S_ORIGINAL:
            flags.append(s.state)
        if s.id in commit_ids:
            flags.append("freezes on commit")
        first = (s.body or "").strip().split("\n", 1)[0].replace("\r", "")[:90]
        rows.append(f"{s.id}  {s.tool_name or '?'}  {s.tokens()} tok"
                    + (f"  [{', '.join(flags)}]" if flags else "")
                    + f"\n    args: {_call_args(conv, s)}\n    first: {first}")
    total = sum(s.tokens() for s in cands)
    hidden = len(cands) - len(shown[-LIST_MAX_ROWS:])
    head = (f"{len(cands)} unfrozen tool results, ~{total} tok (oldest first"
            + (f"; {hidden} smaller/older not shown" if hidden else "") + ").\n"
            "Drop with curate_context(drop=[id or {id, note}]); the note becomes the visible stub, "
            "the full text stays recallable. Unnamed results are kept.\n")
    return head + "\n".join(rows)


def _parse_drop(drop) -> list[tuple[str, str]]:
    out = []
    for item in drop or []:
        if isinstance(item, dict):
            sid, note = str(item.get("id", "")).strip(), str(item.get("note", "") or "").strip()
        else:
            text = str(item).strip()
            sid, _, note = text.partition(":")
            sid, note = sid.strip(), note.strip()
        if sid:
            out.append((sid, note[:NOTE_MAX_CHARS]))
    return out


def curate(conv: Conversation, cfg, session_id: str, *, drop=None, rest: str = "keep",
           commit: bool = True, dry_run: bool = False, tool_overhead: int = 0) -> dict:
    if rest not in ("keep", "auto"):
        raise ValueError("rest must be 'keep' or 'auto'")
    segs = conv.segments
    cand_ids = {s.id for s in candidates(conv)}
    before = conv.total_tokens() + tool_overhead
    todo, problems = [], []
    for sid, note in _parse_drop(drop):
        seg = conv.by_id(sid)
        if seg is None:
            problems.append(f"{sid}: not an active segment (archived ids are already gone)")
        elif seg.kind != K_TOOL_RESULT:
            problems.append(f"{sid}: not a tool result")
        elif seg.frozen:
            problems.append(f"{sid}: frozen (immutable cached prefix; only compact_context removes it)")
        elif sid not in cand_ids:
            problems.append(f"{sid}: belongs to the running batch")
        elif seg.state == S_OFFLOADED and seg.note.startswith("dropped by agent"):
            problems.append(f"{sid}: already dropped")
        else:
            todo.append((seg, note))

    def stub(seg, note) -> str:
        full = seg.original_body if seg.original_body is not None else seg.body
        first = full.strip().split("\n", 1)[0].replace("\r", "")[:160]
        return (f"[dropped by agent: {note or 'no longer needed'} | was {seg.original_tokens or seg.tokens()} tok, "
                f"first line: {first} | recall(segment_id=\"{seg.id}\") restores it exactly]")

    freed_est = sum(max(0, s.tokens() - tokens.count(stub(s, n))) for s, n in todo)
    index = {s.id: i for i, s in enumerate(segs)}
    first_change = min((index[s.id] for s, _ in todo), default=None)
    report = {"dry_run": bool(dry_run), "dropped": [], "problems": problems, "rest": rest}

    if dry_run:
        suffix = sum(s.tokens() for s in segs[first_change:]) if first_change is not None else 0
        report.update({"freed_estimate": freed_est, "would_drop": [s.id for s, _ in todo],
                       "cache_rewrite_tokens": suffix,
                       "cache_breakeven_requests": round(1.15 * suffix / (0.1 * freed_est), 1) if freed_est else None,
                       "before_tokens": before})
        return report

    freed = 0
    for seg, note in todo:
        full = seg.original_body if seg.original_body is not None else seg.body
        seg.blob_id = seg.blob_id or put_blob(session_id, seg.id, full)
        if seg.pinned:  # an explicit drop overrides the agent's own earlier pin
            seg.pinned = False
            seg.pin_until_batch = None
            seg.pin_scope = "manual"
        gained = seg.rewrite(stub(seg, note), S_OFFLOADED, "dropped by agent")
        freed += gained
        report["dropped"].append(f"{seg.id} -{gained} tok")

    review = freeze_review.plan(conv, cfg)
    if rest == "auto":
        r = masking.run(conv, review["start"], cfg, session_id, eligible_ids=review["eligible_ids"])
        freed += r["freed"]
        report["auto_masked"] = r
    if first_change is None and report.get("auto_masked", {}).get("freed"):
        first_change = min((i for i, s in enumerate(segs) if s.id in review["eligible_ids"]
                            and s.state != S_ORIGINAL), default=None)

    st = conv.compaction_state
    if commit:
        newly = conv.freeze_prefix(review["end"])
        st["reviewed_through_seq"] = max((s.seq for s in conv.segments[:conv.frozen_upto]), default=-1)
        st["batches"] = st["new_tokens"] = 0
        st.pop("review_due", None)
        st["last_reason"] = "agent review"
        report["newly_frozen"] = newly
        report["frozen_segments"] = conv.frozen_upto
    after = conv.total_tokens() + tool_overhead
    suffix = sum(s.tokens() for s in segs[first_change:]) if first_change is not None else 0
    report.update({"freed": freed, "before_tokens": before, "after_tokens": after,
                   "cache_rewrite_tokens": suffix,
                   "cache_breakeven_requests": round(1.15 * suffix / (0.1 * freed), 1) if freed else None})
    st.setdefault("agent_reviews", []).append({"batch": st.get("batch_clock", 0), "freed": freed,
                                               "dropped": len(report["dropped"])})
    del st["agent_reviews"][:-50]
    return report
