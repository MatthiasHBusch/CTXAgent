"""Agent-curated incremental review (the `curate_context` tool, rewrite_policy='agent').

Only the agent knows which tool results are still load-bearing: a 3 kB grep that located
the bug stays useful for an hour, a 40 kB build log is dead the moment the agent has read
"BUILD_OK". With ``rewrite_policy='agent'`` the engine appends a one-line HINT to the
newest tool result when a review is due (interval, large new context, pressure), and the
agent then

    curate_context()                                   # list the unreviewed results
    curate_context(drop=["id1", {"id": "id2", "note": "BUILD_OK, 0 warnings"}])

* `drop`: the full body goes to a blob (lossless; recall restores it) and the
  visible stub becomes the agent's own note plus the result's first line - a
  better stub than any truncation, because the author knows the takeaway.
* Everything not named is kept verbatim - nothing is removed by accident.
* ``commit`` (default true) promotes the reviewed region to the frozen prefix and resets
  the review counter, so no hint follows right after.
* ``dry_run`` reports the effect and the cache break-even without changing anything.

Cost (economics.py): a drop re-sends everything after the last cache breakpoint before
it (with explicit breakpoints: after the frozen prefix) at the cache-write price. Every
result reports the break-even.

The safety net stays automatic (engine.py): deep compaction at deep_trigger_ratio.
"""
from __future__ import annotations

import json

from ..segments import Conversation, K_TOOL_CALL, K_TOOL_RESULT, S_OFFLOADED, S_ORIGINAL
from ..blobs import put_blob
from .. import tokens
from . import economics, freeze_review

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
    # The answer count is the fact the default guidance's cadence refers to; the frozen
    # benchmark text ('live') keeps the hint it ran with.
    since = ("" if getattr(cfg, 'guidance', 'default') == 'live' else
             f"{batches_pending} answers since the context was last curated or compacted, ")
    return (f"[harness: context review due ({reason}) - {since}{n} unfrozen tool results, ~{tok} tok. "
            "At the next boundary (when the current step is done), call curate_context() to list them and "
            "drop what you no longer need, each with a one-line takeaway; unnamed results are kept verbatim. "
            "If the work before this point is finished and the conversation has moved on, "
            "compact_context(summary=...) is the better tool.]")


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


def _stub(seg, note) -> str:
    full = seg.original_body if seg.original_body is not None else seg.body
    first = full.strip().split("\n", 1)[0].replace("\r", "")[:160]
    return (f"[dropped by agent: {note or 'no longer needed'} | was {seg.original_tokens or seg.tokens()} tok, "
            f"first line: {first} | recall(segment_id=\"{seg.id}\") restores it exactly]")


def _droppable(conv: Conversation, seg, cand_ids: set) -> str | None:
    if seg is None:
        return "not an active segment (archived ids are already gone)"
    if seg.kind != K_TOOL_RESULT:
        return "not a tool result"
    if seg.frozen:
        return "frozen (immutable cached prefix; only compact_context removes it)"
    if seg.id not in cand_ids:
        return "belongs to the running batch"
    if seg.state == S_OFFLOADED and seg.note.startswith("dropped by agent"):
        return "already dropped"
    return None


def assess(conv: Conversation, cfg, todo: list[tuple], *, price: dict | None, family: str,
           tool_overhead: int = 0, calibration: float = 1.0) -> tuple[dict, int | None, int]:
    """(verdict, first changed index, estimated freed tokens) of dropping `todo`."""
    segs = conv.segments
    freed_est = sum(max(0, s.tokens() - tokens.count(_stub(s, n))) for s, n in todo)
    index = {s.id: i for i, s in enumerate(segs)}
    first_change = min((index[s.id] for s, _ in todo), default=None)
    total = conv.total_tokens() + tool_overhead
    invalidated = 0
    if first_change is not None:
        invalidated = max(0, total - economics.cached_boundary(conv, first_change, family, tool_overhead) - freed_est)
    price = price or economics.rates(None, cfg.rewrite_cache_cost_ratio)
    v = economics.verdict(freed_est, invalidated, price, economics.horizon(conv, cfg), calibration)
    return v, first_change, freed_est


def apply_drops(conv: Conversation, session_id: str, todo: list[tuple]) -> tuple[int, list[str]]:
    freed, done = 0, []
    for seg, note in todo:
        full = seg.original_body if seg.original_body is not None else seg.body
        seg.blob_id = seg.blob_id or put_blob(session_id, seg.id, full)
        if seg.pinned:  # an explicit drop overrides the agent's own earlier pin
            seg.pinned = False
            seg.pin_until_batch = None
            seg.pin_scope = "manual"
        gained = seg.rewrite(_stub(seg, note), S_OFFLOADED, "dropped by agent")
        freed += gained
        done.append(f"{seg.id} -{gained} tok")
    return freed, done


def curate(conv: Conversation, cfg, session_id: str, *, drop=None, rest: str = "keep",
           commit: bool = True, dry_run: bool = False, tool_overhead: int = 0,
           price: dict | None = None, family: str = "breakpoints",
           calibration: float = 1.0) -> dict:
    if rest not in ("keep", "auto"):
        raise ValueError("rest must be 'keep'")
    segs = conv.segments
    cand_ids = {s.id for s in candidates(conv)}
    before = conv.total_tokens() + tool_overhead
    todo, problems = [], []
    for sid, note in _parse_drop(drop):
        seg = conv.by_id(sid)
        why = _droppable(conv, seg, cand_ids)
        if why:
            problems.append(f"{sid}: {why}")
        else:
            todo.append((seg, note))
    v, first_change, freed_est = assess(conv, cfg, todo, price=price, family=family,
                                        tool_overhead=tool_overhead, calibration=calibration)
    suffix = sum(s.tokens() for s in segs[first_change:]) if first_change is not None else 0
    report = {"dry_run": bool(dry_run), "dropped": [], "problems": problems}
    if rest == "auto":
        # Chats started before the age-based masking was removed may still pass it.
        report["note"] = "rest='auto' no longer masks anything; results you do not name are kept verbatim"
    if dry_run:
        report.update({"freed_estimate": freed_est, "would_drop": [s.id for s, _ in todo],
                       "cache_rewrite_tokens": suffix, "invalidated_tokens": v["invalidated_tokens"],
                       "cache_breakeven_requests": v["breakeven_requests"],
                       "before_tokens": before})
        return report

    st = conv.compaction_state
    freed, done = apply_drops(conv, session_id, todo)
    report["dropped"] = done
    review = freeze_review.plan(conv, cfg)
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
                   "cache_breakeven_requests": v["breakeven_requests"] if todo else None})
    st.setdefault("agent_reviews", []).append({"batch": st.get("batch_clock", 0), "freed": freed,
                                               "dropped": len(report["dropped"])})
    del st["agent_reviews"][:-50]
    return report
