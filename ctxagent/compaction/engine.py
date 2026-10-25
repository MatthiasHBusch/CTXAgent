"""The compaction orchestrator.

Runs after every agent turn. Escalates through tiers only as far as pressure
requires, cheapest first:

    tier 0  plan retiring prefix  free      cache safety and fresh-batch protection
    tier 1a supersession/dedup    free      the "re-read the same file" case
    tier 1b masking + offload     free      age-tiered observation decay
    tier 2  significance condense 1 cheap LLM call, only under pressure
    tier 3  deep compaction       1 LLM call, rare, rebuilds the prefix

Linked mode commits the reviewed prefix only AFTER tiers 1/2 on a due cycle.
The legacy freeze-first path remains available for saved chats and comparisons.

Two invariants hold at all times:

 I1  Segments before `conv.frozen_upto` have their final form; subsequent
     incremental passes are byte-identical. Tiers 1 and 2 cannot touch them. This is what keeps
     the prompt cache hot -- the expensive mistake TokenPilot documents.

 I2  The assistant(tool_calls) -> tool(tool_call_id) pairing is never broken.
     Tiers 1-2 rewrite bodies only. Tier 3 deletes whole runs, and its cut point
     is snapped to a validated boundary before anything is removed.
"""

from __future__ import annotations

from ..segments import K_MEDIA, S_ORIGINAL, S_OFFLOADED

import copy
import time
from typing import Any

from ..segments import (
    Conversation,
    Segment,
    K_SYSTEM,
    K_USER,
    K_ASSISTANT,
    K_TOOL_CALL,
    K_TOOL_RESULT,
    K_SUMMARY,
)
from .. import tokens
from . import dedup, masking, significance, summarize, deep_plan, freeze_review


class CompactionReport(dict):
    """Plain dict so it serialises straight to the UI event stream."""


def _pressure(total: int, effective_window: int) -> float:
    return total / max(1, effective_window)


def dynamic_tail_ratio(pressure: float, cfg) -> float:
    """How much of history stays verbatim in a deep compaction.

    Linearly interpolates the verbatim tail from `tail_ratio_max` at the trigger
    point down to `tail_ratio_min` at `max_pressure`. Equivalently: we compact
    between 75% and 90% of history, harder the tighter things get, and never all
    of it -- the newest work always survives untouched.
    """
    lo, hi = cfg.deep_trigger_ratio, cfg.max_pressure
    if hi <= lo:
        return cfg.tail_ratio_min
    t = (pressure - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    return cfg.tail_ratio_max + t * (cfg.tail_ratio_min - cfg.tail_ratio_max)


def _snapshot(conv: Conversation) -> list:
    """Exact pre-cycle state of every mutable (non-frozen) segment, plus each one's
    prompt tokens/bytes in order, so a rejected soft cycle can be undone byte-exactly."""
    return [(seg, copy.deepcopy(seg.__dict__), seg.tokens(), seg.body)
            for seg in conv.segments if not seg.frozen]


def _restore(snapshot: list) -> None:
    for seg, state, _tok, _body in snapshot:
        seg.__dict__.clear()
        seg.__dict__.update(state)
        seg.invalidate()


def _rewrite_verdict(conv: Conversation, snapshot: list, freed: int, clock: int, cfg) -> dict:
    """Keep a soft rewrite only if the per-request saving, over the expected remaining
    requests, outweighs re-sending the invalidated suffix uncached.

    Invalidated = old prompt tokens from the first changed segment to the end (that
    is what the provider has cached and must now re-write)."""
    first = next((i for i, (seg, _st, _tok, body) in enumerate(snapshot) if seg.body != body), None)
    invalidated = sum(tok for _seg, _st, tok, _body in snapshot[first:]) if first is not None else 0
    horizon = max(cfg.rewrite_horizon_min, min(cfg.rewrite_horizon_max, clock))
    benefit = freed * horizon
    cost = invalidated * (cfg.rewrite_cache_cost_ratio - 1.0)
    return {'keep': first is None or benefit >= cost, 'freed': freed, 'invalidated': invalidated,
            'horizon': horizon, 'benefit': benefit, 'cost': round(cost)}


def _snap_cut(conv: Conversation, idx: int) -> int:
    """Move a proposed cut index to a safe, semantically sensible boundary.

    Safety: never between an assistant(tool_calls) and its tool results, or the
    API rejects the request. Quality: prefer a user message, which marks a task
    seam -- cutting there is where a summary loses least.
    """
    segs = conv.segments
    n = len(segs)
    idx = max(0, min(idx, n))

    # Prefer the nearest user boundary within a modest search radius.
    radius = max(4, n // 12)
    best = None
    for d in range(0, radius + 1):
        for cand in (idx - d, idx + d):
            if 0 <= cand < n and segs[cand].kind == K_USER:
                best = cand
                break
        if best is not None:
            break
    if best is not None:
        idx = best

    # Safety: walk forward past any tool results whose call sits before the cut. The results
    # need not be adjacent to the call: a tool may write other segments while its batch is in
    # flight (document_close appends a user-kind handoff note before its own result), and the
    # user-boundary preference above can land exactly on such a note. Every result of a call
    # that stays before the cut must stay before it too, or the provider rejects the request
    # ("No tool call found for function call output").
    open_ids: set[str] = set()
    for s in segs[:idx]:
        if s.kind == K_TOOL_CALL:
            for tc in s.tool_calls or []:
                open_ids.add(tc.get("id") or "")
        elif s.kind == K_TOOL_RESULT and s.tool_call_id in open_ids:
            open_ids.discard(s.tool_call_id)
    if open_ids:
        for j in range(idx, n):
            if segs[j].kind == K_TOOL_RESULT and segs[j].tool_call_id in open_ids:
                open_ids.discard(segs[j].tool_call_id)
                idx = j + 1
                if not open_ids:
                    break
    # Never cut immediately before an orphan tool result.
    while idx < n and segs[idx].kind == K_TOOL_RESULT:
        idx += 1
    return idx


class CompactionEngine:
    def __init__(self, session_id: str, cfg, complete_fn):
        self.session_id = session_id
        self.cfg = cfg
        self.complete_fn = complete_fn
        self.turns_since_deep = 10_000
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    async def run(
        self,
        conv: Conversation,
        effective_window: int,
        *,
        tool_overhead: int = 0,
        force_deep: bool = False,
        instructions: str = "",
        completed_batch: bool = True,
    ) -> CompactionReport:
        cfg = self.cfg
        t0 = time.time()
        before_total = conv.total_tokens() + tool_overhead
        report = CompactionReport(
            ts=t0,
            before=before_total,
            after=before_total,
            freed=0,
            tiers=[],
            deep=False,
            window=effective_window,
        )
        schedule = conv.compaction_state
        self.turns_since_deep = schedule.get('turns_since_deep', self.turns_since_deep) + int(completed_batch)
        schedule['turns_since_deep'] = self.turns_since_deep
        # Monotonic tool-batch clock (never reset; survives deep compaction). Timed
        # pins from the `pin` tool expire against it.
        if completed_batch:
            schedule['batch_clock'] = schedule.get('batch_clock', 0) + 1
        clock = schedule.get('batch_clock', 0)
        expired = 0
        for seg in conv.segments:
            if seg.pinned and seg.pin_until_batch is not None and clock > seg.pin_until_batch:
                seg.pinned = False
                seg.pin_until_batch = None
                seg.pin_scope = 'manual'
                seg.note = 'timed pin expired'
                expired += 1
        if expired:
            report['pins_expired'] = expired

        if not cfg.enabled and not force_deep:
            report["after"] = before_total
            return report

        # Legacy policy is preserved for chats that have not opted in. In the
        # linked policy, nothing newly freezes until the due review has finished.
        review = freeze_review.plan(conv, cfg) if cfg.review_before_freeze else None
        if review is not None:
            hot_start = conv.frozen_upto
            eligible_ids = review['eligible_ids']
            newly_frozen = 0
        else:
            hot_start = self._respect_protected(conv, conv.compute_hot_start(cfg.hot_zone_tokens))
            newly_frozen = conv.freeze_prefix(hot_start)
            eligible_ids = None
        report['hot_start'] = hot_start
        report['frozen_segments'] = conv.frozen_upto
        report['newly_frozen'] = newly_frozen
        report['frozen_tokens'] = sum(s.tokens() for s in conv.segments[:conv.frozen_upto])

        # Always observe pressure/counters; defer BOTH review and freezing until
        # a cycle is due. This does not defer persistence or pairing repair.
        if completed_batch:
            schedule['batches'] = schedule.get('batches', 0) + 1
        last_seq = schedule.get('seen_seq', -1)
        added = sum(seg.tokens() for seg in conv if seg.seq > last_seq)
        schedule['new_tokens'] = schedule.get('new_tokens', 0) + added
        schedule['seen_seq'] = max((seg.seq for seg in conv), default=last_seq)
        pressure = _pressure(before_total, effective_window)
        threshold = cfg.rewrite_pressure_ratio
        if cfg.deep_enabled:
            threshold = min(threshold, cfg.deep_trigger_ratio)
        if cfg.document_compaction_enabled and cfg.significance_enabled:
            threshold = min(threshold, cfg.deep_trigger_ratio * 0.8)
        reason = ('forced' if force_deep else
                  'pressure' if pressure >= threshold else
                  'large new context' if schedule['new_tokens'] >= cfg.rewrite_max_new_tokens else
                  'batch interval' if schedule.get('batches', 0) >= cfg.rewrite_interval_batches else None)
        # Agent-curated policy: soft triggers only HINT (compaction/curate.py). The
        # automatic cycle remains the safety net at deep pressure / long neglect.
        if (getattr(cfg, 'rewrite_policy', 'auto') == 'agent' and reason is not None
                and reason != 'forced' and pressure < cfg.deep_trigger_ratio):
            if schedule.get('batches', 0) >= cfg.agent_review_overdue_batches:
                reason = 'agent review overdue'
            else:
                from . import curate
                schedule.setdefault('review_due', {'since_batch': schedule.get('batch_clock', 0), 'reason': reason})
                last_hint = schedule.get('review_hint_batch')
                hint = None
                if completed_batch and (last_hint is None or clock - last_hint >= cfg.rewrite_interval_batches):
                    hint = curate.hint_text(conv, cfg, schedule.get('batches', 0), reason)
                    segs = conv.segments
                    last_call = max((i for i, s in enumerate(segs) if s.kind == K_TOOL_CALL), default=None)
                    target = next((s for s in reversed(segs[last_call:] if last_call is not None else [])
                                   if s.kind == K_TOOL_RESULT and not s.frozen), None)
                    if target is not None:
                        # Not yet sent to the model: appending is cache-safe.
                        target.body = (target.body or '') + '\n\n' + hint
                        target.invalidate()
                        schedule['review_hint_batch'] = clock
                    else:
                        hint = None
                report['schedule'] = {'reason': 'agent review due (' + reason + ')',
                                      'pending_batches': schedule.get('batches', 0),
                                      'new_tokens': schedule['new_tokens'], 'interval': cfg.rewrite_interval_batches,
                                      'overdue_at': cfg.agent_review_overdue_batches}
                if hint:
                    report['review_hint'] = hint
                report['pressure'] = round(pressure, 4)
                report['duration_ms'] = int((time.time() - t0) * 1000)
                return report
        report['schedule'] = {'reason': reason or 'deferred', 'pending_batches': schedule.get('batches', 0),
                              'new_tokens': schedule['new_tokens'], 'interval': cfg.rewrite_interval_batches}
        report['pressure'] = round(pressure, 4)
        if reason is None:
            report['duration_ms'] = int((time.time() - t0) * 1000)
            return report

        # Soft cycles must pay for the cache they invalidate (see rewrite_cost_gate).
        gate_snapshot = (_snapshot(conv) if cfg.rewrite_cost_gate
                         and reason in ('batch interval', 'large new context') else None)

        # Keep two recent unpinned visual observations; immutable PNGs remain on disk.
        media = [s for s in conv if s.kind == K_MEDIA and not s.pinned]
        # Never discard unseen images from the just-completed tool batch.
        last_batch = max((i for i, s in enumerate(conv.segments) if s.kind == K_TOOL_CALL), default=len(conv.segments))
        fresh_ids = {s.id for s in conv.segments[last_batch:] if s.kind == K_MEDIA}
        for seg in media[:-2]:
            if (not seg.frozen and seg.state == S_ORIGINAL and seg.id not in fresh_ids
                    and (eligible_ids is None or seg.id in eligible_ids)):
                gain = seg.rewrite(
                    f'[visual evidence archived; recall(segment_id="{seg.id}") to see images again]',
                    S_OFFLOADED, 'older visual evidence')
                report['freed'] += gain
                report['tiers'].append({'tier': 'media', 'freed': gain})

        # ---- significance is needed by several tiers --------------------
        if cfg.document_compaction_enabled and cfg.significance_enabled:
            significance.score_all(conv, hot_start)

        # ---- tier 1a: supersession / dedup ------------------------------
        if cfg.document_compaction_enabled and cfg.dedup_enabled:
            r = dedup.run(conv, hot_start, eligible_ids=eligible_ids)
            if r["freed"]:
                report["tiers"].append({"tier": "1a-dedup", **r})
                report["freed"] += r["freed"]

        # ---- tier 1b: masking + offload ---------------------------------
        if cfg.document_compaction_enabled and cfg.masking_enabled:
            r = masking.run(conv, hot_start, cfg, self.session_id, eligible_ids=eligible_ids)
            if r["freed"]:
                report["tiers"].append({"tier": "1b-mask", **r})
                report["freed"] += r["freed"]

        if gate_snapshot is not None and report['freed'] > 0:
            verdict = _rewrite_verdict(conv, gate_snapshot, report['freed'], clock, cfg)
            if not verdict['keep']:
                _restore(gate_snapshot)
                report['freed'] = 0
                report['tiers'] = []
            report['cost_gate'] = verdict

        total = conv.total_tokens() + tool_overhead
        pressure = _pressure(total, effective_window)
        report["pressure"] = round(pressure, 4)

        # ---- tier 2: significance-driven condensation -------------------
        # Only under real pressure, and only if the pass would pay for itself.
        if (
            cfg.document_compaction_enabled and cfg.significance_enabled
            and not force_deep
            and pressure >= cfg.deep_trigger_ratio * 0.8
        ):
            cands = significance.rank_for_condensation(
                conv, hot_start, cfg.significance_threshold
            )
            if eligible_ids is not None:
                cands = [s for s in cands if s.id in eligible_ids]
            potential = sum(s.tokens() for s in cands)
            if potential >= cfg.min_gain_tokens and cands:
                batch = cands[:14]
                r = await summarize.condense_batch(
                    batch, cfg.summary_model, self.complete_fn
                )
                if r.get("freed"):
                    report["tiers"].append({"tier": "2-condense", **r})
                    report["freed"] += r["freed"]
                    total = conv.total_tokens() + tool_overhead
                    pressure = _pressure(total, effective_window)
                    report["pressure"] = round(pressure, 4)

        # ---- review commit: cleaned bytes become the immutable prefix ---
        if review is not None:
            # Pinned/fresh material is intentionally retained, not silently
            # masked to hit a target. Only the prefix checked above is promoted.
            newly_frozen = conv.freeze_prefix(review['end'])
            schedule['reviewed_through_seq'] = max(
                (s.seq for s in conv.segments[:conv.frozen_upto]), default=-1)
            report['freeze_review'] = {
                'checked_segments': len(eligible_ids),
                'protected_observations': len(review['protected_ids']),
                'start': review['start'], 'end': conv.frozen_upto,
                'reviewed_through_seq': schedule['reviewed_through_seq'],
            }
            report['hot_start'] = conv.frozen_upto
            report['frozen_segments'] = conv.frozen_upto
            report['newly_frozen'] = newly_frozen
            report['frozen_tokens'] = sum(s.tokens() for s in conv.segments[:conv.frozen_upto])

        schedule['batches'] = schedule['new_tokens'] = 0
        schedule['last_reason'] = reason

        # ---- tier 3: deep compaction ------------------------------------
        want_deep = force_deep or (
            cfg.deep_enabled
            and pressure >= cfg.deep_trigger_ratio
            and self.turns_since_deep >= cfg.deep_cooldown_turns
        )
        if want_deep:
            r = await self._deep(
                conv, effective_window, pressure, tool_overhead, instructions, force_deep
            )
            if r:
                report["tiers"].append(r)
                report["deep"] = not bool(r.get("error"))
                report["freed"] += r.get("freed", 0)

        report["after"] = conv.total_tokens() + tool_overhead
        report["duration_ms"] = int((time.time() - t0) * 1000)
        if report['deep']:
            report['hot_start'] = report['frozen_segments'] = conv.frozen_upto
            report['frozen_tokens'] = sum(s.tokens() for s in conv.segments[:conv.frozen_upto])
            if review is not None:
                report['freeze_review']['deep_reset'] = True
        if (report['tiers'] or report.get('cost_gate')
                or report.get('freeze_review', {}).get('checked_segments')):
            self.history.append(dict(report))
        return report

    # ------------------------------------------------------------------
    def _respect_protected(self, conv: Conversation, hot_start: int) -> int:
        """Pull hot_start back so recent user turns stay rewritable-protected.

        The last N user turns must lie inside the hot zone *and* be marked
        protected so no tier touches them (OpenCode protects the last 2).
        """
        # Recent-user protection is evaluated transiently in deep_plan. Never
        # permanently pin every user message just because it was once recent.
        return hot_start

    # ------------------------------------------------------------------
    async def _deep(
        self,
        conv: Conversation,
        effective_window: int,
        pressure: float,
        tool_overhead: int,
        instructions: str,
        force: bool,
    ) -> dict | None:
        cfg = self.cfg
        conv.migrate_legacy_recall_pins()
        plan = deep_plan.plan(conv, cfg, tool_overhead)
        victims, retained = plan['victims'], plan['retained']
        freed_estimate = sum(s.tokens() for s in victims)
        if not victims or (not force and freed_estimate < cfg.deep_min_gain_tokens):
            return None
        prior = next((s.body for s in reversed(victims) if s.kind == K_SUMMARY), None)
        try:
            text = await summarize.deep_summary(
                victims, prior, cfg.summary_model, self.complete_fn,
                max_tokens=cfg.summary_max_tokens, extra_instructions=instructions,
                content_token_budget=plan['summary_budget'])
        except Exception as e:
            return {"tier": "3-deep", "error": str(e), "freed": 0}
        if not text:
            return {"tier": "3-deep", "error": "empty summary", "freed": 0}
        # Do not trust a model to obey the size contract. Bound its text locally
        # without a second billed request; all source segments remain recallable.
        clipped = tokens.count(text) > plan['summary_budget']
        if clipped:
            marker = "\n[Summary shortened to budget; original segments remain in the archive.]"
            budget = plan['summary_budget'] - tokens.count(marker) - 5
            text = tokens.truncate_to_tokens(text, budget, head_ratio=0.8) + marker
            while tokens.count(text) > plan['summary_budget']:
                text = text[:int(len(text) * .95)]
        wrapped = summarize.wrap_summary(text, len(victims), freed_estimate)
        summary_seg = Segment(seq=victims[0].seq, role="user", kind=K_SUMMARY,
                              body=wrapped, pinned=True)
        after = tool_overhead + 3 + sum(s.tokens() for s in retained) + summary_seg.tokens()
        if after >= plan['before']:
            return {"tier": "3-deep", "error": "Summary would not reduce context; original context retained",
                    "freed": 0, "target_tokens": plan['target_tokens'], "fixed_tokens": plan['fixed_tokens']}
        conv.archive_segments(victims, self.session_id)
        head = 0
        while head < len(retained) and retained[head].kind == K_SYSTEM:
            head += 1
        conv.segments = retained[:head] + [summary_seg] + retained[head:]
        for s in conv:
            s.frozen = False
        conv.frozen_upto = 0
        conv.freeze_prefix(head + 1)
        if cfg.review_before_freeze:
            # The new summary was checked for emptiness, budget and net gain.
            conv.compaction_state['reviewed_through_seq'] = summary_seg.seq
        self.turns_since_deep = 0
        conv.compaction_state['turns_since_deep'] = 0
        blocked = after > plan['target_tokens']
        warning = ("Target limited by system/tool overhead, explicit or unclassified legacy pins, "
                   "recent user messages, minimum summary budget and/or the newest indivisible tool group." if blocked else None)
        return {
            "tier": "3-deep", "freed": plan['before'] - after,
            "compacted_segments": len(victims), "compacted_tokens": freed_estimate,
            "summary_tokens": summary_seg.tokens(), "summary_budget": plan['summary_budget'],
            "summary_clipped": clipped, "kept_verbatim": len(retained),
            "pinned_retained": sum(len(g) for g in deep_plan.groups(retained)
                                   if any(s.pinned and s.kind != K_SYSTEM and s.pin_scope != 'recall' for s in g)),
            "archived": len(victims), "nested_prior": bool(prior), "summary_id": summary_seg.id,
            "target_ratio": cfg.deep_target_ratio, "target_tokens": plan['target_tokens'],
            "actual_ratio": after / plan['before'], "fixed_tokens": plan['fixed_tokens'],
            "target_met": not blocked, "warning": warning,
            "tail_ratio": sum(s.tokens() for s in retained if s.kind != K_SYSTEM) / max(1, plan['before']),
        }

    # ------------------------------------------------------------------
    def stats(self, conv: Conversation, effective_window: int, tool_overhead: int = 0) -> dict:
        """Snapshot for the context-inspector panel."""
        cfg = self.cfg
        hot_start = conv.frozen_upto  # actual boundary, not the prospective next freeze
        frozen_tokens = sum(s.tokens() for s in conv.segments[:hot_start])
        hot_tokens = sum(s.tokens() for s in conv.segments[hot_start:])
        total = conv.total_tokens() + tool_overhead
        by_state: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        for s in conv.segments:
            by_state[s.state] = by_state.get(s.state, 0) + s.tokens()
            by_kind[s.kind] = by_kind.get(s.kind, 0) + s.tokens()
        pressure = _pressure(total, effective_window)
        review = freeze_review.plan(conv, cfg) if cfg.review_before_freeze else None
        candidate = sum(s.tokens() for s in conv.segments[hot_start:review['end']]) if review else 0
        return {
            "total": total,
            "original": conv.original_tokens() + tool_overhead,
            "saved": max(0, conv.original_tokens() - conv.total_tokens()),
            "effective_window": effective_window,
            "pressure": round(pressure, 4),
            "frozen_tokens": frozen_tokens,
            "hot_tokens": hot_tokens,
            "hot_zone_budget": cfg.hot_zone_tokens,
            "hot_excess_tokens": max(0, hot_tokens - cfg.hot_zone_tokens),
            "review_candidate_tokens": candidate,
            "review_before_freeze": cfg.review_before_freeze,
            "initial_result_max_tokens": cfg.initial_result_max_tokens,
            "reviewed_through_seq": conv.compaction_state.get('reviewed_through_seq'),
            "rewrite_interval_batches": cfg.rewrite_interval_batches,
            "rewrite_pending_batches": conv.compaction_state.get('batches', 0),
            "rewrite_new_tokens": conv.compaction_state.get('new_tokens', 0),
            "rewrite_last_reason": conv.compaction_state.get('last_reason'),
            "hot_start": hot_start,
            "tool_overhead": tool_overhead,
            "segments": len(conv.segments),
            "by_state": by_state,
            "by_kind": by_kind,
            "significance": significance.distribution(conv),
            "next_tail_ratio": cfg.deep_target_ratio,
            "deep_target_ratio": cfg.deep_target_ratio,
            "document_compaction_enabled": cfg.document_compaction_enabled,
            "turns_since_deep": self.turns_since_deep,
            "deep_ready": self.turns_since_deep >= cfg.deep_cooldown_turns,
        }
