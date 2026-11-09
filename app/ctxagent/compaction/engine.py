"""The compaction orchestrator.

Runs after every tool batch and every finished answer. It compacts in one way:

    deep compaction   one summary call; the older history becomes a summary, the newest
                      tool group, recent user turns and pins stay verbatim, the originals
                      go to the archive (recall restores them)

It runs when the context reaches `deep_trigger_ratio` of the window (or when forced:
the Compact button, compact_now). Between deep compactions the engine only freezes the
older history (the cache-stable prefix). Self-steered chats (rewrite_policy='agent')
manage their context themselves with compact_context and curate_context; the engine
appends a review hint to the newest tool result when a review is due and keeps deep
compaction as the safety net.

Two invariants hold at all times:

 I1  Segments before `conv.frozen_upto` keep their bytes: the frozen prefix is what the
     provider has cached. Only a deep compaction (or compact_context) rebuilds it.

 I2  The assistant(tool_calls) -> tool(tool_call_id) pairing is never broken. Deep
     compaction removes whole tool groups (deep_plan.groups).
"""

from __future__ import annotations

import time

from ..segments import Conversation, Segment, K_SYSTEM, K_TOOL_CALL, K_TOOL_RESULT, K_SUMMARY
from .. import tokens
from . import summarize, deep_plan, freeze_review


class CompactionReport(dict):
    """Plain dict so it serialises straight to the UI event stream."""


def _pressure(total: int, effective_window: int) -> float:
    return total / max(1, effective_window)


class CompactionEngine:
    def __init__(self, session_id: str, cfg, complete_fn, summary_model_fn=None):
        self.session_id = session_id
        self.cfg = cfg
        self.complete_fn = complete_fn
        # Resolves "" (same as the chat model); the engine only sees CompactionConfig.
        self.summary_model_fn = summary_model_fn
        self.turns_since_deep = 10_000
        # Set by the session when the summariser is the chat model: a callable returning a
        # summarize.PrefixShare for the last chat request, or None.
        self.prefix_share = None

    def _share(self):
        if not getattr(self.cfg, 'summary_share_prefix', False) or not callable(self.prefix_share):
            return None
        try:
            return self.prefix_share()
        except Exception:  # noqa: BLE001 - a broken share only means the fresh prompt
            return None

    def summary_model(self) -> str:
        """The summariser model id for this engine's chat ("" only without a session)."""
        if self.summary_model_fn is not None:
            return self.summary_model_fn()
        return self.cfg.summary_model

    # ------------------------------------------------------------------
    async def run(
        self,
        conv: Conversation,
        effective_window: int,
        *,
        usable_window: int | None = None,
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
            return report

        if completed_batch:
            schedule['batches'] = schedule.get('batches', 0) + 1
        last_seq = schedule.get('seen_seq', -1)
        schedule['new_tokens'] = schedule.get('new_tokens', 0) + sum(seg.tokens() for seg in conv if seg.seq > last_seq)
        schedule['seen_seq'] = max((seg.seq for seg in conv), default=last_seq)
        pressure = _pressure(before_total, effective_window)
        report['pressure'] = round(pressure, 4)
        # The trigger is a share of the window; the session's usable window (output and
        # headroom reserved) caps it, so a high ratio never lets a prompt outgrow the window.
        trigger = cfg.deep_trigger_ratio
        if usable_window:
            trigger = min(trigger, usable_window / max(1, effective_window))
        want_deep = force_deep or (cfg.deep_enabled and pressure >= trigger
                                   and self.turns_since_deep >= cfg.deep_cooldown_turns)

        # The hint goes in before the freeze: the newest result may itself fall outside
        # the working tail, and its bytes must not change once frozen.
        if cfg.self_steered and not want_deep:
            self._hint(conv, report, pressure, completed_batch, clock)

        # Without review_before_freeze, everything older than the working tail is frozen
        # each cycle. With it, fresh and protected results stay out of the frozen prefix
        # (freeze_review.plan); self-steered, the prefix then advances only when the model
        # curates or compacts, and after a deep compaction.
        if not cfg.review_before_freeze:
            newly_frozen = conv.freeze_prefix(conv.compute_hot_start(cfg.hot_zone_tokens))
        elif cfg.self_steered:
            newly_frozen = 0
        else:
            newly_frozen = conv.freeze_prefix(freeze_review.plan(conv, cfg)['end'])
            if newly_frozen:
                schedule['reviewed_through_seq'] = max(s.seq for s in conv.segments[:conv.frozen_upto])
        report['newly_frozen'] = newly_frozen
        report['hot_start'] = report['frozen_segments'] = conv.frozen_upto
        report['frozen_tokens'] = sum(s.tokens() for s in conv.segments[:conv.frozen_upto])

        if want_deep:
            r = await self._deep(conv, tool_overhead, instructions, force_deep)
            if r:
                report["tiers"].append(r)
                report["deep"] = not bool(r.get("error"))
                report["freed"] += r.get("freed", 0)
        report["after"] = conv.total_tokens() + tool_overhead
        if report['deep']:
            schedule['batches'] = schedule['new_tokens'] = 0
            schedule.pop('review_due', None)
            schedule['last_reason'] = 'forced' if force_deep else 'pressure'
            report['hot_start'] = report['frozen_segments'] = conv.frozen_upto
            report['frozen_tokens'] = sum(s.tokens() for s in conv.segments[:conv.frozen_upto])
        report["duration_ms"] = int((time.time() - t0) * 1000)
        return report

    # ------------------------------------------------------------------
    def _hint(self, conv: Conversation, report: dict, pressure: float, completed_batch: bool, clock: int) -> None:
        """Self-steered: when a review is due, append a one-line hint to the newest tool
        result (not yet sent to the model, so appending is cache-safe)."""
        from . import curate
        cfg = self.cfg
        schedule = conv.compaction_state
        reason = ('pressure' if pressure >= cfg.rewrite_pressure_ratio else
                  'large new context' if schedule.get('new_tokens', 0) >= cfg.rewrite_max_new_tokens else
                  'batch interval' if schedule.get('batches', 0) >= cfg.rewrite_interval_batches else None)
        if reason is None:
            return
        schedule.setdefault('review_due', {'since_batch': clock, 'reason': reason})
        report['schedule'] = {'reason': 'agent review due (' + reason + ')',
                              'pending_batches': schedule.get('batches', 0),
                              'new_tokens': schedule.get('new_tokens', 0), 'interval': cfg.rewrite_interval_batches}
        last_hint = schedule.get('review_hint_batch')
        if not completed_batch or (last_hint is not None and clock - last_hint < cfg.rewrite_interval_batches):
            return
        segs = conv.segments
        last_call = max((i for i, s in enumerate(segs) if s.kind == K_TOOL_CALL), default=None)
        target = next((s for s in reversed(segs[last_call:] if last_call is not None else [])
                       if s.kind == K_TOOL_RESULT and not s.frozen), None)
        if target is None:
            return
        hint = curate.hint_text(conv, cfg, schedule.get('batches', 0), reason)
        target.body = (target.body or '') + '\n\n' + hint
        target.invalidate()
        schedule['review_hint_batch'] = clock
        report['review_hint'] = hint

    # ------------------------------------------------------------------
    async def _deep(self, conv: Conversation, tool_overhead: int, instructions: str, force: bool) -> dict | None:
        cfg = self.cfg
        conv.migrate_legacy_recall_pins()
        plan = deep_plan.plan(conv, cfg, tool_overhead)
        victims, retained = plan['victims'], plan['retained']
        freed_estimate = sum(s.tokens() for s in victims)
        if not victims or (not force and freed_estimate < cfg.deep_min_gain_tokens):
            return None
        prior = next((s.body for s in reversed(victims) if s.kind == K_SUMMARY), None)
        # The verbatim tail starts at the first retained segment after the last victim.
        pos = {id(s): i for i, s in enumerate(conv.segments)}
        kept = {id(s) for s in retained}
        last_victim = max(pos[id(s)] for s in victims)
        tail_start = next((s for s in conv.segments[last_victim + 1:] if id(s) in kept), None)
        share = self._share()
        try:
            text = await summarize.deep_summary(
                victims, prior, self.summary_model(), self.complete_fn,
                max_tokens=cfg.summary_max_tokens, extra_instructions=instructions,
                content_token_budget=plan['summary_budget'], share=share,
                keep_from_id=tail_start.id if tail_start is not None else None,
                kept_elsewhere=sum(1 for s in retained if s.kind != K_SYSTEM and pos[id(s)] < last_victim))
        except Exception as e:
            return {"tier": "3-deep", "error": str(e), "freed": 0}
        shared_messages = getattr(share, 'used', None) if share is not None else None
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
            "prefix_shared_messages": shared_messages,
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
        review = freeze_review.plan(conv, cfg) if cfg.review_before_freeze else None
        candidate = sum(s.tokens() for s in conv.segments[hot_start:review['end']]) if review else 0
        return {
            "total": total,
            "original": conv.original_tokens() + tool_overhead,
            "saved": max(0, conv.original_tokens() - conv.total_tokens()),
            "effective_window": effective_window,
            "pressure": round(_pressure(total, effective_window), 4),
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
            "deep_trigger_ratio": cfg.deep_trigger_ratio,
            "deep_target_ratio": cfg.deep_target_ratio,
            "turns_since_deep": self.turns_since_deep,
            "deep_ready": self.turns_since_deep >= cfg.deep_cooldown_turns,
        }
