"""The compaction orchestrator.

Runs after every tool batch and every finished answer. It compacts in one way:

    deep compaction   one summary call; the older history becomes a summary, the newest
                      tool group, recent user turns and pins stay verbatim, the originals
                      go to the archive (recall restores them)

It runs when the context reaches `deep_trigger_ratio` of the window (or when forced:
the Compact button, compact_now). Self-steered chats (rewrite_policy='agent') manage their
context themselves with compact_context and curate_context; the engine appends a review
hint to the newest tool result when a review is due and keeps deep compaction as the
safety net. Which cache points the next request sends is cache_policy.plan_rungs' job.

The assistant(tool_calls) -> tool(tool_call_id) pairing is never broken: deep compaction
removes whole tool groups (deep_plan.groups).
"""

from __future__ import annotations

import time

from ..segments import Conversation, Segment, K_SYSTEM, K_TOOL_CALL, K_TOOL_RESULT, K_SUMMARY
from .. import tokens
from . import summarize, deep_plan


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
        pressure = _pressure(before_total, effective_window)
        report['pressure'] = round(pressure, 4)
        # The trigger is a share of the window; the session's usable window (output and
        # headroom reserved) caps it, so a high ratio never lets a prompt outgrow the window.
        trigger = cfg.deep_trigger_ratio
        if usable_window:
            trigger = min(trigger, usable_window / max(1, effective_window))
        want_deep = force_deep or (cfg.deep_enabled and pressure >= trigger
                                   and self.turns_since_deep >= cfg.deep_cooldown_turns)

        if cfg.self_steered and not want_deep:
            self._hint(conv, report, pressure, completed_batch, clock)

        if want_deep:
            r = await self._deep(conv, tool_overhead, instructions, force_deep)
            if r:
                report["tiers"].append(r)
                report["deep"] = not bool(r.get("error"))
                report["freed"] += r.get("freed", 0)
        report["after"] = conv.total_tokens() + tool_overhead
        if report['deep']:
            schedule['batches'] = 0
            schedule.pop('review_due', None)
            schedule['last_reason'] = 'forced' if force_deep else 'pressure'
        report["duration_ms"] = int((time.time() - t0) * 1000)
        return report

    # ------------------------------------------------------------------
    def _hint(self, conv: Conversation, report: dict, pressure: float, completed_batch: bool, clock: int) -> None:
        """Self-steered: when a review is due, append the hint to the newest tool result (not
        yet sent to the model, so appending is cache-safe). The hint names the cache
        breakpoints in the history (curate.hint); what a review is, and how curate_context
        works, the system prompt says."""
        from . import curate
        cfg = self.cfg
        schedule = conv.compaction_state
        reason = ('pressure' if pressure >= cfg.rewrite_pressure_ratio else
                  'batch interval' if schedule.get('batches', 0) >= cfg.rewrite_interval_batches else None)
        if reason is None:
            return
        schedule.setdefault('review_due', {'since_batch': clock, 'reason': reason})
        report['schedule'] = {'reason': 'agent review due (' + reason + ')',
                              'pending_batches': schedule.get('batches', 0), 'interval': cfg.rewrite_interval_batches}
        last_hint = schedule.get('review_hint_batch')
        if not completed_batch or (last_hint is not None and clock - last_hint < cfg.rewrite_interval_batches):
            return
        segs = conv.segments
        last_call = max((i for i, s in enumerate(segs) if s.kind == K_TOOL_CALL), default=None)
        target = next((s for s in reversed(segs[last_call:] if last_call is not None else [])
                       if s.kind == K_TOOL_RESULT), None)
        if target is None:
            return
        hint = curate.hint(conv)
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
        total = conv.total_tokens() + tool_overhead
        by_state: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        for s in conv.segments:
            by_state[s.state] = by_state.get(s.state, 0) + s.tokens()
            by_kind[s.kind] = by_kind.get(s.kind, 0) + s.tokens()
        return {
            "total": total,
            "original": conv.original_tokens() + tool_overhead,
            "saved": max(0, conv.original_tokens() - conv.total_tokens()),
            "effective_window": effective_window,
            "pressure": round(_pressure(total, effective_window), 4),
            "initial_result_max_tokens": cfg.initial_result_max_tokens,
            "rewrite_interval_batches": cfg.rewrite_interval_batches,
            "rewrite_pending_batches": conv.compaction_state.get('batches', 0),
            "rewrite_last_reason": conv.compaction_state.get('last_reason'),
            "tool_overhead": tool_overhead,
            "segments": len(conv.segments),
            "by_state": by_state,
            "by_kind": by_kind,
            "deep_trigger_ratio": cfg.deep_trigger_ratio,
            "deep_target_ratio": cfg.deep_target_ratio,
            "turns_since_deep": self.turns_since_deep,
            "deep_ready": self.turns_since_deep >= cfg.deep_cooldown_turns,
        }
