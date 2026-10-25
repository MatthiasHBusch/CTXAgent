"""Bounded, content-free context/cost inspection. No network or chat mutations.

All token sizes are local estimates; cache hits and costs come from recorded API
usage. The CLI reads SQLite in read-only mode, not through a live AgentSession.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sqlite3

from . import tokens
from .segments import K_TOOL_RESULT


def _number(value):
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return value
    return None


def recent_usage(usage, recent=20):
    """Use agent-only requests, weighted hits, and matched cost cohorts."""
    rows = [r for r in usage.get('recent_requests', []) if r.get('source') == 'agent'][-recent:]
    prompt = sum(r.get('usage', {}).get('prompt_tokens', 0) or 0 for r in rows)
    read = sum(r.get('usage', {}).get('cache_read', 0) or 0 for r in rows)
    billed = [r for r in rows if _number(r.get('usage', {}).get('cost')) is not None]
    # Itemization may be absent even if the total is present. Never treat absence
    # as zero or divide partial itemization by a different cohort's total.
    itemized = [r for r in billed if _number(r.get('raw_usage', {}).get('cost_details', {})
                .get('upstream_inference_prompt_cost')) is not None]
    input_cost = sum(r['raw_usage']['cost_details']['upstream_inference_prompt_cost'] for r in itemized)
    matched_total = sum(r['usage']['cost'] for r in itemized)
    ratios = []
    for r in rows:
        local = r.get('cache_diagnostic', {}).get('prompt_tokens_estimate')
        actual = r.get('usage', {}).get('prompt_tokens')
        if local and actual:
            ratios.append(actual / local)
    return {
        'agent_requests': len(rows), 'prompt_tokens_reported': prompt,
        'cache_read_tokens_reported': read, 'weighted_cache_hit_rate': read / prompt if prompt else None,
        'reported_cost': sum(r['usage']['cost'] for r in billed) if billed else None,
        'missing_cost_requests': len(rows) - len(billed),
        'input_itemized_requests': len(itemized),
        'reported_input_cost': input_cost if itemized else None,
        'reported_input_fraction_matched': input_cost / matched_total if matched_total else None,
        'prefix_change_requests': sum(any('prefix changed' in x for x in
            r.get('cache_diagnostic', {}).get('reasons', [])) for r in rows),
        'provider_to_local_token_ratio_range': [min(ratios), max(ratios)] if ratios else None,
        'last_cache_reasons': rows[-1].get('cache_diagnostic', {}).get('reasons', []) if rows else [],
    }


def report(conv, cfg, usage=None, *, top=5, recent=20):
    from .tools import openai_schema

    top = max(1, min(10, int(top)))
    recent = max(1, min(100, int(recent)))
    schema = openai_schema(cfg.enabled_tools)
    overhead = tokens.count_tools(schema)
    by_kind, by_state = Counter(), Counter()
    outputs = defaultdict(list)
    largest = []
    frozen_tokens = 0
    frozen_originals = 0
    frozen_original_tokens = 0
    # Use the ACTUAL immutable boundary, not the next prospective freeze point.
    boundary = conv.frozen_upto
    for i, seg in enumerate(conv):
        size = seg.tokens()
        by_kind[seg.kind] += size
        by_state[seg.state] += size
        frozen = i < boundary or seg.frozen
        frozen_tokens += size if frozen else 0
        if seg.kind == K_TOOL_RESULT:
            outputs[seg.tool_name or '?'].append(size)
            if frozen and seg.state == 'original':
                frozen_originals += 1
                frozen_original_tokens += size
        largest.append({'segment_id': seg.id, 'kind': seg.kind, 'tool': seg.tool_name,
                        'tokens_estimate': size, 'state': seg.state, 'frozen': frozen,
                        'pinned': seg.pinned})
    sizes = []
    for name, values in outputs.items():
        ordered = sorted(values)
        sizes.append({'tool': name, 'results': len(values), 'tokens_estimate': sum(values),
                      'max_tokens_estimate': max(values),
                      'p95_tokens_estimate': ordered[max(0, math.ceil(len(values) * .95) - 1)]})
    document_live = conv.workspace.token_count()
    browser_live = conv.browser.token_count()
    total = conv.total_tokens() + overhead + document_live + browser_live
    cc = cfg.compaction
    warnings = []
    if not cc.enabled:
        warnings.append('Automatic compaction is disabled.')
    elif not cc.document_compaction_enabled:
        warnings.append('Incremental dedup/masking/condensation is disabled; cadence cannot reduce document tokens.')
    if (cc.enabled and cc.document_compaction_enabled and not cc.review_before_freeze
            and cc.hot_zone_tokens < cc.rewrite_max_new_tokens):
        warnings.append('Hot budget is smaller than the new-token flush threshold; observations can freeze before reduction.')
    if frozen_originals:
        warnings.append('Frozen original observations cannot be incrementally reduced; use focused reads or a deliberate deep pass.')
    if total >= cfg.effective_window():
        warnings.append('Estimated prompt exceeds usable window; reduce context before another request.')
    # These are legacy UI/config fields, not inputs to the current deep planner.
    warnings.append('tail_ratio_min/max, max_pressure and rehydrate_* are legacy settings; current deep planning uses deep_target_ratio.')
    return {
        'basis': 'Local token estimates, not exact provider context or billing. No message bodies, paths or arguments included.',
        'context': {'total_tokens_estimate': total, 'configured_window': cfg.context_window,
                    'effective_window': cfg.effective_window(), 'pressure_estimate': total / max(1, cfg.effective_window()),
                    'tool_schema_tokens_estimate': overhead, 'tool_schema_fraction': overhead / max(1, total),
                    'document_workspace_tokens_estimate': document_live,
                    'browser_workspace_tokens_estimate': browser_live,
                    'live_browser_tabs': len(conv.browser.views),
                    'frozen_tokens_estimate': frozen_tokens, 'frozen_segments': boundary,
                    'frozen_original_tool_results': frozen_originals,
                    'frozen_original_tool_tokens_estimate': frozen_original_tokens,
                    'by_kind': dict(by_kind), 'by_state': dict(by_state)},
        'policy': {'enabled': cc.enabled, 'document_compaction_enabled': cc.document_compaction_enabled,
                   'review_before_freeze': cc.review_before_freeze,
                   'initial_result_max_tokens': cc.initial_result_max_tokens,
                   'reviewed_through_seq': conv.compaction_state.get('reviewed_through_seq'),
                   'hot_zone_tokens': cc.hot_zone_tokens, 'rewrite_interval_batches': cc.rewrite_interval_batches,
                   'rewrite_max_new_tokens': cc.rewrite_max_new_tokens, 'deep_trigger_ratio': cc.deep_trigger_ratio,
                   'deep_target_ratio': cc.deep_target_ratio, 'summary_max_tokens': cc.summary_max_tokens,
                   'summary_min_tokens': cc.summary_min_tokens,
                   'pending_batches': conv.compaction_state.get('batches', 0)},
        'recent_usage': recent_usage(usage or {}, recent),
        'largest_segments': sorted(largest, key=lambda x: -x['tokens_estimate'])[:top],
        'largest_tool_outputs': sorted(sizes, key=lambda x: -x['tokens_estimate'])[:top],
        'largest_tool_schemas': sorted([
            {'tool': s['function']['name'], 'tokens_estimate': tokens.count_tools([s]) - 8}
            for s in schema], key=lambda x: -x['tokens_estimate'])[:top],
        'warnings': warnings,
    }


def main():
    from .config import DB_PATH, SessionConfig
    from .segments import Conversation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DB_PATH)
    parser.add_argument('--chat-id', help='Defaults to most recently updated saved chat.')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--recent', type=int, default=20)
    args = parser.parse_args()
    # No init(), WAL changes, AgentSession repair, or save_chat side effects.
    c = sqlite3.connect(args.db.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        if args.chat_id:
            row = c.execute('SELECT conversation,config,usage FROM chats WHERE id=?', (args.chat_id,)).fetchone()
        else:
            row = c.execute('SELECT conversation,config,usage FROM chats ORDER BY updated_at DESC LIMIT 1').fetchone()
    finally:
        c.close()
    if not row:
        parser.error('No saved chat found.')
    raw, cfg, usage = (json.loads(x) for x in row)
    print(json.dumps(report(Conversation.from_dict(raw), SessionConfig.from_dict(cfg), usage,
                            top=args.top, recent=args.recent), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
