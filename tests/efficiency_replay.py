"""Offline deterministic policy sweep of a READ-ONLY saved trajectory.

No model calls or tool execution. Quality, eviction, provider block granularity,
long-context tariffs, and changed model behavior are NOT simulated. Not a bill
prediction. Current retained history is a partial trajectory after deep passes.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import tempfile

from ctxagent import tokens, tools, blobs, config
from ctxagent.cache_policy import digest
from ctxagent.compaction import CompactionEngine
from ctxagent.config import SessionConfig, CompactionConfig, DB_PATH
from ctxagent.segments import Conversation, Segment, K_TOOL_CALL


def batches_for(source, limit):
    # Each complete call/result group is replayed before the next model request.
    # Keep initial instructions with the first call, not as a fictitious batch.
    batches, batch, has_call = [], [], False
    for s in source:
        if s['kind'] == K_TOOL_CALL:
            if has_call:
                batches.append(batch)
                batch = []
            has_call = True
        batch.append(s)
    if has_call:
        batches.append(batch)
    if batches:
        # A live read-only snapshot can contain intentions whose tools are still
        # running. They are not a completed batch or a legitimate next request.
        calls = next(s for s in batches[-1] if s['kind'] == K_TOOL_CALL)
        expected = {t.get('id') for t in calls.get('tool_calls') or []}
        actual = {s.get('tool_call_id') for s in batches[-1] if s['kind'] == 'tool_result'}
        if expected - actual:
            batches.pop()
    batches = batches[-limit:]
    # Initial system instructions always survive even when the trajectory is sliced.
    systems = [s for s in source if s['kind'] == 'system']
    if batches:
        existing = {s['id'] for s in batches[0]}
        batches[0] = [s for s in systems if s['id'] not in existing] + batches[0]
    return batches


async def sweep(source, cfg, *, limit=100, linked=False):
    from ctxagent.result_delivery import prepare
    batches = batches_for(source, limit)
    overhead = tokens.count_tools(tools.openai_schema(cfg.enabled_tools))
    policies = [(False, cfg.compaction.hot_zone_tokens, cfg.compaction.rewrite_interval_batches, False, 0)]
    if linked:
        policies += [(True, cfg.compaction.hot_zone_tokens, cfg.compaction.rewrite_interval_batches, False, 0)]
        policies += [(True, hot, interval, True, cap)
                     for hot in (10000, 50000) for interval in (5, 10, 20)
                     for cap in (0, 4000, 8000, 12000)]
    else:
        policies += [(True, hot, interval, False, 0) for hot in (10000, 50000, 80000) for interval in (1, 5, 10)]
    rows = []
    for enabled, hot, interval, review, cap in policies:
        cc = CompactionConfig.from_dict(cfg.compaction.to_dict())
        cc.enabled = True
        cc.document_compaction_enabled = enabled
        cc.hot_zone_tokens, cc.rewrite_interval_batches = hot, interval
        cc.review_before_freeze, cc.initial_result_max_tokens = review, cap
        cc.deep_enabled = cc.significance_enabled = False
        engine = CompactionEngine('offline-sweep', cc, None)
        conv = Conversation()
        previous, prompt_sum, shared_sum, peak, rewrites = [], 0, 0, 0, 0
        reasons = {}
        for batch in batches:
            for raw in batch:
                clean = dict(raw)
                clean.update(body=raw.get('original_body') if raw.get('original_body') is not None else raw['body'],
                             original_body=None, state='original', frozen=False, blob_id=None)
                # Preserve explicit user pins; recall pins are no longer active
                # in this reconstructed run (recall tool effects aren't replayed).
                if clean.get('pin_scope') == 'recall':
                    clean['pinned'] = False
                seg = Segment.from_dict(clean)
                prepare(seg, cc, 'offline-sweep', error=seg.body.startswith(('Error:', '[cancelled]')))
                conv.add(seg)
            r = await engine.run(conv, cfg.effective_window(), tool_overhead=overhead)
            reason = r.get('schedule', {}).get('reason', 'disabled')
            reasons[reason] = reasons.get(reason, 0) + 1
            rewrites += bool(r['tiers'])
            msgs = conv.to_messages()
            hashes = [digest(m) for m in msgs]
            sizes = [tokens.count_message(m) for m in msgs]
            common = 0
            for a, b in zip(previous, hashes):
                if a != b:
                    break
                common += 1
            prompt = sum(sizes) + overhead + 3
            shared = sum(sizes[:common]) + (overhead if previous else 0)
            prompt_sum += prompt
            shared_sum += shared
            peak = max(peak, prompt)
            previous = hashes
        rows.append({'incremental': enabled, 'review_before_freeze': review,
                     'initial_result_max_tokens': cap, 'hot_tokens': hot, 'interval': interval,
                     'requests': len(batches), 'prompt_tokens_estimate': prompt_sum,
                     'cache_read_tokens_ideal_estimate': shared_sum,
                     'ideal_hit_rate': shared_sum / prompt_sum if prompt_sum else None,
                     # Both normalized tariffs cover measured read discounts here.
                     'cost_units_read_010_write_125': shared_sum * .10 + (prompt_sum - shared_sum) * 1.25,
                     'cost_units_read_005_write_125': shared_sum * .05 + (prompt_sum - shared_sum) * 1.25,
                     'final_context_estimate': conv.total_tokens() + overhead,
                     'peak_context_estimate': peak, 'fits_window_without_deep': peak <= cfg.effective_window(),
                     'rewriting_batches': rewrites, 'schedule_reasons': reasons})
    for key in ('cost_units_read_010_write_125', 'cost_units_read_005_write_125'):
        baseline = rows[0][key]
        for row in rows:
            row[key + '_vs_disabled'] = row[key] / baseline if baseline else None
    return {'assumptions': [
        'Fixed current retained trajectory; source files/tools/model never rerun',
        'Recovered originals/blobs when available; lost content cannot be reconstructed',
        'Ideal unlimited shared-prefix caching, local tokenizer, static schema',
        'Normalized input write 1.25x, read .10x and .05x; not actual billing',
        'Actual scheduler with pressure/new-token overrides; deep and semantic passes disabled',
        'Explicit pins preserved; old recall protection cleared in reconstructed run',
        'No prediction of quality, output/reasoning, provider eviction, or long-context tariffs'],
        'source_snapshot_sha256': digest(source), 'config_sha256': digest(cfg.to_dict()),
        'source_segments': len(source), 'recoverable_originals': sum(s.get('original_body') is not None for s in source),
        'configured_window': cfg.context_window, 'effective_window': cfg.effective_window(), 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DB_PATH)
    parser.add_argument('--chat-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=100)
    parser.add_argument('--linked', action='store_true', help='Compare linked review/freeze and initial-delivery caps.')
    parser.add_argument('--include-archive', action='store_true', help='Recover archived segments read-only; omit synthetic deep summaries.')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new path.')
    c = sqlite3.connect(args.db.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        row = c.execute('SELECT conversation,config FROM chats WHERE id=?', (args.chat_id,)).fetchone()
    finally:
        c.close()
    if not row:
        parser.error('Chat not found.')
    saved = json.loads(row[0])
    source = saved['segments']
    recovered_archive = 0
    if args.include_archive:
        recovered = []
        for handle in saved.get('archive', {}).values():
            body = blobs.get_blob(handle)
            if body is not None:
                recovered.append(json.loads(body))
        recovered_archive = len(recovered)
        # Prefer active versions of matching IDs, never duplicate a summary AND
        # the original history it stands for. All source access remains read-only.
        by_id = {s['id']: s for s in recovered + source if s['kind'] != 'summary'}
        source = sorted(by_id.values(), key=lambda s: s['seq'])
    cfg = SessionConfig.from_dict(json.loads(row[1]))
    for s in source:
        if s.get('blob_id'):
            body = blobs.get_blob(s['blob_id'])
            if body is not None:
                s['original_body'] = body
    # Engine offload writes ONLY to temporary storage, restored even on failure.
    old_dirs = (config.DATA_DIR, config.BLOB_DIR, blobs.BLOB_DIR)
    with tempfile.TemporaryDirectory(prefix='ctxagent-policy-sweep-') as temp:
        try:
            config.DATA_DIR = Path(temp)
            config.BLOB_DIR = blobs.BLOB_DIR = Path(temp) / 'blobs'
            result = asyncio.run(sweep(source, cfg, limit=max(1, min(100, args.batches)), linked=args.linked))
        finally:
            config.DATA_DIR, config.BLOB_DIR, blobs.BLOB_DIR = old_dirs
    result['archive_segments_recovered'] = recovered_archive
    result['synthetic_deep_summaries_omitted'] = args.include_archive
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('incremental linked cap   hot  interval  input-cost-vs-disabled(.10/.05) hit    final   rewrites')
    for r in result['rows']:
        print(f"{str(r['incremental']):11} {str(r['review_before_freeze']):6} {r['initial_result_max_tokens']:5} {r['hot_tokens']:5} {r['interval']:3} "
              f"{r['cost_units_read_010_write_125_vs_disabled']:7.3f}/{r['cost_units_read_005_write_125_vs_disabled']:.3f} "
              f"{r['ideal_hit_rate'] or 0:6.1%} {r['final_context_estimate']:7} {r['rewriting_batches']:4}")
    print('Aggregate results:', args.output)


if __name__ == '__main__':
    main()
