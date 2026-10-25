"""Offline counterfactual replay. No LLM/API generation, no source-file rereads.

Reads current chat history through localhost, reconstructs recoverable observation
bodies, and replays fixed tool batches. This estimates input costs under ideal
prefix caching, NOT answer quality or provider eviction/tokenization. Results
contain only aggregate counts, no conversation content. Long-context tariff
transitions and model behavior changes are intentionally not predicted.
"""
import asyncio
import json
import os
from pathlib import Path
import tempfile

import httpx


async def replay(data):
    from ctxagent.segments import Conversation, Segment, K_TOOL_CALL, K_TOOL_RESULT, K_MEDIA
    from ctxagent.config import CompactionConfig
    from ctxagent.compaction import CompactionEngine
    from ctxagent import tokens
    from ctxagent.cache_policy import digest

    source = data['segments']
    # Tool call + results + associated media is the scheduling unit.
    batches, batch = [], []
    for raw in source:
        if raw['kind'] == K_TOOL_CALL and batch:
            batches.append(batch)
            batch = []
        batch.append(raw)
    if batch:
        batches.append(batch)
    # Keep a deterministic bounded workload; original effects/tools never rerun.
    batches = batches[-100:]
    observed_originals = sum(bool(s.get('original_body')) for b in batches for s in b)
    rows = []
    for hot in (10000, 50000):
        for mode in ('freeze_each', 'freeze_on_cycle'):
            for interval in (1, 2, 3, 5, 10, 20):
                cfg = CompactionConfig.from_dict(data['config']['compaction'])
                cfg.hot_zone_tokens = hot
                cfg.deep_enabled = cfg.significance_enabled = cfg.offload_enabled = False
                cfg.protected_recent_user_turns = 0
                # Replay controls cadence externally, including after implementation changes.
                if hasattr(cfg, 'rewrite_interval_batches'):
                    cfg.rewrite_interval_batches = 1
                engine = CompactionEngine('cadence-replay', cfg, None)
                conv = Conversation()
                previous_hashes = []
                total_prompt = total_read = mutations = rewrites = 0
                final_tokens = max_prompt = 0
                for idx, b in enumerate(batches, 1):
                    for raw in b:
                        clean = dict(raw)
                        clean.update(body=raw.get('original_body') if raw.get('original_body') is not None else raw['body'],
                            original_body=None, state='original', frozen=False, pinned=False, blob_id=None)
                        seg = Segment.from_dict(clean)
                        conv.add(seg)
                    before = {s.id: s.body for s in conv}
                    if idx % interval == 0:
                        report = await engine.run(conv, 10000000)
                        rewrites += bool(report['tiers'])
                    elif mode == 'freeze_each':
                        conv.freeze_prefix(conv.compute_hot_start(hot))
                    mutations += sum(before.get(s.id) != s.body for s in conv)
                    messages = conv.to_messages()
                    hashes = [digest(m) for m in messages]
                    sizes = [tokens.count_message(m) for m in messages]
                    common = 0
                    for a, z in zip(previous_hashes, hashes):
                        if a != z:
                            break
                        common += 1
                    prompt = sum(sizes)
                    total_prompt += prompt
                    total_read += sum(sizes[:common])
                    max_prompt = max(max_prompt, prompt)
                    final_tokens = prompt
                    previous_hashes = hashes
                rows.append({'hot_budget': hot, 'mode': mode, 'interval': interval,
                    'requests': len(batches), 'prompt_tokens_est': total_prompt,
                    'cache_read_tokens_est': total_read,
                    'ideal_hit_rate': total_read / total_prompt if total_prompt else 0,
                    'input_cost_units': (total_prompt-total_read)*1.25 + total_read*.1,
                    'rewrite_cycles': rewrites, 'rewritten_segments': mutations,
                    'final_context_tokens_est': final_tokens, 'peak_context_tokens_est': max_prompt})
    return {'assumptions': ['Fixed retained history, not an interactive A/B trial',
             'Ideal unlimited prefix cache; 0.1x reads and 1.25x misses/writes',
             'No output/reasoning cost change predicted; no quality evaluation',
             'Deep/semantic compaction disabled, only deterministic rewriting',
             'Bodies lost by earlier engine cannot be reconstructed; current bodies used if originals absent',
             'Last 100 recorded tool batches; 10k and 50k budgets compared'],
            'recoverable_original_bodies': observed_originals, 'rows': rows}


def main():
    root = Path(__file__).resolve().parent.parent
    with httpx.Client(timeout=30) as client:
        cid = client.get('http://127.0.0.1:8765/api/chats').json()[0]['id']
        data = client.get(f'http://127.0.0.1:8765/api/chats/{cid}').json()
    # Offload can retain only a stub in original_body. Recover immutable blob bytes
    # before redirecting the replay's writes to temporary storage.
    from ctxagent import config, blobs
    for s in data['segments']:
        if s.get('blob_id'):
            body = blobs.get_blob(s['blob_id'])
            if body is not None:
                s['original_body'] = body
    with tempfile.TemporaryDirectory(prefix='cadence-replay-') as temp:
        config.DATA_DIR = Path(temp)
        config.BLOB_DIR = blobs.BLOB_DIR = Path(temp) / 'blobs'
        result = asyncio.run(replay(data))
    (root / '.ctxagent/cadence-replay.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('hot   mode             N   cost-units    ideal-hit final-context rewrites')
    for r in result['rows']:
        print(f"{r['hot_budget']:5} {r['mode']:17} {r['interval']:2} {r['input_cost_units']:12.0f} "
              f"{r['ideal_hit_rate']:9.1%} {r['final_context_tokens_est']:13} {r['rewrite_cycles']:8}")
    print('Aggregate result: .ctxagent/cadence-replay.json')


if __name__ == '__main__':
    main()
