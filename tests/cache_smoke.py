"""Opt-in billable smoke test; three small requests to the configured cheap model.

Run explicitly: python -B -m tests.cache_smoke. Never included in offline suites.
No user documents are sent. Stores only usage/model metadata and cache anchors.
"""
import asyncio
import json
from pathlib import Path
import uuid

from ctxagent import config, llm


async def main():
    config.load_env()
    session = 'cache-smoke-' + uuid.uuid4().hex
    messages = [{'role': 'system', 'content': 'Reply with OK. This is a synthetic cache transport test.'},
                {'role': 'user', 'content': '\n'.join(f'Reference entry {i}: square circle triangle.' for i in range(240))}]
    results = []
    for i in range(3):
        events = [e async for e in llm.stream_chat(model='openai/gpt-5.6-luna',
            messages=messages + [{'role': 'user', 'content': f'Test {i}; reply OK.'}],
            frozen_msg_count=2, session_id=session, max_tokens=256,
            reasoning_effort='off', provider_policy='eu_us')]
        row = {'request': i, 'usage': [e for e in events if e['type'] == 'usage'],
               'errors': [e for e in events if e['type'] == 'error'],
               'received_text': any(e['type'] == 'text' for e in events)}
        results.append(row)
        print(json.dumps(row), flush=True)
        if row['errors']:
            break
        await asyncio.sleep(1)
    dest = config.DATA_DIR / 'cache-smoke.json'
    dest.write_text(json.dumps(results, indent=2), encoding='utf-8')
    if any(r['errors'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
