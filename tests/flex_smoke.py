"""Opt-in BILLABLE Flex integration smoke. Two tiny synthetic requests, no user data.

Uses flex_only: never falls back to standard pricing. 256 output-token caps.
Not included in offline suites; run only with explicit user-authorized testing.
"""
import asyncio
import json
from pathlib import Path
import uuid

from ctxagent import config, flex, llm


async def main():
    config.load_env()
    session='flex-smoke-'+uuid.uuid4().hex
    report=[]
    messages=[{'role':'user','content':'Synthetic service-tier test. Reply only OK.'}]
    try:
        discovered=await flex.discover('openai/gpt-6-astra','eu_us',refresh=True)
        print('Astra eligible Flex endpoints:',[r['tag'] for r in discovered['candidates']],flush=True)
        events=[e async for e in llm.stream_chat(model='openai/gpt-6-astra',messages=messages,
                    pricing_mode='flex_only',reasoning_effort='low',max_tokens=256,session_id=session)]
        usage=[e for e in events if e['type']=='usage']
        row={'model':'openai/gpt-6-astra','streaming':True,'usage':usage,
             'errors':[e for e in events if e['type']=='error'],
             'received_text':any(e['type']=='text' for e in events)}
        report.append(row); print(json.dumps(row),flush=True)
        if row['errors'] or not usage:
            raise RuntimeError('Astra Flex smoke did not complete')
        observed=[]
        async def record(raw,**meta): observed.append({'usage':raw,**meta})
        text=await llm.complete(model='openai/gpt-5.6-luna',messages=messages,
            pricing_mode='flex_only',reasoning_effort='low',max_tokens=256,
            session_id=session+':summary',usage_callback=record)
        row={'model':'openai/gpt-5.6-luna','streaming':False,'usage':observed,'received_text':bool(text)}
        report.append(row); print(json.dumps(row),flush=True)
        if not all(r['usage'] and r['usage'][0].get('routing',{}).get('flex_confirmed') for r in report):
            raise RuntimeError('Provider did not confirm Flex for every request; inspect result')
    finally:
        (config.DATA_DIR/'flex-smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    asyncio.run(main())
