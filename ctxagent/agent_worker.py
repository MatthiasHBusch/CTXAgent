"""Detached child uses exactly the normal AgentSession + compaction/accounting loop.

The existing job supervisor owns timeout/cancellation, process identity and logs.
A crashed request is marked uncertain/failed, never automatically replayed.
"""
from __future__ import annotations
import asyncio
from contextlib import aclosing
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ctxagent import delegation, store
from ctxagent.config import load_env


async def run(identifier):
    from ctxagent.agent import get_session
    row = delegation._row(identifier)
    session = get_session(row['child'])
    if not session:
        raise RuntimeError('Child chat was deleted')
    session.conv.repair_tool_pairs()  # repair interrupted prior runs only in the owning worker
    result, error = '', ''
    async with aclosing(session.run_turn(row['request']['task'])) as stream:
        async for ev in stream:
            # Do not duplicate reasoning or content into supervisor logs.
            kind = ev.get('type')
            if kind == 'final':
                result = ev.get('text', '')
                if ev.get('finish_reason') == 'length':
                    error = 'Model output limit reached; handoff may be incomplete'
            elif kind == 'error':
                error = ev.get('error', 'Agent failed')
            elif kind in {'tool_start','tool_result','compaction','usage','cancelled'}:
                print(json.dumps({'type': kind, 'tool': ev.get('name'), 'ok': ev.get('ok')}, ensure_ascii=False), flush=True)
            if kind == 'cancelled':
                error = 'Agent cancelled'
    return result, error


def main(identifier):
    load_env()
    store.init()
    delegation.init()
    while True:
        delegation.reap()
        row = delegation._row(identifier)
        if row['state'] != 'queued':
            return 1
        if delegation.claim(identifier):
            break
        time.sleep(0.5)
    try:
        # Publish final only after asyncio.run drains thread executors.
        text, error = asyncio.run(run(identifier))
    except BaseException as exc:
        text, error = '', f'{type(exc).__name__}: {exc}'
    delegation.finish(identifier, text, error)
    return int(bool(error))


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
