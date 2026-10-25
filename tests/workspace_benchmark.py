"""Offline fixed-task replay: explicit workspace + grouped tool planning.

No model calls, no real billing. Latency is a synthetic blocking-read experiment;
request counts come from a scripted fake model, NOT proof of autonomous planning.
"""
import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from ctxagent.agent import AgentSession
from ctxagent.config import SessionConfig, CompactionConfig
from ctxagent import tools, tokens
from ctxagent.segments import Conversation
from ctxagent.tool_batch import execute_batch

ROOT = Path(__file__).resolve().parent.parent


def read_call(i):
    return {'id': f'read-{i}', 'type': 'function', 'function': {
        'name': 'read_file', 'arguments': json.dumps({'path': f'{i}.txt'})}}


async def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for i in range(8):
            (root / f'{i}.txt').write_text(f'Independent file {i}: result {i*i}', encoding='utf-8')
        cfg = SessionConfig(workdirs=[td], compaction=CompactionConfig(enabled=False))
        async def workflow(grouped):
            session = AgentSession('benchmark', cfg, Conversation())
            session.persist = lambda: None
            requests = 0
            prompts = []
            received = []
            async def fake_model(**kw):
                nonlocal requests
                requests += 1
                prompts.append(tokens.count_messages(kw['messages']))
                if requests <= (1 if grouped else 8):
                    calls = [read_call(i) for i in range(8)] if grouped else [read_call(requests-1)]
                    yield {'type': 'tool_calls', 'tool_calls': calls}
                else:
                    yield {'type': 'text', 'text': 'finished fixed eight-file task'}
                yield {'type': 'done', 'finish_reason': 'stop'}
            with patch('ctxagent.agent.llm.stream_chat', fake_model):
                events = [e async for e in session.run_turn('Read the eight independent result files.')]
            for e in events:
                if e['type'] == 'error':
                    raise AssertionError(e)
                if e['type'] == 'tool_result':
                    assert e['ok'], e
                    received.append(e['segment']['body'])
            assert len(received) == 8
            return {'model_requests': requests, 'prompt_tokens_estimate_sum': sum(prompts),
                    'read_results': received}
        one = await workflow(False)
        batched = await workflow(True)
        assert one.pop('read_results') == batched.pop('read_results')

        spec = tools.REGISTRY['read_file']
        original = spec.fn
        async def delayed(*args, **kwargs):
            time.sleep(.03)  # Synthetic blocking I/O, not real disk-performance evidence.
            return await original(*args, **kwargs)
        async def time_batch(calls):
            session = AgentSession('latency', cfg, Conversation())
            session.persist = lambda: None
            start = time.perf_counter()
            results = [e async for e in execute_batch(session, calls) if e[0] == 'result']
            assert len(results) == len(calls)
            return time.perf_counter() - start
        with patch.object(spec, 'fn', delayed):
            grouped_ms = 1000 * await time_batch([read_call(i) for i in range(8)])
            start = time.perf_counter()
            for i in range(8):
                await time_batch([read_call(i)])
            serial_ms = 1000 * (time.perf_counter() - start)

        session = AgentSession('workspace', cfg, Conversation())
        before = session.conv.to_messages()
        receipt = await session._exec_tool('document_open', {'path': '0.txt'})
        assert not receipt.error, receipt.text
        assert 'Independent file 0' not in receipt.text
        view = session.conv.workspace.views[0]
        original_version = view['version']
        (root / '0.txt').write_text('Externally updated working file', encoding='utf-8')
        session.conv.workspace.refresh(session.sandbox())
        assert 'Externally updated' in session.conv.workspace.messages()[0]['content']
        assert session.conv.workspace.views[0]['version'] != original_version
        await session._exec_tool('document_close', {'document_id': view['id']})
        assert session.conv.workspace.messages() == []
        assert session.conv.to_messages() == before

        report = {'kind': 'offline scripted replay; not billing/task-quality evidence',
                  'fixed_eight_file_workflow': {'one_at_a_time': one, 'grouped': batched,
                    'same_read_results': True},
                  'synthetic_blocking_read_latency': {'serial_ms': round(serial_ms, 1),
                    'max_four_concurrent_ms': round(grouped_ms, 1)},
                  'workspace_lifecycle': 'open, external refresh, close; no duplicate receipt payload, no history changes'}
        out = ROOT / '.ctxagent' / 'workspace-benchmark.json'
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
