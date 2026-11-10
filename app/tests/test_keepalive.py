"""Prompt-cache keep-alive (2026-11-10): while a tool batch outlasts the 5-minute cache, the
last request is re-sent without output (direct Anthropic API: max_tokens 0; Claude on
OpenRouter: max_tokens 1), billed as a cache read.
In a long game-project session 68 requests after waits of ~10 minutes re-wrote their whole prompt ($25 of
$201); keep-alives would have cost about $3."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ctxagent import agent as agentmod, anthropic_api, config, llm, tools as toolmod
from ctxagent.agent import AgentSession
from ctxagent.config import CompactionConfig, SessionConfig
from ctxagent.segments import Conversation

DIRECT = 'claude-opus-5-5'
USAGE = {'input_tokens': 3, 'cache_read_input_tokens': 100_000, 'cache_creation_input_tokens': 0, 'output_tokens': 0}


def call(cid='toolu_wait'):
    return {'id': cid, 'type': 'function', 'function': {'name': 'run_command', 'arguments': json.dumps({'command': 'build'})}}


class KeepWarm(unittest.IsolatedAsyncioTestCase):
    async def test_resends_the_captured_request_with_max_tokens_zero(self):
        sent = []

        async def create(**kw):
            sent.append(kw)
            return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: dict(USAGE)))
        client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
        captured = {'kwargs': {'model': DIRECT, 'max_tokens': 128_000, 'messages': [{'role': 'user', 'content': 'x'}],
                               'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'xhigh'}}}
        with patch.object(anthropic_api, '_get_client', return_value=client):
            raw = await anthropic_api.keep_warm(captured)
        self.assertEqual(sent[0]['max_tokens'], 0)
        self.assertNotIn('stream', sent[0])
        self.assertEqual({k: v for k, v in sent[0].items() if k != 'max_tokens'},
                         {k: v for k, v in captured['kwargs'].items() if k != 'max_tokens'})
        self.assertEqual(captured['kwargs']['max_tokens'], 128_000)          # the capture is untouched
        self.assertAlmostEqual(raw['cost'], 100_000 * 0.2e-6 + 3 * 4e-6)       # a cache read, nothing else


class DuringABatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p1 = patch.object(config, 'DATA_DIR', Path(self.tmp.name))
        p2 = patch.object(agentmod, 'KEEPALIVE_S', 0.1)
        for p in (p1, p2):
            p.start()
            self.addCleanup(p.stop)
        llm.KEEPALIVE_REFUSED.clear()
        self.addCleanup(llm.KEEPALIVE_REFUSED.clear)

    def session(self, model=DIRECT, **comp):
        cfg = SessionConfig(model=model, compaction=CompactionConfig(**comp), workdirs=[self.tmp.name])
        s = AgentSession('keepalive', cfg, Conversation())
        s.persist = lambda: None
        return s

    def drive(self, s, batch_seconds=0.45, keep=None, turns=1):
        keep = keep or AsyncMock(return_value={'usage': anthropic_api.usage_payload(DIRECT, dict(USAGE)),
                                               'provider': 'Anthropic', 'generation_id': None})
        n = [0]

        async def stream(**kw):
            n[0] += 1
            if kw.get('capture') is not None:      # what llm.stream_chat leaves (direct API, Claude on OpenRouter)
                if kw['model'].startswith('claude-'):
                    kw['capture'].update({'route': 'anthropic', 'kwargs': {'model': kw['model']}, 'started': time.time()})
                else:
                    kw['capture'].update(llm._keepalive_capture(kw['model'], {'model': kw['model']}, time.time()) or {})
            if n[0] % 2 == 1:
                yield {'type': 'tool_calls', 'tool_calls': [call(f'toolu_{n[0]}')]}
            else:
                yield {'type': 'text', 'text': 'built'}
            yield {'type': 'done', 'finish_reason': 'stop'}

        async def slow_batch(session, tool_calls, finish_reason=None):
            tc = tool_calls[0]
            yield 'start', tc, 'run_command', {}, None, 0
            await asyncio.sleep(batch_seconds)
            yield 'result', tc, 'run_command', {}, toolmod.ToolResult(text='BUILD_OK'), int(batch_seconds * 1000)

        async def go():
            for _ in range(turns):
                [ev async for ev in s.run_turn('build it')]
        with patch('ctxagent.agent.llm.stream_chat', stream), patch('ctxagent.agent.execute_batch', slow_batch), \
                patch.object(llm, 'keep_warm', keep), \
                patch('ctxagent.agent.accounting.pricing', AsyncMock(return_value={'rates': {}, 'basis': 'test'})):
            asyncio.run(go())        # (no price lookup over the network for OpenRouter ids)
        return keep

    def test_keepalives_while_the_batch_runs_and_none_after(self):
        s = self.session()
        keep = self.drive(s)
        self.assertGreaterEqual(keep.await_count, 2)
        self.assertLessEqual(keep.await_count, 5)
        ka = s.usage['by_source']['keepalive']
        self.assertEqual(ka['requests'], keep.await_count)
        self.assertAlmostEqual(ka['cost'], keep.await_count * (100_000 * 0.2e-6 + 3 * 4e-6))
        calls = keep.await_count
        time.sleep(0.3)                                    # the turn is over: no more keep-alives
        self.assertEqual(keep.await_count, calls)

    def test_openrouter_for_claude_models_only(self):
        self.assertGreaterEqual(self.drive(self.session(model='anthropic/claude-opus-5.5')).await_count, 2)
        self.assertEqual(self.drive(self.session(model='openai/gpt-6.1-sol')).await_count, 0)
        self.assertEqual(self.drive(self.session(cache_keepalive=False)).await_count, 0)

    def test_at_most_half_as_many_as_one_rewrite_costs(self):
        s = self.session()
        s.usage = {'recent_requests': [{'source': 'agent', 'pricing': {'rates': {
            'prompt': 1.0, 'input_cache_read': 1.0, 'input_cache_write': 6.0, 'completion': 1.0}}}]}
        keep = self.drive(s, batch_seconds=0.9)
        self.assertEqual(keep.await_count, 3)              # write/read = 6: three, then the cache is let go

    def test_a_refused_keepalive_stops_them_and_does_not_break_the_batch(self):
        s = self.session(model='anthropic/claude-opus-5.5')
        keep = self.drive(s, keep=AsyncMock(side_effect=llm.KeepAliveRefused('HTTP 400: max_tokens')))
        self.assertEqual(keep.await_count, 1)
        self.assertIn('openrouter:anthropic/claude-opus-5.5', llm.KEEPALIVE_REFUSED)
        self.assertTrue(any(seg.body == 'BUILD_OK' for seg in s.conv.segments))   # the batch finished
        self.assertNotIn('keepalive', s.usage.get('by_source', {}))

    def test_a_keepalive_that_misses_the_cache_stops_and_two_switch_them_off(self):
        missed = {'usage': anthropic_api.usage_payload(DIRECT, {'input_tokens': 3, 'cache_read_input_tokens': 10_000,
                                                                'cache_creation_input_tokens': 90_000, 'output_tokens': 0}),
                  'provider': 'Anthropic', 'generation_id': None}
        s = self.session()
        keep = self.drive(s, keep=AsyncMock(return_value=missed), turns=3)
        self.assertEqual(keep.await_count, 2)              # one per batch, then off for the chat
        self.assertEqual(s._keepalive_misses, 2)

    def test_setting_round_trip(self):
        self.assertTrue(CompactionConfig().cache_keepalive)
        self.assertFalse(CompactionConfig.from_dict({'cache_keepalive': False}).cache_keepalive)
        with self.assertRaises(ValueError):
            CompactionConfig.from_dict({'cache_keepalive': 'yes'})


class OpenRouterKeepWarm(unittest.IsolatedAsyncioTestCase):
    async def test_same_payload_not_streamed_with_max_tokens_one(self):
        sent = []

        async def post(payload):
            sent.append(payload)
            return 200, {'id': 'gen-1', 'provider': 'Anthropic', 'usage': {'prompt_tokens': 100_000,
                         'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 99_990}, 'cost': 0.02}}
        routed = {'model': 'anthropic/claude-opus-5.5', 'messages': [{'role': 'user', 'content': 'x'}], 'stream': True,
                  'max_tokens': 128_000, 'session_id': 'ctxagent-abc', 'reasoning': {'effort': 'xhigh'},
                  'provider': {'order': ['anthropic']}}
        captured = llm._keepalive_capture('anthropic/claude-opus-5.5', routed, time.time())
        with patch.object(llm, '_post_json', post):
            out = await llm.keep_warm(captured)
        self.assertEqual((sent[0]['stream'], sent[0]['max_tokens']), (False, 1))
        self.assertEqual({k: v for k, v in sent[0].items() if k not in ('stream', 'max_tokens')},
                         {k: v for k, v in routed.items() if k not in ('stream', 'max_tokens')})
        self.assertEqual((out['provider'], out['generation_id']), ('Anthropic', 'gen-1'))
        self.assertTrue(routed['stream'])                  # the captured request is untouched

    async def test_refusal(self):
        async def post(payload):
            return 400, {'error': {'message': 'max_tokens too small'}}
        captured = llm._keepalive_capture('anthropic/claude-opus-5.5', {'model': 'x'}, time.time())
        with patch.object(llm, '_post_json', post), self.assertRaises(llm.KeepAliveRefused):
            await llm.keep_warm(captured)

    def test_no_capture_for_models_without_a_write_premium(self):
        self.assertIsNone(llm._keepalive_capture('openai/gpt-6.1-sol', {}, 0.0))
        self.assertIsNotNone(llm._keepalive_capture('anthropic/claude-sonnet-5.5', {}, 0.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
