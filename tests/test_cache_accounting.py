"""Offline cache-policy and cost-accounting regressions; no billable calls."""
import copy
import json
import unittest
from unittest.mock import patch

import httpx

from ctxagent import accounting, cache_policy, llm
from ctxagent.segments import Conversation, K_TOOL_CALL, Segment
from tests.test_productivity import make_session


RATES = {'prompt': 0.000005, 'completion': 0.000025,
         'input_cache_read': 0.0000005, 'input_cache_write': 0.00000625}
RAW = {'prompt_tokens': 10000, 'completion_tokens': 1000,
       'prompt_tokens_details': {'cached_tokens': 7000, 'cache_write_tokens': 2000},
       'completion_tokens_details': {'reasoning_tokens': 600}, 'cost': 0.046}


class CacheTests(unittest.TestCase):
    def test_static_system_prefix_shared_across_chats(self):
        """Chats with different workdirs/user additions share one cached static block;
        the per-chat session block follows it unmarked (breakpoint budget unchanged)."""
        from ctxagent.agent import build_system_prompt
        from ctxagent.config import SessionConfig
        a = build_system_prompt(SessionConfig(workdirs=['C:/a']))
        b = build_system_prompt(SessionConfig(workdirs=['D:/b'], system_prompt='be brief'))
        self.assertIn('D:/b', b.split(cache_policy.SESSION_SEPARATOR, 1)[1])
        for model, key in (('anthropic/claude-opus-5.5', 'cache_control'),
                           ('openai/gpt-6.1-sol', 'prompt_cache_breakpoint')):
            sa, sb = (cache_policy.apply([{'role': 'system', 'content': s}, {'role': 'user', 'content': 'q'}],
                                         model, 1)[0][0]['content'] for s in (a, b))
            self.assertEqual(sa[0], sb[0])
            self.assertIn(key, sa[0])
            self.assertNotIn(key, sa[1])
            self.assertEqual(sa[0]['text'] + sa[1]['text'], a)
        # Routing key is shared across chats with the same prefix, distinct per model/tools.
        from ctxagent.agent import cache_routing_key
        ka = cache_routing_key(SessionConfig(model='m1', workdirs=['C:/a']), [{'name': 't'}])
        kb = cache_routing_key(SessionConfig(model='m1', workdirs=['D:/b'], system_prompt='x'), [{'name': 't'}])
        self.assertEqual(ka, kb)
        self.assertNotEqual(ka, cache_routing_key(SessionConfig(model='m2', workdirs=['C:/a']), [{'name': 't'}]))
        self.assertNotEqual(ka, cache_routing_key(SessionConfig(model='m1', workdirs=['C:/a']), [{'name': 'u'}]))
        # Without a session part the message is marked as before.
        plain = cache_policy.apply([{'role': 'system', 'content': 'static only'}], 'anthropic/claude-opus-5.5', 1)[0]
        self.assertEqual(len(plain[0]['content']), 1)
        self.assertIn('cache_control', plain[0]['content'][0])

    def test_system_prompt_without_compaction_engine(self):
        """Engine off -> no 'Context compaction' / 'Managing your own context' sections
        (nothing is shortened, the context tools are absent); engine on -> unchanged."""
        from ctxagent.agent import build_system_prompt, BASE_SYSTEM
        from ctxagent.config import SessionConfig, CompactionConfig
        on = build_system_prompt(SessionConfig(workdirs=['C:/a']))
        off = build_system_prompt(SessionConfig(workdirs=['C:/a'], compaction=CompactionConfig(enabled=False)))
        self.assertTrue(on.startswith(BASE_SYSTEM.rstrip('\n')))
        for phrase in ('Context compaction', 'Managing your own context', 'recall(', 'compact_context', 'curate_context'):
            self.assertIn(phrase, on)
            self.assertNotIn(phrase, off)
        self.assertIn('Guidelines', off)
        self.assertIn('Working directories for this chat', off)
        self.assertTrue(off.startswith(on.split('\nContext compaction\n', 1)[0].rstrip('\n')))

    def test_empty_tool_call_does_not_drop_prefix_breakpoint(self):
        messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'long prefix'},
                    {'role': 'assistant', 'content': None, 'tool_calls': []}]
        original = copy.deepcopy(messages)
        out, anchors = cache_policy.apply(messages, 'anthropic/test', 3)
        self.assertEqual(anchors, [0, 1])
        self.assertIn('cache_control', out[1]['content'][0])
        self.assertEqual(messages, original)
        self.assertIsNone(out[2]['content'])

    def test_old_anchor_retained_after_large_growth_and_ttl(self):
        messages = [{'role': 'user', 'content': str(i)} for i in range(80)]
        out, anchors = cache_policy.apply(messages, 'anthropic/test', 70, 20, '1h')
        self.assertEqual(anchors, [20, 69])
        self.assertEqual(out[20]['content'][0]['cache_control']['ttl'], '1h')
        self.assertEqual(out[70]['content'], '70')

    def test_tail_marker_is_extra_and_never_a_stable_anchor(self):
        messages = [{'role': 'system', 'content': 'system'}] + [{'role': 'user', 'content': str(i)} for i in range(1, 80)]
        messages.append({'role': 'tool', 'tool_call_id': 't', 'content': 'latest result'})
        out, anchors = cache_policy.apply(messages, 'anthropic/test', 70, 20, tail=True)
        self.assertEqual(anchors, [0, 20, 69])  # tail excluded: callers persist max(anchors)
        self.assertEqual(out[80]['content'][0]['cache_control'], {'type': 'ephemeral'})
        marked = [i for i, m in enumerate(out) if isinstance(m['content'], list)
                  and any('cache_control' in b for b in m['content'])]
        self.assertEqual(marked, [0, 20, 69, 80])  # within the 4-breakpoint limit
        # Fully frozen history: the tail coincides with the frozen anchor, no duplicate.
        out, _ = cache_policy.apply(messages, 'claude-opus-5-5', len(messages), tail=True)
        self.assertEqual(sum('cache_control' in b for m in out if isinstance(m['content'], list) for b in m['content']), 2)
        # Default (diagnostics, direct backend) stays without a tail marker.
        self.assertEqual(cache_policy.apply(messages, 'anthropic/test', 70, 20)[0][80]['content'], 'latest result')
        # OpenRouter requests use it; OpenAI-family hints are untouched by it.
        self.assertIn('cache_control', llm.apply_cache_breakpoints(messages, 'anthropic/test', 70)[80]['content'][0])
        out, _ = cache_policy.apply(messages, 'openai/gpt-6-astra', 70, tail=True)
        self.assertEqual(out[80]['content'], 'latest result')

    def test_openai_explicit_only_supported_family(self):
        messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'data'}]
        out, anchors = cache_policy.apply(messages, 'openai/gpt-6-astra', 2)
        self.assertIn('prompt_cache_breakpoint', out[1]['content'][0])
        self.assertNotIn('cache_control', out[1]['content'][0])
        self.assertEqual(cache_policy.apply(messages, 'openai/gpt-4.1', 2), (messages, []))
        self.assertEqual(cache_policy.apply(messages, 'other/model', 2), (messages, []))

    def test_prefix_diagnostics_shared_mutated_schema_model_and_gap(self):
        messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'prefix'}]
        first = cache_policy.diagnose(messages, [], 'test', 2, None, 'high', 1)
        same = cache_policy.diagnose(messages + [{'role': 'user', 'content': 'new'}], [], 'test', 2, first, 'high', 2)
        self.assertEqual(same['shared_messages'], 2)
        changed = cache_policy.diagnose([messages[0], {'role': 'user', 'content': 'changed'}], [], 'test', 2, first, 'high', 400)
        self.assertEqual(changed['shared_messages'], 1)
        self.assertTrue(any('frozen prefix changed' in r for r in changed['reasons']))
        self.assertTrue(any('5 minutes' in r for r in changed['reasons']))
        schema = cache_policy.diagnose(messages, [{'tool': 'changed'}], 'other', 2, first, 'low', 2)
        self.assertEqual(schema['shared_messages'], 0)
        self.assertIn('model changed', schema['reasons'])
        self.assertIn('tool schema changed', schema['reasons'])
        self.assertNotIn('message_hashes', cache_policy.public_diagnostic(first))

    def test_reasoning_and_diagnostics_persist_but_diagnostics_not_prompt(self):
        c = Conversation()
        c.cache_state = {'message_hashes': ['hash']}
        seg = c.new('assistant', K_TOOL_CALL, tool_calls=[], reasoning_details=[
            {'type': 'reasoning.encrypted', 'data': 'opaque', 'id': 'rs1', 'index': 0}])
        restored = Conversation.from_dict(c.to_dict())
        self.assertEqual(restored.cache_state, c.cache_state)
        self.assertEqual(restored.to_messages()[0]['reasoning_details'], seg.reasoning_details)
        self.assertNotIn('message_hashes', json.dumps(restored.to_messages()))


class AccountingTests(unittest.TestCase):
    def test_nonoverlapping_components_and_reasoning_not_double_billed(self):
        normalized = accounting.normalize(RAW)
        self.assertEqual([normalized[k] for k in ('input', 'cache_read', 'cache_write', 'output', 'reasoning')],
                         [1000, 7000, 2000, 400, 600])
        estimate = accounting.estimate(normalized, {'rates': RATES})
        self.assertTrue(estimate['complete'])
        self.assertAlmostEqual(estimate['total'], RAW['cost'])
        self.assertAlmostEqual(estimate['components']['reasoning'], 0.015)
        self.assertAlmostEqual(estimate['residual'], 0)

    def test_missing_reasoning_is_unknown_not_zero(self):
        raw = {**RAW, 'completion_tokens_details': {}}
        n = accounting.normalize(raw)
        self.assertIsNone(n['reasoning'])
        self.assertIsNone(n['output'])
        self.assertEqual(n['output_unsplit'], 1000)
        e = accounting.estimate(n, {'rates': RATES})
        self.assertTrue(e['complete'])
        self.assertAlmostEqual(e['total'], RAW['cost'])

    def test_zero_cost_and_zero_reasoning_valid(self):
        u = {}
        raw = {'prompt_tokens': 10, 'completion_tokens': 5, 'cost': 0,
               'completion_tokens_details': {'reasoning_tokens': 0}}
        accounting.record(u, raw, model='free', price={'rates': {k: 0 for k in RATES}})
        self.assertEqual(u['cost'], 0)
        self.assertEqual(u['reported_cost_requests'], 1)
        self.assertEqual(u['token_components']['output'], 5)
        self.assertEqual(u['component_unknown_requests']['reasoning'], 0)

    def test_legacy_migration_missing_prices_and_idempotent_request(self):
        u = {'cost': 50, 'requests': 10, 'prompt_tokens': 100000, 'cache_read': 0}
        entry = accounting.record(u, RAW, model='test', generation_id='g1')
        self.assertEqual(u['legacy_unallocated_cost'], 50)
        self.assertAlmostEqual(u['cost'], 50.046)
        self.assertEqual(u['cache_write'], 2000)
        self.assertEqual(u['component_unknown_requests']['input'], 1)
        self.assertEqual(u['unreconciled_reported_cost'], .046)
        self.assertIsNone(accounting.record(u, RAW, model='test', generation_id='g1'))
        self.assertEqual(u['requests'], 11)
        self.assertEqual(entry['raw_usage'], RAW)
        accounting.initialize(u)
        self.assertEqual(u['legacy_unallocated_cost'], 50)

    def test_last_recent_lifetime_separate_and_summary_in_total(self):
        u = {'cost': 1, 'prompt_tokens': 1000000, 'cache_read': 0}
        accounting.record(u, RAW, model='test', price={'rates': RATES})
        accounting.record(u, RAW, model='summary', source='summary', price={'rates': RATES})
        self.assertAlmostEqual(u['recent_cache_hit_rate'], .7)
        self.assertAlmostEqual(u['last_cache_hit_rate'], .7)
        self.assertLess(u['cache_hit_rate'], .02)
        self.assertAlmostEqual(u['by_source']['summary']['cost'], .046)
        self.assertAlmostEqual(u['cost'], 1.092)

    def test_ambiguous_provider_tariffs_not_guessed_and_long_context_override(self):
        eps = [{'provider_name': 'P', 'pricing': RATES}, {'provider_name': 'P', 'pricing': {**RATES, 'prompt': .1}}]
        resolved = accounting.resolve_price(eps, 'P', 10000)
        self.assertNotIn('prompt', resolved['rates'])
        self.assertIn('completion', resolved['rates'])
        self.assertEqual(accounting.resolve_price(eps, 'unknown', 1)['rates'], {})
        eps = [{'provider_name': 'P', 'pricing': {**RATES, 'overrides': [{'min_prompt_tokens': 20000, 'prompt': .2}]}}]
        self.assertEqual(accounting.resolve_price(eps, 'P', 20000)['rates']['prompt'], .2)
        self.assertEqual(accounting.resolve_price(eps, 'P', 19999)['rates']['prompt'], RATES['prompt'])

    def test_invalid_usage_does_not_invent_split(self):
        n = accounting.normalize({'prompt_tokens': 10, 'completion_tokens': 2,
            'prompt_tokens_details': {'cached_tokens': 100},
            'completion_tokens_details': {'reasoning_tokens': 50}, 'cost': 'NaN'})
        self.assertIsNone(n['input'])
        self.assertIsNone(n['reasoning'])
        self.assertIsNone(n['cost'])
        self.assertEqual(len(n['warnings']), 2)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_usage_snapshots_counted_once_metadata_and_reasoning_preserved(self):
        captured = []
        chunks = [
            {'id': 'gen-1', 'provider': 'P', 'model': 'openai/gpt-6-astra', 'usage': {'prompt_tokens': 100},
             'choices': [{'delta': {'reasoning_details': [{'index': 0, 'type': 'reasoning.encrypted', 'data': 'part1'}]}}]},
            {'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'cost': .001,
                       'prompt_tokens_details': {'cached_tokens': 80, 'cache_write_tokens': 10}},
             'choices': [{'delta': {'content': 'answer', 'reasoning_details': [{'index': 0, 'type': 'reasoning.encrypted', 'data': 'part2'}]}}]},
        ]
        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, text=''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n')
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        async def allowed(*args): return ['test-provider']
        with patch.object(llm.httpx, 'AsyncClient', return_value=client), patch.object(llm, '_headers', return_value={}), \
             patch.object(llm.provmod, 'allowed_slugs', allowed):
            events = [e async for e in llm.stream_chat(model='openai/gpt-6-astra', session_id='chat1',
                messages=[{'role': 'user', 'content': 'prefix'}], frozen_msg_count=1)]
        usage = [e for e in events if e['type'] == 'usage']
        self.assertEqual(len(usage), 1)
        self.assertEqual(usage[0]['usage']['prompt_tokens'], 100)
        self.assertEqual(usage[0]['provider'], 'P')
        self.assertEqual(usage[0]['generation_id'], 'gen-1')
        self.assertEqual(captured[0]['session_id'], 'chat1')
        self.assertIn('prompt_cache_breakpoint', captured[0]['messages'][0]['content'][0])
        blocks = next(e['reasoning_details'] for e in events if e['type'] == 'reasoning_details')
        self.assertEqual(blocks[0]['data'], 'part1part2')

    async def test_summary_usage_includes_empty_retry(self):
        calls = []
        async def post(*args):
            calls.append(args[0])
            return {'id': str(len(calls)), 'provider': 'P', 'usage': RAW,
                    'choices': [{'message': {'content': '' if len(calls) == 1 else 'summary'}}]}
        observed = []
        async def observe(raw, **meta): observed.append((raw, meta))
        async def allowed(*args): return ['test-provider']
        with patch.object(llm, '_post_with_backoff', post), patch.object(llm.provmod, 'allowed_slugs', allowed):
            result = await llm.complete(model='test', messages=[], usage_callback=observe, session_id='s:summary')
        self.assertEqual(result, 'summary')
        self.assertEqual(len(observed), 2)
        self.assertEqual(calls[0]['session_id'], 's:summary')

    async def test_tariff_inference_requires_reported_total_match(self):
        eps = [{'provider_name': 'P', 'tag': 'standard', 'pricing': RATES},
               {'provider_name': 'P', 'tag': 'flex', 'pricing': {k: v / 2 for k, v in RATES.items()}}]
        async def endpoints(*args): return eps
        with patch.object(accounting.providers, 'model_endpoints', endpoints):
            accounting._PRICE_CACHE.clear()
            price = await accounting.pricing('test', 'P', 10000, RAW)
            self.assertEqual(price['endpoint_tag'], 'standard')
            self.assertIn('inferred', price['basis'])
            unknown = await accounting.pricing('test', 'P', 10000, {**RAW, 'cost': 99})
            self.assertNotIn('prompt', unknown['rates'])

    async def test_agent_passes_reasoning_blocks_back_with_tool_results(self):
        from ctxagent.tools import ToolResult
        session = make_session()
        blocks = [{'type': 'reasoning.encrypted', 'data': 'opaque-provider-state', 'index': 0}]
        seen = []
        async def stream(**kw):
            seen.append(kw['messages'])
            if len(seen) == 1:
                yield {'type': 'reasoning_details', 'reasoning_details': blocks}
                yield {'type': 'tool_calls', 'tool_calls': [{'id': 'x', 'type': 'function',
                    'function': {'name': 'read_file', 'arguments': '{}'}}]}
            else:
                yield {'type': 'text', 'text': 'finished'}
            yield {'type': 'done', 'finish_reason': 'stop'}
        async def execute(*a): return ToolResult('tool output')
        session._exec_tool = execute
        with patch('ctxagent.agent.llm.stream_chat', stream):
            events = [e async for e in session.run_turn('test')]
        assistant = next(m for m in seen[1] if m.get('tool_calls'))
        self.assertEqual(assistant['reasoning_details'], blocks)
        self.assertEqual(seen[1][-1]['role'], 'tool')

    async def test_agent_summary_hook_records_cost(self):
        session = make_session()
        async def complete(**kw):
            await kw['usage_callback'](RAW, model='summary', provider='P', generation_id='summary-gen')
            return 'summary'
        async def price(*args, **kwargs): return {'rates': RATES}
        with patch('ctxagent.agent.llm.complete', complete), patch.object(accounting, 'pricing', price), \
             patch.object(accounting, 'append_ledger'):
            self.assertEqual(await session._complete(model='summary', messages=[]), 'summary')
        self.assertAlmostEqual(session.usage['cost'], .046)
        self.assertEqual(session.usage['by_source']['summary']['requests'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
