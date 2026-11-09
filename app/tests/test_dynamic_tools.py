"""Native dynamic tools on the direct Anthropic API: late-bound operations declared with
defer_loading, surfaced and removed by tool-change segments, placed where the API
accepts system messages, re-added after compaction, catalog fallback when refused.
Offline: requests are built with anthropic_api.build_request or captured from a scripted
stream; nothing is sent."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxagent import anthropic_api, config, document_catalog, dynamic_tools, llm
from ctxagent.canvas import OPERATIONS
from ctxagent.segments import K_TOOL_CHANGE, Conversation
from tests.test_productivity import make_session

DIRECT = 'claude-opus-5-5'


def request(messages, tools, **kw):
    with patch('ctxagent.anthropic_api.expand_messages', side_effect=lambda m: m):
        return anthropic_api.build_request(model=DIRECT, messages=messages, tools=tools, max_tokens=100,
                                           reasoning_effort='off', frozen_msg_count=0,
                                           previous_cache_anchor=None, cache_ttl='5m', **kw)


def change(add=(), remove=()):
    return {'role': 'system', 'content':
            [{'type': 'tool_addition', 'tool': {'type': 'tool_reference', 'name': n}} for n in add]
            + [{'type': 'tool_removal', 'tool': {'type': 'tool_reference', 'name': n}} for n in remove]}


TOOLS = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'r', 'parameters': {'type': 'object'}}},
         {'type': 'function', 'defer_loading': True,
          'function': {'name': 'canvas_draw', 'description': 'd', 'parameters': {'type': 'object'}}},
         {'type': 'function', 'defer_loading': True,
          'function': {'name': 'canvas_undo', 'description': 'u', 'parameters': {'type': 'object'}}}]


def call(cid, name):
    return {'role': 'assistant', 'content': None,
            'tool_calls': [{'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}]}


class Reset(unittest.TestCase):
    def setUp(self):
        saved = set(anthropic_api._REJECTED)
        anthropic_api._REJECTED.clear()
        self.addCleanup(lambda: (anthropic_api._REJECTED.clear(), anthropic_api._REJECTED.update(saved)))


class RequestTests(Reset):
    def test_deferred_tools_and_the_beta(self):
        kw = request([{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'hi'}], TOOLS)
        self.assertEqual([t.get('defer_loading', False) for t in kw['tools']], [False, True, True])
        self.assertIn(dynamic_tools.BETA, kw['betas'])
        self.assertEqual(anthropic_api.TOOL_CHANGES_BETA, dynamic_tools.BETA)
        self.assertNotIn('cache_control', kw)   # explicit tail mark instead
        self.assertIn('cache_control', kw['messages'][-1]['content'][-1])

    def test_changes_go_after_user_turns_and_before_the_assistant(self):
        messages = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'draw'},
                    change(add=['canvas_draw']), call('c1', 'canvas_draw'),
                    {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
                    change(remove=['canvas_draw']),
                    {'role': 'user', 'content': 'LIVE CANVAS ...'}]   # the trailer behind the change
        kw = request(messages, TOOLS, uncached_tail=1)
        roles = [m['role'] for m in kw['messages']]
        self.assertEqual(roles, ['user', 'system', 'assistant', 'user', 'system'])
        self.assertEqual(kw['messages'][1]['content'][0]['type'], 'tool_addition')
        self.assertEqual(kw['messages'][-1]['content'],
                         [{'type': 'tool_removal', 'tool': {'type': 'tool_reference', 'name': 'canvas_draw'}}])
        tail = kw['messages'][3]['content']
        self.assertEqual(tail[0]['type'], 'tool_result')
        self.assertIn('cache_control', tail[0])      # before the trailer, not on it
        self.assertNotIn('cache_control', tail[-1])
        # The next request: the same bytes up to the cached tool result.
        later = messages[:-1] + [{'role': 'assistant', 'content': 'done'}, {'role': 'user', 'content': 'thanks'}]
        kw2 = request(later, TOOLS)
        self.assertEqual([m['role'] for m in kw2['messages']], ['user', 'system', 'assistant', 'user', 'system',
                                                                 'assistant', 'user'])
        def upto_mark(ms):  # everything through the tail-marked tool result
            head = ms[:3] + [{**ms[3], 'content': ms[3]['content'][:1]}]
            return json.dumps([{**m, 'content': [{k: v for k, v in b.items() if k != 'cache_control'}
                                                 for b in m['content']]} for m in head])
        self.assertEqual(upto_mark(kw['messages']), upto_mark(kw2['messages']))

    def test_stray_and_undeclared_changes_are_dropped(self):
        messages = [{'role': 'user', 'content': 'x'},
                    change(add=['canvas_draw', 'unknown_tool'], remove=['canvas_undo']),
                    {'role': 'assistant', 'content': 'y'}, {'role': 'user', 'content': 'z'},
                    change(add=['canvas_draw'])]
        kw = request(messages, TOOLS)
        self.assertEqual(kw['messages'][1]['content'],
                         [{'type': 'tool_addition', 'tool': {'type': 'tool_reference', 'name': 'canvas_draw'}}])
        self.assertEqual([m['role'] for m in kw['messages']], ['user', 'system', 'assistant', 'user'])

    def test_refused_beta_falls_back_to_plain_tools(self):
        exc = Exception("tools.1: Extra inputs are not permitted: 'defer_loading'")
        self.assertEqual(anthropic_api._rejected_feature(exc), 'tool_changes')
        anthropic_api._REJECTED.add('tool_changes')
        messages = [{'role': 'user', 'content': 'x'}, change(add=['canvas_draw'])]
        kw = request(messages, TOOLS)
        self.assertEqual([t['name'] for t in kw['tools']], ['read_file', 'canvas_draw'])
        self.assertFalse(any('defer_loading' in t for t in kw['tools']))
        self.assertNotIn(dynamic_tools.BETA, kw.get('betas', []))
        self.assertEqual([m['role'] for m in kw['messages']], ['user'])
        self.assertFalse(dynamic_tools.supported(DIRECT))

    def test_without_deferred_tools_changes_vanish(self):
        messages = [{'role': 'user', 'content': 'x'}, change(add=['canvas_draw'])]
        self.assertEqual([m['role'] for m in request(messages, TOOLS[:1])['messages']], ['user'])
        self.assertEqual(anthropic_api.strip_tool_changes(messages), messages[:1])

    def test_openrouter_payload_has_no_tool_changes(self):
        import asyncio
        from unittest.mock import AsyncMock
        sent = []

        def route(payload, plan):
            sent.append(payload)
            raise ConnectionAbortedError('captured')

        async def run():
            try:
                async for _ in llm.stream_chat(model='anthropic/claude-opus-5.5', messages=[
                        {'role': 'user', 'content': 'x'}, change(add=['canvas_draw'])]):
                    pass
            except ConnectionAbortedError:
                pass
        with patch('ctxagent.llm._apply_route', route), patch('ctxagent.llm._headers', return_value={}), \
                patch('ctxagent.llm.flex.plans', AsyncMock(return_value=[{'tier': 'standard'}])):
            asyncio.run(run())
        self.assertTrue(sent)
        self.assertEqual(len(sent[0]['messages']), 1)
        self.assertFalse(any(anthropic_api.is_tool_change(m) for m in sent[0]['messages']))


class AgentTests(Reset):
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = patch.object(config, 'DATA_DIR', Path(self.temp.name) / 'data')
        p.start()
        self.addCleanup(p.stop)
        (Path(self.temp.name) / 'work').mkdir()

    def session(self, model=DIRECT):
        s = make_session(root=Path(self.temp.name) / 'work')
        s.cfg.model = model
        return s

    def test_which_models(self):
        self.assertTrue(dynamic_tools.supported('claude-opus-5-5'))
        self.assertTrue(dynamic_tools.supported('claude-sonnet-5-5-20260901'))
        self.assertFalse(dynamic_tools.supported('claude-sonnet-5'))
        self.assertFalse(dynamic_tools.supported('anthropic/claude-opus-5.5'))
        self.assertFalse(dynamic_tools.supported('openai/gpt-6'))

    def test_schema_declares_every_operation_once(self):
        s = self.session()
        schema = s.tool_schema()
        deferred = [t['function']['name'] for t in schema if t.get('defer_loading')]
        self.assertTrue(set(OPERATIONS) <= set(deferred))
        self.assertNotIn(document_catalog.DISPATCHER, [t['function']['name'] for t in schema])
        self.assertEqual(document_catalog.messages(s), [])
        before = json.dumps(schema)
        s.conv.canvas.open_blank(100, 100)
        self.assertEqual(json.dumps(s.tool_schema()), before)          # tools[] stays put
        self.assertLess(s.tool_overhead(), __import__('ctxagent.tokens', fromlist=['x']).count_tools(schema))
        catalog = self.session('anthropic/claude-opus-5.5')
        self.assertIn(document_catalog.DISPATCHER, [t['function']['name'] for t in catalog.tool_schema()])
        self.assertFalse(any(t.get('defer_loading') for t in catalog.tool_schema()))

    def test_sync_adds_removes_and_readds_after_compaction(self):
        s = self.session()
        self.assertIsNone(dynamic_tools.sync(s))
        s.conv.canvas.open_blank(100, 100)
        seg = dynamic_tools.sync(s)
        self.assertEqual(seg.kind, K_TOOL_CHANGE)
        self.assertEqual(set(seg.observation['add']), set(OPERATIONS))
        self.assertIsNone(dynamic_tools.sync(s))
        self.assertGreater(dynamic_tools.surfaced_tokens(s), 100)
        s.conv.segments.remove(seg)                                     # archived by compaction
        again = dynamic_tools.sync(s)
        self.assertEqual(set(again.observation['add']), set(OPERATIONS))
        s.conv.canvas.close()
        gone = dynamic_tools.sync(s)
        self.assertEqual(set(gone.observation['remove']), set(OPERATIONS))
        self.assertEqual(dynamic_tools.surfaced(s.conv), set())
        clone = Conversation.from_dict(json.loads(s.conv.dumps()))
        self.assertEqual([x.observation for x in clone.segments if x.kind == K_TOOL_CHANGE],
                         [again.observation, gone.observation])

    def test_turn_records_the_change_and_runs_a_native_call(self):
        s = self.session()
        s.conv.canvas.open_blank(120, 80)
        seen = []

        async def stream(**kw):
            seen.append(kw)
            if len(seen) == 1:
                yield {'type': 'tool_calls', 'tool_calls': [{'id': 'n1', 'type': 'function', 'function': {
                    'name': 'canvas_draw', 'arguments': json.dumps({'ops': [{'type': 'marker', 'at': [5, 5]}]})}}]}
            else:
                yield {'type': 'text', 'text': 'drawn'}
            yield {'type': 'done', 'finish_reason': 'stop'}

        async def run():
            return [ev async for ev in s.run_turn('draw')]
        import asyncio
        with patch('ctxagent.agent.llm.stream_chat', stream):
            asyncio.run(run())
        first = seen[0]
        changes = [m for m in first['messages'] if anthropic_api.is_tool_change(m)]
        self.assertEqual(len(changes), 1)
        self.assertEqual(first['messages'][-2], changes[0])           # then the canvas trailer
        self.assertTrue(any(t.get('defer_loading') and t['function']['name'] == 'canvas_draw'
                            for t in first['tools']))
        self.assertEqual(s.conv.canvas.current()['version'], 2)       # the native call ran
        self.assertEqual(len([m for m in seen[1]['messages'] if anthropic_api.is_tool_change(m)]), 1)
        self.assertEqual(sum(1 for x in s.conv.segments if x.kind == K_TOOL_CHANGE), 1)
        # The request as the API gets it: change after the user turn, before the call.
        kw = request(first['messages'], first['tools'], uncached_tail=first['uncached_tail'])
        self.assertEqual([m['role'] for m in kw['messages']], ['user', 'system'])

    def test_catalog_path_unchanged(self):
        s = self.session('anthropic/claude-opus-5.5')
        s.conv.canvas.open_blank(100, 100)
        self.assertIsNone(dynamic_tools.sync(s) if dynamic_tools.active(s) else None)
        self.assertTrue(document_catalog.messages(s))
        self.assertFalse(dynamic_tools.active(s))


if __name__ == '__main__':
    unittest.main(verbosity=2)
