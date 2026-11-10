"""Live views never land between a tool call and its results.

Seen in a real chat on the direct Anthropic API (2026-11): HTTP 400 "messages.2: `tool_use`
ids were found without `tool_result` blocks immediately after". The engine froze the cached
prefix by token count, right after a large write_file call; the open document views, which go
right behind the frozen prefix, then sat between that call and its result. Every later request
of the chat failed the same way. Offline: requests are captured from a scripted stream.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxagent import anthropic_api, config
from ctxagent.agent import AgentSession
from ctxagent.compaction.summarize import PrefixShare
from ctxagent.config import CompactionConfig, SessionConfig
from ctxagent.segments import (K_ASSISTANT, K_SYSTEM, K_TOOL_CALL, K_TOOL_RESULT, K_USER, Conversation,
                               wire_problems)

DIRECT = 'claude-opus-5-5'


def anthropic_problems(kw: dict) -> list[str]:
    """Anthropic's rule: every tool_use answered by tool_result blocks at the START of the next message."""
    with patch('ctxagent.anthropic_api.expand_messages', side_effect=lambda m: m):
        req = anthropic_api.build_request(model=kw.get('model', DIRECT), messages=kw['messages'], tools=kw['tools'],
                                          max_tokens=100, reasoning_effort='off',
                                          cache_anchors=kw.get('cache_anchors') or (),
                                          cache_ttl='5m', uncached_tail=kw.get('uncached_tail', 0))
    out, msgs = [], req['messages']
    for i, m in enumerate(msgs):
        ids = [b['id'] for b in m['content'] if m['role'] == 'assistant' and b.get('type') == 'tool_use']
        if not ids:
            continue
        nxt = msgs[i + 1] if i + 1 < len(msgs) else {'role': None, 'content': []}
        lead = []
        for b in nxt['content'] if nxt['role'] == 'user' else []:
            if b.get('type') != 'tool_result':
                break
            lead.append(b['tool_use_id'])
        if set(ids) - set(lead):
            out.append(f'messages.{i}: {ids} not answered first by the next message')
    return out


def call(cid, name='write_file', args=None):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args or {})}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = patch.object(config, 'DATA_DIR', Path(self.tmp.name) / 'data')
        p.start()
        self.addCleanup(p.stop)
        self.work = Path(self.tmp.name) / 'work'
        self.work.mkdir()
        (self.work / 'game.gd').write_text('extends Node\n' + 'var x = 1\n' * 200, encoding='utf-8')

    def session(self, model=DIRECT, conv=None):
        cfg = SessionConfig(model=model, compaction=CompactionConfig(), workdirs=[str(self.work)])
        s = AgentSession('pairing', cfg, conv or Conversation())
        s.persist = lambda: None
        s.conv.workspace.open(s.sandbox(), 'game.gd')
        return s

    def run_turn(self, s, script):
        seen = []

        async def stream(**kw):
            seen.append(kw)
            for ev in script(len(seen), s):
                yield ev
            yield {'type': 'done', 'finish_reason': 'stop'}

        async def go():
            return [ev async for ev in s.run_turn('go')]
        with patch('ctxagent.agent.llm.stream_chat', stream):
            events = asyncio.run(go())
        return seen, events


class TurnTests(Base):
    def big_write(self, n, s):
        if n == 1:   # a call far larger than the 50k hot zone
            return [{'type': 'tool_calls', 'tool_calls': [call('toolu_big', args={
                'path': 'big.gd', 'content': 'func f():\n\tpass\n' * 20000})]}]
        return [{'type': 'text', 'text': 'done'}]

    def test_a_large_call_keeps_its_result_right_behind_it(self):
        for model in (DIRECT, 'anthropic/claude-opus-5.5'):
            with self.subTest(model=model):
                s = self.session(model)
                seen, _ = self.run_turn(s, self.big_write)
                kinds = [x.kind for x in s.conv.segments]
                self.assertIn(K_TOOL_CALL, kinds)
                for kw in seen:
                    self.assertEqual(wire_problems(kw['messages']), [])
                    self.assertEqual(anthropic_problems(kw), [])
                second = [m['role'] for m in seen[1]['messages']]
                self.assertEqual(second[second.index('assistant') + 1], 'tool')

    def test_a_chat_stored_frozen_inside_a_group_still_sends_valid_requests(self):
        # Stored by an older version: the frozen prefix ended right after the call.
        old = Conversation()
        old.new('system', K_SYSTEM, 'system')
        old.new('user', K_USER, 'write it')
        old.new('assistant', K_TOOL_CALL, '', tool_calls=[call('toolu_old')])
        old.new('tool', K_TOOL_RESULT, 'written', tool_call_id='toolu_old', tool_name='write_file')
        old.new('assistant', K_ASSISTANT, 'done')
        stored = old.to_dict()
        stored['frozen_upto'] = 3                                          # system, user, call
        for d in stored['segments'][:3]:
            d['frozen'] = True
        conv = Conversation.from_dict(json.loads(json.dumps(stored)))
        self.assertFalse(hasattr(conv, 'frozen_upto'))
        s = self.session(conv=conv)
        seen, _ = self.run_turn(s, lambda n, s: [{'type': 'text', 'text': 'ok'}])
        msgs = seen[0]['messages']
        self.assertEqual(wire_problems(msgs), [])
        self.assertEqual(anthropic_problems(seen[0]), [])
        roles = [m['role'] for m in msgs]
        tool_at = roles.index('tool')
        self.assertEqual(roles[tool_at - 1], 'assistant')
        self.assertTrue(any('game.gd' in json.dumps(m) for m in msgs))

    def test_settings_saved_during_a_turn_apply_at_its_next_request(self):
        s = self.session()
        before = s.cfg.max_output_tokens

        def script(n, s):
            if n == 1:
                cfg = SessionConfig.from_dict({**s.cfg.to_dict(), 'max_output_tokens': 12_345})
                self.assertTrue(s.set_config(cfg))                      # the turn is running
                self.assertEqual(s.cfg.max_output_tokens, before)        # this request keeps its settings
                return [{'type': 'tool_calls', 'tool_calls': [call('t1', 'list_dir', {'path': '.'})]}]
            return [{'type': 'text', 'text': 'done'}]
        seen, events = self.run_turn(s, script)
        self.assertEqual(before, 0)                                      # unlimited: the model's maximum
        self.assertEqual([kw['max_tokens'] for kw in seen], [128_000, 12_345])
        self.assertEqual([e['config']['max_output_tokens'] for e in events if e['type'] == 'config_applied'],
                         [12_345])
        self.assertIsNone(s.pending_config)
        self.assertFalse(s.set_config(SessionConfig.from_dict(s.cfg.to_dict())))   # idle: at once


class UnitTests(unittest.TestCase):
    def conv(self):
        c = Conversation()
        c.new('system', K_SYSTEM, 'sys')
        c.new('user', K_USER, 'go')
        c.new('assistant', K_TOOL_CALL, '', tool_calls=[call('a')])
        c.new('tool', K_TOOL_RESULT, 'r', tool_call_id='a')
        c.new('assistant', K_ASSISTANT, 'done')
        return c

    def test_group_start_snaps_back_to_the_call(self):
        c = self.conv()
        self.assertEqual(c.group_start(3), 2)
        self.assertEqual(c.group_start(2), 2)
        self.assertEqual(c.group_start(4), 4)

    def test_repair_cut_back_inside_a_group_stays_valid(self):
        c = self.conv()
        del c.segments[3]                                   # the result went missing
        self.assertTrue(c.repair_tool_pairs())
        self.assertEqual(wire_problems(c.to_messages()), [])

    def test_sources_follow_the_message_order(self):
        c = self.conv()
        c.segments.insert(3, c.segments[1].__class__(seq=99, role='user', kind=K_USER, body='note'))
        sources = []
        msgs = c.to_messages(sources)
        self.assertEqual(len(sources), len(msgs))
        self.assertEqual([i for i, _ in sources], [0, 1, 2, 4, 3, 5])   # the result right after its call
        del c.segments[4]                                                # no result: a placeholder
        sources = []
        c.to_messages(sources)
        self.assertEqual(sources[3], (2, None))

    def test_converter_puts_results_first(self):
        messages = [{'role': 'user', 'content': 'go'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [call('x'), call('y')]},
                    {'role': 'user', 'content': 'DOCUMENT VIEW'},
                    {'role': 'tool', 'tool_call_id': 'x', 'content': 'rx'}]
        _, out = anthropic_api.convert_messages(messages)
        self.assertEqual([b['type'] for b in out[2]['content']], ['tool_result', 'tool_result', 'text'])
        self.assertEqual([b['tool_use_id'] for b in out[2]['content'][:2]], ['x', 'y'])   # y: placeholder

    def test_summary_prefix_is_not_cut_inside_a_group(self):
        frame = {'messages': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'},
                              {'role': 'assistant', 'content': None, 'tool_calls': [call('a')]},
                              {'role': 'tool', 'tool_call_id': 'a', 'content': 'r'},
                              {'role': 'user', 'content': 'more'}],
                 'anchors': [2], 'sent_at': 0}
        share = PrefixShare(frame, None, 100)
        self.assertEqual(share.upto(2), 5)                  # not 3: that ends on the open call
        frame['anchors'] = [3]
        self.assertEqual(share.upto(2), 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
