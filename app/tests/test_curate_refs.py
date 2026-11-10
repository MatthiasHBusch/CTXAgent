"""curate_context since 2026-11-10: no listing, no ids, no frozen part.

* curate_context(reference1, replacement1, ...): a reference counts back from the newest item
  before the call (1 = the newest tool result, tool call or answer), in the order the model
  reads them; text beside tool calls, user messages and the running batch are not items (a
  live check on Opus 5.5 counted that way, 4 of 4).
* A replaced result or answer becomes a stub with the agent's text and a recall handle; a
  replaced call keeps its name, its arguments become {"replaced", "recall_segment_id"}.
* Without pairs nothing changes. recall returns the original as a new result and leaves
  the history as it is.
* The hint is one line; the system prompt explains the review.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxagent import agent as agentmod, config
from ctxagent.agent import AgentSession
from ctxagent.compaction import curate
from ctxagent.compaction.engine import CompactionEngine
from ctxagent.config import CompactionConfig, SessionConfig
from ctxagent.segments import (Conversation, K_ASSISTANT, K_SYSTEM, K_TOOL_CALL, K_TOOL_RESULT, K_USER,
                               S_ORIGINAL)


def chat() -> Conversation:
    c = Conversation()
    c.new('system', K_SYSTEM, 'system prompt')
    c.new('user', K_USER, 'build the thing')

    def batch(calls, results, text=''):
        c.new('assistant', K_TOOL_CALL, text, tool_calls=[
            {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
            for cid, name, args in calls])
        for (cid, name, _), body in zip(calls, results):
            c.new('tool', K_TOOL_RESULT, body, tool_call_id=cid, tool_name=name)
    batch([('c1', 'run_command', {'command': 'make'})], ['BUILD LOG\n' + 'line\n' * 2000], 'Building.')
    batch([('c2', 'write_file', {'path': 'a.py', 'content': 'print(1)\n' * 400}),
           ('c3', 'grep_search', {'pattern': 'foo'})], ['wrote a.py', 'a.py:1 foo'])
    c.new('assistant', K_ASSISTANT, 'Built it and wrote a.py.')
    c.new('user', K_USER, 'now clean up')
    batch([('c4', 'curate_context', {})], [])          # the batch running the curate call
    return c


class Items(unittest.TestCase):
    def test_counted_back_from_the_newest_in_reading_order(self):
        found = [i.describe() for i in reversed(curate.items(chat()))]
        self.assertEqual([d.split(' (')[0] for d in found], [
            'answer', 'grep_search result', 'write_file result', 'grep_search call', 'write_file call',
            'run_command result', 'run_command call'])          # 'Building.' beside the call is no item
        self.assertIn('Built it', found[0])

    def test_pairs_in_any_shape(self):
        for pairs in ([3, 'a', 5, 'b'], [[3, 'a'], [5, 'b']], [{'reference': 3, 'replacement': 'a'},
                      {'reference': '5', 'replacement': 'b'}], {'3': 'a', '#5': 'b'}):
            self.assertEqual(curate.parse_pairs(pairs)[0], [(3, 'a'), (5, 'b')], pairs)
        todo, problems = curate.parse_pairs([3, 'a', 'x', 'b', 7])
        self.assertEqual(todo, [(3, 'a')])
        self.assertEqual(len(problems), 2)


class Replace(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(config, 'DATA_DIR', Path(tmp.name))
        p.start()
        self.addCleanup(p.stop)

    def test_without_pairs_nothing_changes(self):
        c = chat()
        c.compaction_state['batches'] = 17
        before = c.dumps()
        rep = curate.curate(c, 'chat', None)
        self.assertEqual(c.dumps(), before)
        self.assertIn('nothing replaced', curate.report_text(rep))
        self.assertEqual(curate.curate(c, 'chat', [])['replaced'], [])
        self.assertEqual(c.compaction_state['batches'], 17)

    def test_a_result_and_a_call(self):
        c = chat()
        c.compaction_state['batches'] = 17
        log = next(s for s in c.segments if s.tool_name == 'run_command' and s.kind == K_TOOL_RESULT)
        call = next(s for s in c.segments if s.kind == K_TOOL_CALL and s.tool_calls[0]['id'] == 'c2')
        rep = curate.curate(c, 'chat', [6, 'make: BUILD_OK, 0 warnings', 5, 'wrote a.py (400 lines)'])
        self.assertEqual([r.split(':')[0] for r in rep['replaced']], ['5', '6'])
        self.assertIn('write_file call', rep['replaced'][0])
        self.assertGreater(rep['freed'], 2000)
        self.assertTrue(log.body.startswith('[replaced: make: BUILD_OK, 0 warnings | recall(segment_id="'))
        args = json.loads(call.tool_calls[0]['function']['arguments'])
        self.assertEqual(args, {'replaced': 'wrote a.py (400 lines)', 'recall_segment_id': call.id})
        self.assertEqual(json.loads(call.tool_calls[1]['function']['arguments']), {'pattern': 'foo'})
        self.assertEqual(c.compaction_state['batches'], 0)
        msgs = c.to_messages()                             # still a valid request
        self.assertEqual(sum(1 for m in msgs if m.get('role') == 'tool'), 4)

    def test_problems_are_reported_and_nothing_else_happens(self):
        c = chat()
        rep = curate.curate(c, 'chat', [40, 'x', 'first', 'y'])
        self.assertEqual(rep['replaced'], [])
        text = curate.report_text(rep)
        self.assertIn('there are 7 items', text)
        self.assertIn("'first' is not a reference", text)


class Session(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(config, 'DATA_DIR', Path(tmp.name))
        p.start()
        self.addCleanup(p.stop)
        cfg = SessionConfig(compaction=CompactionConfig(rewrite_policy='agent'), workdirs=[tmp.name])
        self.s = AgentSession('curate', cfg, chat())
        self.s.persist = lambda: None

    def run_tool(self, name, args):
        return asyncio.run(self.s._exec_known_tool(name, args))

    def test_tool_and_recall(self):
        res = self.run_tool('curate_context', {'pairs': [6, 'make: BUILD_OK']})
        self.assertFalse(res.error)
        self.assertIn('6: run_command result (BUILD LOG)', res.text)
        log = next(s for s in self.s.conv.segments if s.tool_name == 'run_command' and s.kind == K_TOOL_RESULT)
        stub = log.body
        back = self.run_tool('recall', {'segment_id': log.id})
        self.assertIn('BUILD LOG', back.text)
        self.assertTrue(back.pinned)
        self.assertEqual(log.body, stub)                   # the history keeps its bytes
        self.assertNotEqual(log.state, S_ORIGINAL)

    def test_recall_of_a_replaced_call_returns_its_arguments(self):
        self.run_tool('curate_context', {'pairs': [5, 'wrote a.py']})
        call = next(s for s in self.s.conv.segments if s.kind == K_TOOL_CALL and s.tool_calls[0]['id'] == 'c2')
        back = self.run_tool('recall', {'segment_id': call.id})
        self.assertIn('[tool call write_file]', back.text)
        self.assertIn('print(1)', back.text)

    def test_no_arguments_and_the_old_schema(self):
        self.assertIn('nothing replaced', self.run_tool('curate_context', {}).text)
        old = self.run_tool('curate_context', {'drop': ['abc']})
        self.assertTrue(old.error)
        self.assertIn('counting back from the newest', old.text)
        self.assertIn('3: write_file result', self.run_tool('curate_context', {'3': 'wrote a.py'}).text)

    def test_schema_and_prompt(self):
        spec = next(t for t in self.s.tool_schema() if t['function']['name'] == 'curate_context')['function']
        self.assertTrue(spec['description'].startswith(
            'curate_context(reference1, replacement1, reference2, replacement2, as many following as you like...)'))
        self.assertEqual(list(spec['parameters']['properties']), ['pairs'])
        prompt = ' '.join(agentmod.build_system_prompt(self.s.cfg).split())
        self.assertIn('References count back from the newest', prompt)
        self.assertIn('context review is due', prompt)


class Hint(unittest.IsolatedAsyncioTestCase):
    async def test_the_hint_is_one_line(self):
        c = chat()
        del c.segments[-3:]                                # back to the write_file/grep batch
        cfg = CompactionConfig(rewrite_policy='agent', deep_enabled=False, rewrite_interval_batches=2)
        c.compaction_state['batches'] = 5
        rep = await CompactionEngine('h', cfg, None).run(c, 1_000_000)
        self.assertEqual(rep['review_hint'], curate.HINT)
        self.assertTrue(c.segments[-1].body.endswith('\n\n[harness: context review due]'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
