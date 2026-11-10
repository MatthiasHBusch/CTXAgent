"""compact_context(summary, until): the cut counts back like a curate_context reference.

* `until` = n replaces everything after the system prompt up to and including item n, where
  1 is the newest tool result, tool call or answer before the call (curate.items).
* A tool batch is not split: a cut inside one moves before the batch.
* Without `until` the cut is the newest user message, which stays.
* `summary` and `until` are the tool's only arguments; `keep_from`, `keep_ids`, `dry_run`
  and `index` went on 2026-11-10 (a model call naming one is told so).
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxagent import config
from ctxagent.agent import AgentSession
from ctxagent.compaction import manual
from ctxagent.config import CompactionConfig, SessionConfig
from ctxagent.segments import Conversation, K_ASSISTANT, K_SUMMARY, K_SYSTEM, K_TOOL_CALL, K_TOOL_RESULT, K_USER

SUMMARY = 'Handoff: the build works (make: BUILD_OK) and a.py is written.'


def chat() -> Conversation:
    """Items, newest first: 1 answer, 2 grep_search result, 3 write_file result, 4 grep_search
    call, 5 write_file call, 6 run_command result, 7 run_command call."""
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
    batch([('c4', 'compact_context', {})], [])         # the batch running the compact call
    return c


def shape(c: Conversation) -> list[str]:
    return [s.tool_name or (s.tool_calls[0]['id'] if s.kind == K_TOOL_CALL else s.kind) for s in c.segments]


class Until(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(config, 'DATA_DIR', Path(tmp.name))
        p.start()
        self.addCleanup(p.stop)

    def test_up_to_and_including_the_item(self):
        c = chat()
        rep = manual.compact(c, 't', summary=SUMMARY, until=1)          # the answer goes too
        self.assertEqual(shape(c), [K_SYSTEM, K_SUMMARY, K_USER, 'c4'])
        self.assertEqual(rep['cut'], 'item 1')
        self.assertIn('up to and including item 1', c.segments[1].body)
        self.assertIn(SUMMARY, c.segments[1].body)
        c = chat()
        manual.compact(c, 't', summary=SUMMARY, until=2)                # the last result of a batch
        self.assertEqual(shape(c), [K_SYSTEM, K_SUMMARY, K_ASSISTANT, K_USER, 'c4'])

    def test_a_tool_batch_is_not_split(self):
        for n in (3, 4, 5):                                             # a result or a call inside the batch
            c = chat()
            rep = manual.compact(c, 't', summary=SUMMARY, until=n)
            self.assertEqual(shape(c), [K_SYSTEM, K_SUMMARY, 'c2', 'write_file', 'grep_search',
                                        K_ASSISTANT, K_USER, 'c4'], n)
            self.assertIn(f'the tool batch of item {n}', rep['cut'])
            self.assertIn('before the tool batch', c.segments[1].body)

    def test_the_replaced_results_stay_recallable(self):
        c = chat()
        log = next(s for s in c.segments if s.tool_name == 'run_command' and s.kind == K_TOOL_RESULT)
        manual.compact(c, 't', summary=SUMMARY, until=6)
        self.assertIn(log.id, c.segments[1].body)                       # the index names it
        self.assertIsNotNone(c.archived_by_id(log.id))

    def test_without_until_the_newest_user_message_stays(self):
        c = chat()
        rep = manual.compact(c, 't', summary=SUMMARY)
        self.assertEqual(shape(c), [K_SYSTEM, K_SUMMARY, K_USER, 'c4'])
        self.assertEqual(rep['cut'], 'the last user message')

    def test_problems(self):
        for bad in (8, 0, 'x'):
            with self.assertRaises(ValueError):
                manual.compact(chat(), 't', summary=SUMMARY, until=bad)
        with self.assertRaises(ValueError):
            manual.compact(chat(), 't', summary='   ', until=1)         # nothing would remain
        c = chat()
        with self.assertRaises(ValueError):                             # item 7 is the first call: its batch
            manual.compact(c, 't', summary='x' * 3000, until=7)         # stays, the rest is smaller than the summary


class Session(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(config, 'DATA_DIR', Path(tmp.name))
        p.start()
        self.addCleanup(p.stop)
        cfg = SessionConfig(compaction=CompactionConfig(rewrite_policy='agent'), workdirs=[tmp.name])
        self.s = AgentSession('compact', cfg, chat())
        self.s.persist = lambda: None

    def run_tool(self, name, args):
        # _exec_tool_inner is the path of a model call: it checks the arguments first
        # (tools.prepare_arguments), which _exec_known_tool alone would skip.
        return asyncio.run(self.s._exec_tool_inner(name, args))

    def test_schema(self):
        spec = next(t for t in self.s.tool_schema() if t['function']['name'] == 'compact_context')['function']
        self.assertEqual(list(spec['parameters']['properties']), ['summary', 'until'])
        self.assertIn('curate_context reference', spec['description'])

    def test_tool(self):
        res = self.run_tool('compact_context', {'summary': SUMMARY, 'until': 2})
        self.assertFalse(res.error, res.text)
        # (the session gave the open call c4 a result when it loaded the chat: one more segment)
        self.assertEqual(shape(self.s.conv)[:5], [K_SYSTEM, K_SUMMARY, K_ASSISTANT, K_USER, 'c4'])
        bad = self.run_tool('compact_context', {'summary': SUMMARY, 'until': 40})
        self.assertTrue(bad.error)
        self.assertIn('compact_context: until=40', bad.text)

    def test_the_earlier_arguments_are_gone(self):
        before = shape(self.s.conv)
        for name, value in (('dry_run', True), ('keep_from', 'current_batch'), ('keep_ids', []), ('index', False)):
            res = self.run_tool('compact_context', {'summary': SUMMARY, name: value})
            self.assertTrue(res.error)
            self.assertIn(f"unknown argument '{name}'", res.text)
        self.assertEqual(shape(self.s.conv)[:len(before)], before)      # nothing was compacted
        with self.assertRaises(TypeError):
            manual.compact(self.s.conv, 't', summary=SUMMARY, dry_run=True)

    def test_an_argument_the_tool_does_not_have_is_named(self):
        res = self.run_tool('compact_context', {'summary': SUMMARY, 'untill': 2})
        self.assertTrue(res.error)
        self.assertIn("unknown argument 'untill'", res.text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
