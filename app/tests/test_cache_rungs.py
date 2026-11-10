"""Cache points since the frozen prefix went (2026-11-10): every request marks the static
system prompt, up to two rungs in the history and the tail (cache_policy.plan_rungs).

* A rung is the last unchanged message before a change, or the previous request's tail once
  the history behind the newest rung has grown by RUNG_SPACING_TOKENS (or the request appended
  more messages than the 20-block look-back reaches).
* Live views go behind the previous request's end when they appear or change; the message
  before them stays a rung while they are open.
* The request after a curate marks the message before the oldest replaced item.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxagent import cache_policy as cp, config, tools as toolmod
from ctxagent.agent import AgentSession
from ctxagent.config import CompactionConfig, SessionConfig
from ctxagent.segments import Conversation

BIG = cp.RUNG_SPACING_TOKENS // 4 + 1


class Planner:
    """Feeds plan_rungs the way agent._run_turn does, one request at a time."""

    def __init__(self):
        self.history = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'task'}]
        self.views: list = []
        self.prior = None

    def request(self):
        hashes = [cp.digest(m) for m in self.history]
        sizes = [BIG if m['role'] == 'tool' else 50 for m in self.history]
        plan = cp.plan_rungs(self.prior, hashes, sizes, cp.digest(self.views), len(self.views))
        self.prior = {'history_hashes': hashes, 'rungs': plan['rungs'], 'views_at': plan['views_at'],
                      'views_hash': cp.digest(self.views), 'views_count': len(self.views),
                      'history_tail': len(self.history) - 1}
        return plan

    def batch(self, k):
        self.history += [{'role': 'assistant', 'content': f'call {k}'}, {'role': 'tool', 'content': f'result {k}'}]


class PlanRungs(unittest.TestCase):
    def test_a_rung_every_spacing_tokens_and_at_most_two(self):
        p = Planner()
        seen = []
        for k in range(20):
            p.batch(k)
            seen.append(p.request()['rungs'])
        self.assertEqual(seen[0], [])
        self.assertTrue(all(len(r) <= cp.MAX_RUNGS for r in seen))
        newest = [r[-1] for r in seen if r]
        self.assertEqual(newest, sorted(newest))           # rungs follow the history
        gaps = {b - a for a, b in zip(sorted(set(newest)), sorted(set(newest))[1:])}
        self.assertEqual(gaps, {8})                        # four batches of BIG tokens

    def test_a_change_puts_a_rung_right_before_it(self):
        p = Planner()
        for k in range(12):
            p.batch(k)
            p.request()
        p.history[15] = {'role': 'tool', 'content': 'replaced'}
        plan = p.request()
        self.assertEqual(plan['edit_at'], 14)
        self.assertIn(14, plan['rungs'])
        self.assertTrue(all(r < 15 for r in plan['rungs']))
        p.batch(99)
        self.assertIn(14, p.request()['rungs'])            # and it stays a rung

    def test_a_long_request_keeps_the_previous_tail(self):
        p = Planner()
        p.batch(0)
        p.request()
        p.history += [{'role': 'user', 'content': f'n{i}'} for i in range(cp.LOOKBACK_GUARD + 1)]
        self.assertEqual(p.request()['rungs'], [3])

    def test_views_go_behind_the_previous_end_and_keep_their_rung(self):
        p = Planner()
        for k in range(10):
            p.batch(k)
            plan = p.request()
        self.assertEqual(plan['views_at'], max(plan['rungs']))   # without views: the newest rung
        at = len(p.history) - 1
        p.batch(10)
        p.views = [{'role': 'user', 'content': 'doc v1'}]
        plan = p.request()
        self.assertEqual(plan['views_at'], at)             # appearing views: behind the previous end
        self.assertIn(at, plan['rungs'])
        for k in range(11, 30):
            p.batch(k)
            plan = p.request()
            self.assertEqual(plan['views_at'], at)         # unchanged views stay put
            self.assertIn(at, plan['rungs'])               # and the message before them stays marked
            self.assertLessEqual(len(plan['rungs']), cp.MAX_RUNGS)
        self.assertGreater(max(plan['rungs']), at)         # the other rung follows the history
        end = len(p.history) - 1
        p.batch(30)
        p.views = [{'role': 'user', 'content': 'doc v2'}]
        plan = p.request()
        self.assertEqual(plan['edit_at'], at)              # a change re-writes from the old place
        self.assertEqual(plan['views_at'], end)            # and the views move behind the previous end
        self.assertEqual(plan['rungs'], [at, end])

    def test_a_change_before_the_views_moves_them_too(self):
        p = Planner()
        for k in range(6):
            p.batch(k)
            p.request()
        p.batch(6)
        p.views = [{'role': 'user', 'content': 'doc'}]
        at = p.request()['views_at']
        for k in range(7, 10):
            p.batch(k)
            p.request()
        end = len(p.history) - 1
        self.assertLess(3, at)
        p.history[3] = {'role': 'tool', 'content': 'replaced'}
        plan = p.request()
        self.assertEqual((plan['edit_at'], plan['views_at'], plan['rungs']), (2, end, [2, end]))

    def test_views_in_a_shrunk_history_go_behind_the_newest_rung(self):
        p = Planner()
        for k in range(10):
            p.batch(k)
            p.request()
        p.views = [{'role': 'user', 'content': 'doc'}]
        p.batch(10)
        p.request()
        p.history = p.history[:1] + [{'role': 'user', 'content': 'summary'}] + p.history[-4:]
        plan = p.request()
        self.assertEqual((plan['views_at'], plan['rungs']), (0, []))
        p.batch(11)
        self.assertEqual(p.request()['views_at'], 0)       # they stay until they change
        end = len(p.history) - 1
        p.batch(12)
        p.views = [{'role': 'user', 'content': 'doc v2'}]
        self.assertEqual(p.request()['views_at'], end)

    def test_no_prior_cache_no_rungs(self):
        p = Planner()
        for k in range(10):
            p.batch(k)
            p.request()
        p.prior = None                                     # model, tools or reasoning changed
        self.assertEqual(p.request()['rungs'], [])

    def test_apply_marks_at_most_four(self):
        msgs = [{'role': 'system', 'content': 's'}] + [{'role': 'user', 'content': str(i)} for i in range(9)]
        out, anchors = cp.apply(msgs, 'anthropic/claude-opus-5.5', [3, 6, 8], tail=True)
        marked = [i for i, m in enumerate(out) if isinstance(m['content'], list)]
        self.assertEqual(anchors, [0, 6, 8])
        self.assertEqual(marked, [0, 6, 8, 9])


def call(cid, name, args):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


class AfterACurate(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(config, 'DATA_DIR', Path(tmp.name))
        p.start()
        self.addCleanup(p.stop)
        cfg = SessionConfig(model='claude-opus-5-5', workdirs=[tmp.name], compaction=CompactionConfig(
            rewrite_policy='agent', deep_enabled=False, cache_keepalive=False))
        self.s = AgentSession('rungs', cfg, Conversation())
        self.s.persist = lambda: None

    def test_the_next_request_marks_the_message_before_the_replaced_one(self):
        replies = [call('c1', 'run_command', {'command': 'make'}), call('c2', 'run_command', {'command': 'test'}),
                   call('c3', 'curate_context', {'pairs': [3, 'make: BUILD_OK']}), None]
        sent = []

        async def stream(**kw):
            sent.append(kw)
            reply = replies[len(sent) - 1]
            if reply:
                yield {'type': 'tool_calls', 'tool_calls': [reply]}
            else:
                yield {'type': 'text', 'text': 'done'}
            yield {'type': 'done', 'finish_reason': 'stop'}

        async def batch(session, tool_calls, finish_reason=None):
            for tc in tool_calls:
                name = tc['function']['name']
                args = json.loads(tc['function']['arguments'])
                yield 'start', tc, name, args, None, 0
                res = (await session._exec_known_tool(name, args) if name == 'curate_context'
                       else toolmod.ToolResult(text='LOG\n' + 'line\n' * 3000))
                yield 'result', tc, name, args, res, 1

        async def go():
            [ev async for ev in self.s.run_turn('build it')]
        with patch('ctxagent.agent.llm.stream_chat', stream), patch('ctxagent.agent.execute_batch', batch):
            asyncio.run(go())
        self.assertEqual(len(sent), 4)
        last = sent[-1]
        replaced = next(i for i, m in enumerate(last['messages']) if m.get('role') == 'tool'
                        and str(m.get('content', '')).startswith('[replaced: make: BUILD_OK'))
        self.assertEqual(replaced, 3)
        self.assertIn(replaced - 1, last['cache_anchors'])
        self.assertEqual(self.s.conv.cache_state['edit_at'], replaced - 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
