"""Cache points since the frozen prefix went (2026-11-10): every request marks the static
system prompt, up to two rungs in the history and the tail (cache_policy.plan_rungs).

* While nothing changes, the near rung lies about NEAR_RUNG_TOKENS before the end of the
  history (in a short one in its middle) and the far rung halfway to it; a rung stays while
  it is within the slack of its place. A request that appends more messages than the
  20-block look-back reaches has the previous request's tail as its near rung.
* A change request marks the newest rung before the change and the last unchanged message.
* Live views go behind the previous request's end when they appear or change; the message
  before them stays a rung while they are open. Without a previous end (a compaction, a
  tool change) they go before the newest step.
* A view that changes again within two requests is sent at the end of the request, behind
  the tail mark, until it has been unchanged for two requests (cache_policy.restless_views).
* The request after a curate marks the message before the oldest replaced item.
"""
import asyncio
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxagent import cache_policy as cp, config, tools as toolmod
from ctxagent.agent import AgentSession
from ctxagent.config import CompactionConfig, SessionConfig
from ctxagent.segments import Conversation

BIG = 5_000


class Planner:
    """Feeds plan_rungs the way agent._run_turn does, one request at a time."""

    def __init__(self):
        self.history = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'task'}]
        self.views: list = []
        self.prior = None

    def ends(self):
        """Tokens up to the end of each message."""
        return list(itertools.accumulate(BIG if m['role'] == 'tool' else 50 for m in self.history))

    def request(self):
        hashes = [cp.digest(m) for m in self.history]
        sizes = [BIG if m['role'] == 'tool' else 50 for m in self.history]
        newest = max(i for i, m in enumerate(self.history) if m['role'] != 'tool')
        plan = cp.plan_rungs(self.prior, hashes, sizes, cp.digest(self.views), len(self.views), newest - 1)
        self.prior = {'history_hashes': hashes, 'rungs': plan['rungs'], 'views_at': plan['views_at'],
                      'views_hash': cp.digest(self.views), 'views_count': len(self.views),
                      'history_tail': len(self.history) - 1}
        return plan

    def batch(self, k):
        self.history += [{'role': 'assistant', 'content': f'call {k}'}, {'role': 'tool', 'content': f'result {k}'}]


class PlanRungs(unittest.TestCase):
    def test_one_rung_before_the_newest_part_and_one_halfway_to_it(self):
        p = Planner()
        seen = []
        for k in range(40):                                # 40 batches of BIG: 200k tokens
            p.batch(k)
            seen.append((p.request()['rungs'], p.ends()))
        self.assertTrue(all(len(rungs) <= cp.MAX_RUNGS for rungs, _ in seen))
        for rungs, ends in seen[30:]:                      # 155k and more
            far, near = rungs
            self.assertTrue(cp.NEAR_RUNG_TOKENS <= ends[-1] - ends[near] <= 1.5 * cp.NEAR_RUNG_TOKENS)
            halfway = (ends[-1] - cp.NEAR_RUNG_TOKENS) / 2
            self.assertTrue(.75 * halfway <= ends[far] <= 1.25 * halfway, (ends[far], halfway))
        # A rung serves several requests before it is placed anew: about 25k tokens of growth.
        self.assertLessEqual(len({rungs[0] for rungs, _ in seen[30:]}), 3)
        self.assertLessEqual(len({rungs[1] for rungs, _ in seen[30:]}), 3)

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
        self.assertIn(3, p.request()['rungs'])

    def test_views_go_behind_the_previous_end_and_keep_their_rung(self):
        p = Planner()
        for k in range(10):
            p.batch(k)
            plan = p.request()
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

    def test_views_in_a_shrunk_history_go_before_the_newest_step(self):
        p = Planner()
        for k in range(10):
            p.batch(k)
            p.request()
        p.views = [{'role': 'user', 'content': 'doc'}]
        p.batch(10)
        p.request()
        p.history = p.history[:1] + [{'role': 'user', 'content': 'summary'}] + p.history[-4:]
        plan = p.request()                                 # system, summary, call, result | call, result
        self.assertEqual((plan['views_at'], plan['rungs']), (3, [3]))
        p.batch(11)
        self.assertEqual(p.request()['views_at'], 3)       # they stay until they change
        end = len(p.history) - 1
        p.batch(12)
        p.views = [{'role': 'user', 'content': 'doc v2'}]
        self.assertEqual(p.request()['views_at'], end)

    def test_views_without_a_usable_previous_request_go_before_the_newest_step(self):
        p = Planner()
        for k in range(10):
            p.batch(k)
            p.request()
        p.views = [{'role': 'user', 'content': 'doc'}]
        p.batch(10)
        p.request()
        p.batch(11)
        p.prior = None                                     # the tools changed: nothing cached serves this
        plan = p.request()
        at = len(p.history) - 3                            # in front of the newest call and its result
        self.assertEqual(plan['views_at'], at)
        self.assertIn(at, plan['rungs'])

    def test_without_a_usable_cache_the_rungs_are_placed_by_position(self):
        p = Planner()
        for k in range(10):
            p.batch(k)
            p.request()
        p.prior = None                                     # model, tools or reasoning changed
        (far, near), ends = p.request()['rungs'], p.ends()   # this request writes the entries there
        self.assertTrue(.4 <= ends[near] / ends[-1] <= .5)   # 50k of history: the middle
        self.assertTrue(.15 <= ends[far] / ends[-1] <= .25)

    def test_a_change_reads_from_the_newest_rung_before_it(self):
        p = Planner()
        for k in range(40):
            p.batch(k)
            plan = p.request()
        far, near = plan['rungs']
        p.history[near - 1] = {'role': 'tool', 'content': 'replaced'}     # between the two rungs
        plan = p.request()
        self.assertEqual(plan['rungs'], [far, near - 2])   # read from the far rung, mark before the change
        p.batch(40)
        self.assertEqual(p.request()['rungs'], [far, near - 2])           # which is the near rung from now on

    def test_apply_marks_at_most_four(self):
        msgs = [{'role': 'system', 'content': 's'}] + [{'role': 'user', 'content': str(i)} for i in range(9)]
        out, anchors = cp.apply(msgs, 'anthropic/claude-opus-5.5', [3, 6, 8], tail=True)
        marked = [i for i, m in enumerate(out) if isinstance(m['content'], list)]
        self.assertEqual(anchors, [0, 6, 8])
        self.assertEqual(marked, [0, 6, 8, 9])


class RestlessViews(unittest.TestCase):
    def ends(self, versions):
        """One request per entry, the version of view 'd' in it: whether it went to the end."""
        prior, out = None, []
        for v in versions:
            hashes = {'d': f'v{v}', 'other': 'same'}
            calm, restless = cp.restless_views(prior, hashes)
            prior = {'view_hashes': hashes, 'views_calm': calm, 'views_restless': sorted(restless)}
            self.assertNotIn('other', restless)
            out.append('d' in restless)
        return out

    def test_a_view_edited_request_after_request_goes_to_the_end(self):
        # opened, edited three times, left alone: back into the history after two quiet requests
        self.assertEqual(self.ends([1, 2, 3, 4, 4, 4, 4]), [False, True, True, True, True, False, False])

    def test_a_test_run_between_two_edits_does_not_park_it(self):
        self.assertEqual(self.ends([1, 1, 2, 2, 3, 3, 4]), [False, False, True, True, True, True, True])

    def test_a_view_that_changes_now_and_then_stays_in_the_history(self):
        self.assertEqual(self.ends([1, 1, 1, 2, 2, 2, 3, 3, 3]), [False] * 9)


def call(cid, name, args):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


class SessionCase(unittest.TestCase):
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


class ARestlessViewInTheRequest(SessionCase):
    def test_a_document_edited_in_every_request_is_sent_behind_the_tail_mark(self):
        versions = [1, 2, 3, 3, 3]
        sent = []
        self.s.live_views = lambda: [('document d', {'role': 'user', 'content': f'DOC version {versions[len(sent)]}'})]

        async def stream(**kw):
            sent.append(kw)
            if len(sent) < len(versions):
                yield {'type': 'tool_calls', 'tool_calls': [call(f'c{len(sent)}', 'run_command', {'command': 'x'})]}
            else:
                yield {'type': 'text', 'text': 'done'}
            yield {'type': 'done', 'finish_reason': 'stop'}

        async def batch(session, tool_calls, finish_reason=None):
            for tc in tool_calls:
                yield 'start', tc, 'run_command', {}, None, 0
                yield 'result', tc, 'run_command', {}, toolmod.ToolResult(text='ok'), 1

        async def go():
            [ev async for ev in self.s.run_turn('edit it')]
        with patch('ctxagent.agent.llm.stream_chat', stream), patch('ctxagent.agent.execute_batch', batch):
            asyncio.run(go())
        where = [next(i for i, m in enumerate(kw['messages']) if str(m.get('content')).startswith('DOC'))
                 for kw in sent]
        self.assertEqual([kw['uncached_tail'] for kw in sent], [0, 1, 1, 1, 0])
        for kw, at in zip(sent[1:4], where[1:4]):
            self.assertEqual(at, len(kw['messages']) - 1)          # the last message, not cached
        # Quiet for two requests: in the history again, where the previous request ended.
        last = sent[-1]
        self.assertEqual(where[-1], len(last['messages']) - 3)     # in front of the newest call and result
        self.assertIn(where[-1] - 1, last['cache_anchors'])        # with the rung before it


class AfterACurate(SessionCase):
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
