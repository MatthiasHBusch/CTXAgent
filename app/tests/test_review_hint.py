"""The review hint names the cache breakpoints (2026-11-10).

A request carries at most four breakpoints: the system prompt, two rungs in the history and
the tail (cache_policy). Replacing an item re-writes everything behind the nearest
breakpoint before it, so the items between two breakpoints are best replaced in one call.
The hint says where the rungs of the next request lie:

* as references, counted the way a curate_context call in the next response counts (the
  batch that carries the hint included), with a few words of the item;
* only where the provider caches at breakpoints; otherwise the hint is the plain line.
"""
import itertools
import json
import unittest
from unittest.mock import patch

from ctxagent import cache_policy as cp, tokens
from ctxagent.compaction import curate
from ctxagent.compaction.engine import CompactionEngine
from ctxagent.config import CompactionConfig
from ctxagent.segments import Conversation, K_ASSISTANT, K_SYSTEM, K_TOOL_CALL, K_TOOL_RESULT, K_USER


def batch(c, cid, name, args, body, text=''):
    c.new('assistant', K_TOOL_CALL, text, tool_calls=[
        {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}])
    return c.new('tool', K_TOOL_RESULT, body, tool_call_id=cid, tool_name=name)


def chat():
    """system, user, 3 batches, an answer, user, 2 batches. Returns the chat and its results."""
    c = Conversation()
    c.new('system', K_SYSTEM, 'system prompt')
    c.new('user', K_USER, 'build the thing')
    res = [batch(c, 'c1', 'run_command', {'command': 'make'}, 'BUILD LOG\n' + 'line\n' * 50),
           batch(c, 'c2', 'read_file', {'path': 'a.py'}, 'print(1)\n' * 50),
           batch(c, 'c3', 'grep_search', {'pattern': 'foo'}, 'a.py:1 foo')]
    c.new('assistant', K_ASSISTANT, 'Built it and read a.py.')
    c.new('user', K_USER, 'now the tests')
    res += [batch(c, 'c4', 'run_command', {'command': 'pytest'}, '12 passed'),
            batch(c, 'c5', 'read_file', {'path': 'b.py'}, 'x = 1')]
    return c, res


def sent(c, rungs, upto, family='anthropic'):
    """The cache state a request left that sent the first `upto` history messages."""
    hashes = [cp.digest(m) for m in c.to_messages()][:upto]
    c.cache_state = {'cache_family': family, 'history_hashes': hashes, 'rungs': rungs,
                     'history_tail': upto - 1, 'views_hash': cp.digest([]), 'views_count': 0, 'views_at': 0}


def fixed_rungs():
    """The next request keeps the rungs of the last one: these tests are about naming them."""
    return patch.object(cp, 'plan_rungs', lambda prior, *a, **kw: {'rungs': list(prior['rungs'])})


def message_of(c, seg):
    sources = []
    c.to_messages(sources)
    return next(k for k, (i, _) in enumerate(sources) if c.segments[i] is seg)


class Breakpoints(unittest.TestCase):
    def test_reference_and_words_of_the_item_at_each_rung(self):
        c, res = chat()
        n = len(c.to_messages())
        sent(c, [message_of(c, res[0]), message_of(c, res[2])], n - 2)      # before the newest batch
        with fixed_rungs():
            found = curate.breakpoints(c)
            self.assertEqual([(ref, item.seg) for ref, item in found], [(6, res[2]), (10, res[0])])
            self.assertEqual(curate.hint(c), '[harness: context review due; cache breakpoints '
                                             'after item 6 = grep_search result (a.py:1 foo) and '
                                             'after item 10 = run_command result (BUILD LOG)]')
        # A curate call in the next response counts the same way.
        batch(c, 'c9', 'curate_context', {}, '')
        del c.segments[-1]
        items = curate.items(c)
        self.assertIs(items[-6].seg, res[2])
        self.assertIs(items[-10].seg, res[0])

    def test_a_rung_on_a_message_that_is_no_item_names_the_item_before_it(self):
        c, res = chat()
        n = len(c.to_messages())
        user = next(s for s in c.segments if s.body == 'now the tests')
        sent(c, [message_of(c, user)], n - 2)
        with fixed_rungs():
            (ref, item), = curate.breakpoints(c)
        self.assertEqual((ref, item.part, item.seg.body), (5, 'answer', 'Built it and read a.py.'))

    def test_without_a_rung_the_hint_says_so(self):
        c, res = chat()
        sent(c, [], len(c.to_messages()) - 2)
        with fixed_rungs():
            self.assertEqual(curate.breakpoints(c), [])
            self.assertEqual(curate.hint(c), '[harness: context review due; no cache breakpoint in the history yet]')

    def test_the_rungs_the_next_request_places_are_named(self):
        c = Conversation()
        c.new('system', K_SYSTEM, 'system prompt')
        c.new('user', K_USER, 'build the thing')
        for k in range(13):                                  # 130k tokens
            batch(c, f'c{k}', 'run_command', {'command': f'step {k}'}, f'RESULT {k}\n' + 'word ' * 10_000)
        sent(c, [], len(c.to_messages()))                    # the last request had no rung yet
        batch(c, 'c99', 'read_file', {'path': 'c.py'}, 'y = 2')
        ends = list(itertools.accumulate(tokens.count_message(m) for m in c.to_messages()))
        (near_ref, near), (far_ref, far) = curate.breakpoints(c)
        self.assertLess(near_ref, far_ref)                   # newest first
        halfway = (ends[-1] - cp.NEAR_RUNG_TOKENS) / 2
        self.assertTrue(halfway - 11_000 <= ends[message_of(c, far.seg)] <= halfway)
        self.assertTrue(cp.NEAR_RUNG_TOKENS <= ends[-1] - ends[message_of(c, near.seg)]
                        <= cp.NEAR_RUNG_TOKENS + 11_000)     # the last message that ends 50k before the end

    def test_plain_hint_without_breakpoints(self):
        c, res = chat()
        self.assertIsNone(curate.breakpoints(c))             # no request sent yet
        self.assertEqual(curate.hint(c), curate.HINT)
        sent(c, [message_of(c, res[0])], len(c.to_messages()) - 2, family='provider-managed')
        self.assertEqual(curate.hint(c), curate.HINT)        # automatic prefix caching


class EngineHint(unittest.IsolatedAsyncioTestCase):
    async def test_the_engine_appends_it_to_the_newest_result(self):
        c, res = chat()
        sent(c, [message_of(c, res[1])], len(c.to_messages()) - 2)
        cfg = CompactionConfig(rewrite_policy='agent', deep_enabled=False, rewrite_interval_batches=2)
        c.compaction_state['batches'] = 5
        with fixed_rungs():
            rep = await CompactionEngine('h', cfg, None).run(c, 1_000_000)
        want = '[harness: context review due; cache breakpoints after item 8 = read_file result (print(1))]'
        self.assertEqual(rep['review_hint'], want)
        self.assertEqual(res[-1].body, 'x = 1\n\n' + want)


if __name__ == '__main__':
    unittest.main(verbosity=2)
