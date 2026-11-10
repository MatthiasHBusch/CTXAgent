"""Self-steered context management (the prompt section, compact_context / curate_context and
their cost figures) and the prefix-sharing summariser.

Offline: no model calls. The provider is a scripted httpx transport where a request must go
over the wire (prefix sharing); everything else runs the compaction code directly.
GUIDANCE_EXPERIMENTS.md records the experiments behind the defaults.
"""
import copy
import json
import time
import unittest
from unittest.mock import patch

import httpx

from ctxagent import accounting, agent as agentmod, llm, tokens
from ctxagent.compaction import curate, economics, manual, summarize
from ctxagent.config import GUIDANCE, CompactionConfig, SessionConfig
from ctxagent.segments import (Conversation, K_ASSISTANT, K_SUMMARY, K_SYSTEM, K_TOOL_CALL,
                               K_TOOL_RESULT, K_USER)

FABLE = {'read': 0.25e-6, 'write': 12.5e-6, 'input': 10e-6, 'output': 50e-6, 'known': True}   # 1:50
CHEAP_WRITE = {'read': 1.0, 'write': 1.05, 'input': 1.0, 'output': 1.0, 'known': True}        # ~1:1


def batch(conv, n=1, lines=300, tag='x'):
    ids = [f'{tag}{len(conv)}-{i}' for i in range(n)]
    conv.new('assistant', K_TOOL_CALL, '', tool_calls=[
        {'id': cid, 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path": "f.py"}'}}
        for cid in ids])
    return [conv.new('tool', K_TOOL_RESULT, '\n'.join(f'{j}\tline {j} of {cid}' for j in range(lines)),
                     tool_name='read_file', tool_call_id=cid) for cid in ids]


def agent_cfg(**kw):
    base = dict(rewrite_policy='agent', rewrite_interval_batches=3, deep_enabled=False)
    base.update(kw)
    return CompactionConfig(**base)


def history(n_results=6, lines=300):
    c = Conversation()
    c.new('system', K_SYSTEM, 'sys', pinned=True)
    c.new('user', K_USER, 'go')
    res = [batch(c, 1, lines)[0] for _ in range(n_results)]
    batch(c, 1, 20)               # the running batch
    c.compaction_state.update(batch_clock=40)
    return c, res


class EconomicsTests(unittest.TestCase):
    def test_verdict_break_even_scales_with_price_ratio(self):
        v = economics.verdict(10_000, 60_000, FABLE, horizon_requests=100)
        self.assertAlmostEqual(v['breakeven_requests'], 60_000 * (12.5 - 0.25) / (10_000 * 0.25), places=0)
        self.assertFalse(v['worth_it'])                      # 294 requests > 100 expected
        self.assertTrue(economics.verdict(10_000, 60_000, CHEAP_WRITE, 100)['worth_it'])
        self.assertIsNone(economics.verdict(0, 5, FABLE, 10)['breakeven_requests'])

    def test_rates_from_usage_and_fallback(self):
        usage = {'recent_requests': [{'source': 'agent', 'pricing': {'rates': {
            'prompt': 4e-6, 'input_cache_read': 0.2e-6, 'input_cache_write': 5e-6, 'completion': 20e-6}}}]}
        r = economics.rates(usage)
        self.assertTrue(r['known'])
        self.assertAlmostEqual(r['write'] / r['read'], 25.0)
        # no write tariff (a miss is billed as input), and unknown prices
        usage['recent_requests'][0]['pricing']['rates'].pop('input_cache_write')
        self.assertEqual(economics.rates(usage)['write'], 4e-6)
        self.assertFalse(economics.rates(None, 20.0)['known'])
        self.assertEqual(economics.rates(None, 20.0)['write'] / economics.rates(None, 20.0)['read'], 20.0)

    def test_cache_family(self):
        self.assertEqual(economics.cache_family('anthropic/claude-fable-5.1'), 'breakpoints')
        self.assertEqual(economics.cache_family('openai/gpt-6.1-sol'), 'prefix')


class CostFigureTests(unittest.TestCase):
    SUMMARY = 'Handoff: ' + 'goal, decisions, files, results and open issues. ' * 8

    def build(self):
        c = Conversation()
        c.new('system', K_SYSTEM, 'sys', pinned=True)
        c.new('user', K_USER, 'task A')
        for _ in range(5):
            batch(c, 2)
        c.new('user', K_USER, 'task B')
        batch(c, 1)
        batch(c, 1)
        return c

    def test_compact_reports_the_break_even_with_and_without_cfg(self):
        rep = manual.compact(self.build(), 't', summary=self.SUMMARY, cfg=CompactionConfig(), price=FABLE)
        self.assertIsNotNone(rep['cache_breakeven_requests'])
        self.assertNotIn('verdict', rep)
        self.assertIsNotNone(manual.compact(self.build(), 't', summary=self.SUMMARY)['cache_breakeven_requests'])
        c = self.build()
        self.assertGreater(manual.compact(c, 't', summary=self.SUMMARY, cfg=CompactionConfig(),
                                          price=FABLE)['saved_tokens'], 0)


def steered(**kw):
    return SessionConfig(compaction=CompactionConfig(rewrite_policy='agent', **kw))


class GuidancePromptTests(unittest.TestCase):
    def test_default_is_base_system_and_live_is_the_frozen_text(self):
        default = agentmod.static_instructions(steered())
        self.assertEqual(default, agentmod.BASE_SYSTEM)
        live = agentmod.static_instructions(steered(guidance='live'))
        self.assertNotEqual(live, default)
        self.assertTrue(live.startswith(default.split(agentmod.MANAGE_MARKER)[0]))
        # A few lines at most (the curate paragraph explains the references since 2026-11-10).
        self.assertLess(abs(tokens.count(live) - tokens.count(default)), 150)
        self.assertIn('Facts the user stated', live)
        self.assertEqual(set(agentmod.GUIDANCE_SECTIONS) | {'default'}, set(GUIDANCE))
        # engine off: the context sections go, whatever the variant
        off = SessionConfig(compaction=CompactionConfig(enabled=False, rewrite_policy='agent', guidance='live'))
        self.assertNotIn('Managing your own context', agentmod.static_instructions(off))

    def test_config_validation_and_round_trip(self):
        cfg = CompactionConfig.from_dict({'guidance': 'live', 'summary_share_prefix': False})
        self.assertEqual(CompactionConfig.from_dict(cfg.to_dict()), cfg)
        # a variant removed since falls back to the standard text; a typo is refused
        self.assertEqual(CompactionConfig.from_dict({'guidance': 'economics_hint'}).guidance, 'default')
        for bad in ({'guidance': 'loud'}, {'rewrite_policy': 'often'}):
            with self.assertRaises(ValueError):
                CompactionConfig.from_dict(bad)
        d = CompactionConfig()
        self.assertEqual((d.guidance, d.rewrite_policy), ('default', 'auto'))


class SelfSteeredTests(unittest.TestCase):
    """One setting (rewrite_policy='agent', shown as "Self-steered context management") brings
    compact_context, curate_context and the "Managing your own context" prompt section."""

    def test_the_switch_brings_both_tools_and_the_prompt_section(self):
        from ctxagent.config import SELF_STEERED_TOOLS
        on, off = steered(), SessionConfig(compaction=CompactionConfig())
        self.assertTrue(on.compaction.self_steered)
        self.assertFalse(off.compaction.self_steered)
        for name in SELF_STEERED_TOOLS:
            self.assertIn(name, on.active_tools())
            self.assertNotIn(name, off.active_tools())
        self.assertIn('Managing your own context', agentmod.static_instructions(on))
        self.assertIn('names the item before each history breakpoint', ' '.join(agentmod.static_instructions(on).split()))
        text = agentmod.static_instructions(off)
        self.assertNotIn('Managing your own context', text)
        self.assertIn('Context compaction', text)              # recall still explained
        # the engine master switch wins: no engine, no self-steering
        dead = SessionConfig(compaction=CompactionConfig(enabled=False, rewrite_policy='agent'))
        self.assertFalse(dead.compaction.self_steered)
        self.assertFalse(set(SELF_STEERED_TOOLS) & set(dead.active_tools()))

    def test_tool_list_ignores_the_context_tools(self):
        from ctxagent import profiles, tools
        from ctxagent.config import SELF_STEERED_TOOLS
        self.assertFalse(set(SELF_STEERED_TOOLS) & {t['name'] for t in tools.catalogue()})
        self.assertFalse(set(SELF_STEERED_TOOLS) & set(tools.DEFAULT_TOOLS))
        # an older chat that still lists them keeps their position (stable cached prefix)
        old = SessionConfig.from_dict({'enabled_tools': ['read_file', 'compact_context', 'curate_context', 'pin'],
                                       'compaction': {'rewrite_policy': 'agent'}})
        self.assertEqual(old.active_tools(), ['read_file', 'compact_context', 'curate_context', 'pin'])
        old.compaction.rewrite_policy = 'auto'
        self.assertEqual(old.active_tools(), ['read_file', 'pin'])
        # ... and a profile comparison does not see them
        a = profiles.comparable({'enabled_tools': ['read_file', 'compact_context']})
        b = profiles.comparable({'enabled_tools': ['read_file']})
        self.assertEqual(profiles.differences(a, b), [])

    def test_a_call_without_self_steering_is_refused(self):
        s = agentmod.AgentSession('t-off', SessionConfig(compaction=CompactionConfig()), Conversation())
        self.assertNotIn('compact_context', s.tool_names())
        self.assertNotIn('compact_context', [t['function']['name'] for t in s.tool_schema()])
        self.assertIn('compact_context', agentmod.AgentSession('t-on', steered(), Conversation()).tool_names())

    def test_compact_context_restarts_the_review_counter(self):
        c, _ = history()
        c.new('user', K_USER, 'next task')
        batch(c, 1, 20)
        c.compaction_state.update(batches=7, new_tokens=9000,
                                  review_due={'since_batch': 33, 'reason': 'batch interval'})
        manual.compact(c, 't', summary='Handoff: ' + 'the first task is finished and verified. ' * 8)
        st = c.compaction_state
        self.assertEqual(st['batches'], 0)
        self.assertNotIn('new_tokens', st)
        self.assertNotIn('review_due', st)

    def test_the_hint_is_one_line_for_every_variant(self):
        self.assertEqual(curate.HINT, '[harness: context review due]')
        self.assertEqual(len(curate.HINT.splitlines()), 1)

    def test_benchmark_live_arm_keeps_its_frozen_text_and_tools(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        try:
            from bench import run
        except ImportError:
            self.skipTest('the benchmark (bench/) is not part of this checkout')
        a = SimpleNamespace(harness='ctxagent_live', model='anthropic/claude-opus-5.5', window=1_000_000, comp=[],
                            max_out=1, max_iters=1, temperature=0.0, reasoning='off', summary_model='x/ignored')
        with tempfile.TemporaryDirectory() as tmp:
            cfg = run.build_ctx_config(a, Path(tmp))
        self.assertEqual(cfg.compaction.guidance, 'live')
        self.assertTrue(cfg.compaction.self_steered)
        self.assertIn('judge what is still relevant', agentmod.static_instructions(cfg))
        self.assertTrue({'compact_context', 'curate_context'} <= set(cfg.active_tools()))


# ------------------------------------------------------------------ prefix sharing
def sse(text, usage):
    chunks = [{'id': f'gen-{time.time_ns()}', 'provider': 'Anthropic', 'model': 'anthropic/claude-sonnet-5.5',
               'choices': [{'delta': {'content': text}}]},
              {'usage': usage, 'choices': [{'delta': {}, 'finish_reason': 'stop'}]}]
    return ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'


USAGE = {'prompt_tokens': 9000, 'completion_tokens': 300, 'cost': 0.01,
         'prompt_tokens_details': {'cached_tokens': 8800, 'cache_write_tokens': 150}}


def chat_session(summary_model):
    from tests.test_productivity import make_session
    s = make_session()
    s.cfg.model = 'anthropic/claude-sonnet-5.5'
    s.cfg.compaction = CompactionConfig(summary_model=summary_model,
                                        deep_min_gain_tokens=10, protected_recent_user_turns=1)
    s.engine.cfg = s.cfg.compaction
    s._sync_system()
    for i in range(4):
        s.conv.new('user', K_USER, f'task {i}: implement part {i}')
        batch(s.conv, 2, 120, tag=f't{i}_')
        s.conv.new('assistant', K_ASSISTANT, f'part {i} done')
    return s


class PrefixShareTests(unittest.IsolatedAsyncioTestCase):
    async def run_turn_then_compact(self, summary_model, age=0.0):
        captured = []

        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, text=sse('## 1. User intent\nshared summary' if len(captured) > 1
                                                else 'final answer', USAGE))
        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        def new_client(*a, **k):                             # one fresh client per request
            return real_client(transport=transport)

        async def allowed(*a):
            return ['anthropic']

        async def price(*a, **k):
            return {'rates': {'prompt': 2e-6, 'completion': 10e-6, 'input_cache_read': 0.1e-6,
                              'input_cache_write': 2.5e-6}}
        fresh_calls = []

        async def fresh(**kw):
            fresh_calls.append(kw)
            return '## 1. User intent\nfresh summary'
        s = chat_session(summary_model)
        with patch.object(llm.httpx, 'AsyncClient', side_effect=new_client), \
                patch.object(llm, '_headers', return_value={}), \
                patch.object(llm.provmod, 'allowed_slugs', allowed), \
                patch.object(accounting, 'pricing', price), patch.object(accounting, 'append_ledger'), \
                patch('ctxagent.agent.llm.complete', fresh):
            [e async for e in s.run_turn('task 4: final question')]
            if age:
                s._last_request['sent_at'] -= age
            [e async for e in s.compact_now()]
        return s, captured, fresh_calls

    async def test_shared_request_is_the_cached_prefix_plus_instruction(self):
        s, captured, fresh = await self.run_turn_then_compact('anthropic/claude-sonnet-5.5')
        self.assertEqual(len(captured), 2, 'one chat request, one summary request')
        self.assertFalse(fresh)
        chat, summ = captured
        for key in ('model', 'tools', 'session_id', 'reasoning', 'tool_choice'):
            self.assertEqual(json.dumps(chat.get(key), sort_keys=True), json.dumps(summ.get(key), sort_keys=True), key)
        n = len(chat['messages'])
        self.assertEqual(len(summ['messages']), n + 1)
        frame_ids = s._last_request['seg_ids']
        summary_seg = next(x for x in s.conv.segments if x.kind == K_SUMMARY)
        kept = [x.id for x in s.conv.segments if x.id in frame_ids and x is not summary_seg]
        cut = min(frame_ids.index(i) for i in kept if frame_ids.index(i) > 0)
        # byte-identical up to the cut (breakpoints included), and the same content up to the end
        self.assertEqual(json.dumps(chat['messages'][:cut]), json.dumps(summ['messages'][:cut]))

        def strip(m):
            # The old tail block carried the request's moving cache marker; a string content is
            # the same single text block to the provider. The cached content is what must match.
            m = copy.deepcopy(m)
            if isinstance(m.get('content'), str):
                m['content'] = [{'type': 'text', 'text': m['content']}]
            for b in m['content'] if isinstance(m.get('content'), list) else []:
                b.pop('cache_control', None)
            return m
        self.assertEqual([strip(m) for m in chat['messages']], [strip(m) for m in summ['messages'][:n]])
        self.assertIn('harness compaction request', json.dumps(summ['messages'][-1]))
        self.assertEqual(summ['messages'][-1]['role'], 'user')
        # cached breakpoints identical: system and frozen end carry the same markers
        marks = lambda p: [i for i, m in enumerate(p['messages'][:n - 1])
                           if 'cache_control' in json.dumps(m)]
        self.assertEqual(marks(chat), marks(summ))
        self.assertIn('shared summary', summary_seg.body)
        self.assertEqual(s.usage['by_source']['summary']['requests'], 1)
        rec = [r for r in s.usage['recent_requests'] if r['source'] == 'summary'][-1]
        self.assertTrue(rec['cache_diagnostic']['prefix_share'])
        deep = next(t for e in s.compaction_log for t in e.get('tiers', []) if t.get('tier') == '3-deep')
        self.assertEqual(deep['prefix_shared_messages'], n)

    async def test_other_summary_model_keeps_the_fresh_prompt(self):
        s, captured, fresh = await self.run_turn_then_compact('openai/gpt-5.6-luna')
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(fresh), 1)
        self.assertEqual(fresh[0]['model'], 'openai/gpt-5.6-luna')
        self.assertIn('fresh summary', next(x for x in s.conv.segments if x.kind == K_SUMMARY).body)

    async def test_expired_cache_keeps_the_fresh_prompt(self):
        s, captured, fresh = await self.run_turn_then_compact('anthropic/claude-sonnet-5.5',
                                                              age=summarize.SHARE_MAX_AGE_S + 60)
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(fresh), 1)

class ReplaySmokeTests(unittest.TestCase):
    """The offline replay (experiments/guidance) on a synthetic three-request trajectory."""

    def setUp(self):
        try:
            import experiments.guidance  # noqa: F401
        except ImportError:
            self.skipTest('the guidance experiments (experiments/) are not part of this checkout')

    def test_append_only_reads_previous_prompt_and_skip_removes_request(self):
        from experiments.guidance import data, replay
        segs, t = [], 0.0

        def seg(kind, body='', **kw):
            nonlocal t
            t += 1
            d = {'id': f's{len(segs)}', 'seq': len(segs) + 1, 'role': {'user': 'user', 'tool_result': 'tool'}.get(
                kind, 'assistant'), 'kind': kind, 'body': body, 'created_at': t, **kw}
            segs.append(d)
            return d
        seg('user', 'go')
        seg('tool_call', '', tool_calls=[{'id': 'c1', 'type': 'function',
                                         'function': {'name': 'read_file', 'arguments': '{}'}}])
        seg('tool_result', 'x ' * 3000, tool_call_id='c1', tool_name='read_file')
        seg('tool_call', '', tool_calls=[{'id': 'c2', 'type': 'function',
                                         'function': {'name': 'curate_context', 'arguments': '{}'}}])
        seg('tool_result', 'listing', tool_call_id='c2', tool_name='curate_context')
        seg('assistant_text', 'done')
        reqs = [data.Request(i, 100.0 + i, 1, 'anthropic/claude-opus-5.5', 10, 0, 1000 * (i + 1), 50, None, True)
                for i in range(3)]
        sess = data.Session('t', {'model': 'anthropic/claude-opus-5.5', 'harness': 'ctxagent_live', 'task': 'x'},
                            None, 'chat', SessionConfig().to_dict(), ['read_file', 'curate_context'], reqs, segs,
                            {'id': 'sys', 'kind': 'system', 'role': 'system', 'body': 'system prompt'})
        rec = replay.replay(sess, replay.Policy('recorded'))
        self.assertEqual(len(rec.snaps), 3)
        self.assertEqual(rec.snaps[1].read_bp, rec.snaps[0].prompt)      # append-only: all cached
        self.assertEqual(rec.snaps[2].read_auto, rec.snaps[1].prompt)
        skip = replay.replay(sess, replay.Policy('skip', decide=lambda c: 'skip'))
        self.assertEqual((len(skip.snaps), skip.removed), (2, 1))

    def test_call_kinds_and_plan_cells_are_labelled(self):
        from experiments.guidance import data, replay
        self.assertEqual(replay.classify('curate_context', {'rest': 'keep'}), 'curate_list')
        self.assertEqual(replay.classify('curate_context', {'drop': [], 'rest': 'keep', 'commit': True}),
                         'curate_commit')
        self.assertEqual(replay.classify('curate_context', {'drop': [], 'rest': 'auto'}), 'curate_drop')
        self.assertEqual(replay.classify('curate_context', {'drop': ['s1'], 'dry_run': True}), 'curate_dry')
        self.assertEqual(replay.classify('compact_context', {'dry_run': True}), 'compact_dry')
        self.assertEqual(data.arm_of({'tag': 'anthropic_x__mixdep-34s0__ctxagent_live', 'harness': 'ctxagent_live'}), 'A')
        self.assertEqual(data.arm_of({'tag': 'p1a_anthropic_x__mixdep-34s0__ctxagent_live',
                                      'harness': 'ctxagent_live'}), 'P1A')
        self.assertEqual(data.arm_of({'tag': 'gc_x', 'harness': 'ctxagent_live'}), 'GC')


if __name__ == '__main__':
    unittest.main(verbosity=2)
