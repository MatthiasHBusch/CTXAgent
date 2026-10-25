"""Offline inspection tests. No network, user DB or blob writes."""
import copy
import json
import unittest
from types import SimpleNamespace

from ctxagent.context_report import report, recent_usage
from ctxagent.config import SessionConfig
from ctxagent.segments import Conversation, K_TOOL_RESULT, K_USER
from ctxagent.compaction import CompactionEngine
from ctxagent import tools, tokens


class ContextReportTests(unittest.TestCase):
    def test_bounded_and_content_free(self):
        c = Conversation()
        c.new('user', K_USER, 'SECRET_USER_INTENT')
        for i in range(15):
            c.new('tool', K_TOOL_RESULT, 'SECRET_OUTPUT ' * (i + 1), tool_name='read_file')
        c.freeze_prefix(5)
        cfg = SessionConfig(enabled_tools=['read_file'])
        cfg.compaction.document_compaction_enabled = False
        before = copy.deepcopy(c.to_dict())
        result = report(c, cfg, top=500)
        text = json.dumps(result)
        self.assertNotIn('SECRET', text)
        self.assertNotIn('original_body', text)
        self.assertEqual(len(result['largest_segments']), 10)
        self.assertEqual(c.to_dict(), before)
        self.assertEqual(result['context']['frozen_segments'], 5)
        self.assertEqual(result['context']['frozen_original_tool_results'], 4)
        self.assertTrue(any('disabled; cadence' in s for s in result['warnings']))

    def test_no_telemetry_is_unknown_not_zero(self):
        r = recent_usage({})
        for key in ('reported_cost', 'weighted_cache_hit_rate', 'reported_input_cost',
                    'reported_input_fraction_matched'):
            self.assertIsNone(r[key])

    def test_weighted_agent_only_and_matched_itemization(self):
        rows = [
            {'source': 'agent', 'usage': {'prompt_tokens': 100, 'cache_read': 50, 'cost': 2},
             'raw_usage': {'cost_details': {'upstream_inference_prompt_cost': 1}}},
            {'source': 'agent', 'usage': {'prompt_tokens': 900, 'cache_read': 900, 'cost': 10}},
            {'source': 'summary', 'usage': {'prompt_tokens': 10000, 'cache_read': 0, 'cost': 100}},
        ]
        r = recent_usage({'recent_requests': rows})
        self.assertEqual(r['weighted_cache_hit_rate'], .95)
        self.assertEqual(r['reported_cost'], 12)
        self.assertEqual(r['reported_input_fraction_matched'], .5)
        self.assertEqual(r['input_itemized_requests'], 1)
        rows[0]['raw_usage']['cost_details']['upstream_inference_prompt_cost'] = None
        self.assertIsNone(recent_usage({'recent_requests': rows})['reported_input_cost'])

    def test_recent_bound_and_missing_cost(self):
        rows = [{'source': 'agent', 'usage': {'prompt_tokens': 10, 'cache_read': 0, 'cost': None}}] * 10
        r = recent_usage({'recent_requests': rows}, recent=2)
        self.assertEqual(r['agent_requests'], 2)
        self.assertEqual(r['missing_cost_requests'], 2)
        self.assertIsNone(r['reported_cost'])

    def test_full_output_headroom(self):
        self.assertEqual(SessionConfig(context_window=200000, max_output_tokens=32000).effective_window(), 155000)
        self.assertEqual(SessionConfig(context_window=200000, max_output_tokens=8000).effective_window(), 179000)
        self.assertEqual(SessionConfig(context_window=32000, max_output_tokens=32000).effective_window(), 1)

    def test_stats_do_not_claim_prospective_freeze(self):
        c = Conversation()
        for i in range(10):
            c.new('tool', K_TOOL_RESULT, 'words ' * 100)
        cfg = SessionConfig()
        cfg.compaction.hot_zone_tokens = 100
        engine = CompactionEngine('report', cfg.compaction, None)
        self.assertEqual(engine.stats(c, 100000)['frozen_tokens'], 0)
        c.freeze_prefix(3)
        self.assertEqual(engine.stats(c, 100000)['frozen_tokens'], sum(s.tokens() for s in c.segments[:3]))
        self.assertEqual(engine.stats(c, 100000)['hot_start'], 3)

    def test_summary_floor_when_fixed_context_exceeds_target(self):
        from ctxagent.compaction.deep_plan import plan
        from ctxagent.config import CompactionConfig
        c = Conversation()
        c.new('user', K_USER, 'old work ' * 1000)
        c.new('tool', K_TOOL_RESULT, 'unseen output ' * 10000)
        cfg = CompactionConfig(protected_recent_user_turns=0, summary_max_tokens=3000,
                               rewrite_cost_gate=False)  # replay of masking mechanics
        p = plan(c, cfg)
        self.assertGreater(p['fixed_tokens'], p['target_tokens'])
        self.assertEqual(p['summary_budget'], 1024)
        self.assertEqual(CompactionConfig.from_dict(cfg.to_dict()).summary_min_tokens, 1024)
        with self.assertRaises(ValueError):
            CompactionConfig.from_dict({'summary_min_tokens': 100})
        cfg.summary_max_tokens = 500
        self.assertEqual(plan(c, cfg)['summary_budget'], 500)

    def test_tool_schema_stable_and_small(self):
        self.assertIn('context_report', tools.DEFAULT_TOOLS)
        self.assertTrue(tools.REGISTRY['context_report'].needs_ctx)
        self.assertLess(tokens.count_tools(tools.openai_schema(['context_report'])), 200)


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_sweep_is_fixed_bounded_and_retains_source(self):
        from tests.efficiency_replay import sweep, batches_for
        from ctxagent.segments import K_SYSTEM, K_TOOL_CALL
        c = Conversation()
        c.new('system', K_SYSTEM, 'instructions')
        for i in range(10):
            c.new('assistant', K_TOOL_CALL, '', tool_calls=[
                {'id': str(i), 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}])
            c.new('tool', K_TOOL_RESULT, 'source output ' * 400,
                  tool_name='read_file', tool_call_id=str(i))
        source = c.to_dict()['segments']
        before = copy.deepcopy(source)
        cfg = SessionConfig(enabled_tools=[])
        cfg.compaction.offload_enabled = False
        cfg.compaction.protected_recent_observations = 2
        cfg.compaction.rewrite_cost_gate = False  # sweep mechanics need rewriting policies
        rows = (await sweep(source, cfg))['rows']
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]['requests'], 10)
        self.assertEqual(rows[0]['cost_units_read_010_write_125_vs_disabled'], 1)
        self.assertLess(min(r['final_context_estimate'] for r in rows), rows[0]['final_context_estimate'])
        self.assertEqual(source, before)
        sliced = batches_for(source, 2)
        self.assertEqual(len(sliced), 2)
        self.assertEqual(sliced[0][0]['kind'], K_SYSTEM)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_shell_command_is_executed_exactly_but_not_echoed_twice(self):
        from unittest.mock import patch
        from pathlib import Path
        import tempfile
        command = 'echo ' + 'LONG_SCRIPT ' * 1000
        with tempfile.TemporaryDirectory() as root:
            with patch('ctxagent.tools.subprocess.run', return_value=SimpleNamespace(
                    returncode=0, stdout='RESULT', stderr='')) as run:
                result = await tools.t_run_command(tools.Sandbox([root]), command, cwd=root)
        self.assertEqual(run.call_args.args[0][-1], command)
        self.assertIn('RESULT', result.text)
        self.assertIn('command preview shortened', result.text)
        self.assertNotIn(command, result.text)
        self.assertLess(len(result.text), 500)

    async def test_tool_uses_current_session(self):
        session = SimpleNamespace(conv=Conversation(), cfg=SessionConfig(enabled_tools=[]), usage={})
        result = await tools.t_context_report(tools.Sandbox([]), ctx={'session': session})
        self.assertEqual(json.loads(result.text)['context']['tool_schema_tokens_estimate'], 0)
        self.assertFalse(result.mutates)


if __name__ == '__main__':
    unittest.main(verbosity=2)
