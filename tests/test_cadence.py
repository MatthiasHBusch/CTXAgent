"""Cadence checks: defer rewrites, never defer persistence/freeze or pressure checks."""
import unittest
from unittest.mock import patch

from ctxagent.compaction import CompactionEngine
from ctxagent.config import CompactionConfig
from ctxagent.segments import Conversation, K_TOOL_RESULT
from tests.test_compaction import fake_complete


def config(**kwargs):
    return CompactionConfig(rewrite_interval_batches=5, rewrite_max_new_tokens=1000000,
                            significance_enabled=False, deep_enabled=False, **kwargs)


class CadenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_rewrites_on_fifth_batch_but_freezes_each_batch(self):
        conv = Conversation()
        cfg = config(hot_zone_tokens=1000)
        engine = CompactionEngine('cadence', cfg, fake_complete)
        with patch('ctxagent.compaction.engine.masking.run', return_value={'freed': 0}) as masking:
            for i in range(5):
                conv.new('tool', K_TOOL_RESULT, 'observation ' * 700)
                report = await engine.run(conv, 100000)
                self.assertEqual(masking.call_count, int(i == 4))
                if i < 4:
                    self.assertEqual(report['schedule']['reason'], 'deferred')
            self.assertGreater(conv.frozen_upto, 0)
            self.assertEqual(conv.compaction_state['batches'], 0)
            self.assertEqual(report['schedule']['reason'], 'batch interval')

    async def test_state_and_cooldown_survive_reload(self):
        c = Conversation()
        cfg = config()
        e = CompactionEngine('reload', cfg, fake_complete)
        for i in range(3):
            await e.run(c, 100000)
        clone = Conversation.from_dict(c.to_dict())
        other = CompactionEngine('reload', cfg, fake_complete)
        report = await other.run(clone, 100000)
        self.assertEqual(report['schedule']['reason'], 'deferred')
        self.assertEqual(clone.compaction_state['batches'], 4)
        self.assertEqual(other.turns_since_deep, e.turns_since_deep + 1)
        report = await other.run(clone, 100000)
        self.assertEqual(report['schedule']['reason'], 'batch interval')

    async def test_pressure_override(self):
        c = Conversation()
        c.new('tool', K_TOOL_RESULT, 'words ' * 2000)
        e = CompactionEngine('pressure', config(), fake_complete)
        report = await e.run(c, 2200)
        self.assertEqual(report['schedule']['reason'], 'pressure')
        self.assertEqual(c.compaction_state['batches'], 0)

    async def test_large_output_override(self):
        c = Conversation()
        c.new('tool', K_TOOL_RESULT, 'words ' * 3000)
        cfg = config()
        cfg.rewrite_max_new_tokens = 1000
        e = CompactionEngine('large', cfg, fake_complete)
        report = await e.run(c, 100000)
        self.assertEqual(report['schedule']['reason'], 'large new context')

    async def test_final_response_does_not_increment_batch_count(self):
        c = Conversation()
        e = CompactionEngine('final', config(), fake_complete)
        await e.run(c, 100000)
        before = c.compaction_state['turns_since_deep']
        await e.run(c, 100000, completed_batch=False)
        self.assertEqual(c.compaction_state['batches'], 1)
        self.assertEqual(c.compaction_state['turns_since_deep'], before)

    async def test_force_overrides_cadence(self):
        c = Conversation()
        e = CompactionEngine('force', config(), fake_complete)
        report = await e.run(c, 100000, force_deep=True, completed_batch=False)
        self.assertEqual(report['schedule']['reason'], 'forced')

    async def test_no_rewrite_or_body_changes_on_deferred_pass(self):
        c = Conversation()
        for i in range(20):
            c.new('tool', K_TOOL_RESULT, str(i) + ' repeated material ' * 60)
        cfg = config(hot_zone_tokens=100000)
        e = CompactionEngine('unchanged', cfg, fake_complete)
        before = c.to_messages()
        await e.run(c, 1000000)
        self.assertEqual(before, c.to_messages())

    async def test_deep_threshold_overrides_schedule(self):
        c = Conversation()
        c.new('tool', K_TOOL_RESULT, 'words ' * 300)
        cfg = config()
        cfg.deep_enabled = True
        cfg.deep_trigger_ratio = .1
        e = CompactionEngine('deep', cfg, fake_complete)
        report = await e.run(c, 1000)
        self.assertEqual(report['schedule']['reason'], 'pressure')

    def test_config_roundtrip_and_defaults(self):
        cfg = CompactionConfig.from_dict({})
        self.assertEqual(cfg.rewrite_interval_batches, 5)
        self.assertEqual(CompactionConfig.from_dict(cfg.to_dict()).rewrite_max_new_tokens, 20000)
        with self.assertRaises(ValueError):
            CompactionConfig.from_dict({'rewrite_pressure_ratio': float('nan')})


if __name__ == '__main__':
    unittest.main(verbosity=2)
