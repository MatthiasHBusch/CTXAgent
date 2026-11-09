"""Cadence checks: review hints (self-steered) come on the interval, on large new context
or under pressure; freezing and pressure checks run every batch; below the deep trigger
the engine never changes a body."""
import unittest

from ctxagent.compaction import CompactionEngine
from ctxagent.config import CompactionConfig
from ctxagent.segments import Conversation, K_SYSTEM, K_TOOL_RESULT, K_USER
from tests.test_compaction import fake_complete
from tests.test_context_controls import batch


def config(**kwargs):
    base = dict(rewrite_interval_batches=5, rewrite_max_new_tokens=1000000, deep_enabled=False,
                rewrite_policy='agent')
    base.update(kwargs)
    return CompactionConfig(**base)


def started():
    c = Conversation()
    c.new('system', K_SYSTEM, 'sys', pinned=True)
    c.new('user', K_USER, 'go')
    return c


class CadenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_hint_on_fifth_batch_but_freezes_each_batch(self):
        conv = started()
        engine = CompactionEngine('cadence', config(hot_zone_tokens=1000), fake_complete)
        for i in range(5):
            res = batch(conv, n=1, lines=150)
            report = await engine.run(conv, 1_000_000)
            self.assertEqual('review_hint' in report, i == 4)
            if i < 4:
                self.assertNotIn('schedule', report)
        self.assertGreater(conv.frozen_upto, 0)
        self.assertEqual(report['schedule']['reason'], 'agent review due (batch interval)')
        self.assertIn('[harness: context review due', res[0].body)

    async def test_hint_repeats_only_after_another_interval(self):
        conv = started()
        engine = CompactionEngine('repeat', config(rewrite_interval_batches=2, hot_zone_tokens=10_000_000),
                                  fake_complete)
        hinted = []
        for i in range(6):
            batch(conv, n=1, lines=20)
            hinted.append('review_hint' in await engine.run(conv, 1_000_000))
        self.assertEqual(hinted, [False, True, False, True, False, True])

    async def test_state_and_cooldown_survive_reload(self):
        c = started()
        cfg = config()
        e = CompactionEngine('reload', cfg, fake_complete)
        for _ in range(3):
            batch(c, n=1, lines=20)
            await e.run(c, 1_000_000)
        clone = Conversation.from_dict(c.to_dict())
        other = CompactionEngine('reload', cfg, fake_complete)
        batch(clone, n=1, lines=20)
        report = await other.run(clone, 1_000_000)
        self.assertNotIn('review_hint', report)
        self.assertEqual(clone.compaction_state['batches'], 4)
        self.assertEqual(other.turns_since_deep, e.turns_since_deep + 1)
        batch(clone, n=1, lines=20)
        report = await other.run(clone, 1_000_000)
        self.assertIn('batch interval', report['schedule']['reason'])

    async def test_pressure_hint(self):
        c = started()
        batch(c, n=1, lines=800)
        e = CompactionEngine('pressure', config(), fake_complete)
        report = await e.run(c, int(c.total_tokens() / 0.8))
        self.assertIn('(pressure)', report['schedule']['reason'])
        self.assertIn('review_hint', report)

    async def test_large_output_hint(self):
        c = started()
        batch(c, n=1, lines=800)
        e = CompactionEngine('large', config(rewrite_max_new_tokens=1000), fake_complete)
        report = await e.run(c, 1_000_000)
        self.assertIn('(large new context)', report['schedule']['reason'])

    async def test_final_response_does_not_increment_batch_count(self):
        c = started()
        e = CompactionEngine('final', config(), fake_complete)
        await e.run(c, 100000)
        before = c.compaction_state['turns_since_deep']
        await e.run(c, 100000, completed_batch=False)
        self.assertEqual(c.compaction_state['batches'], 1)
        self.assertEqual(c.compaction_state['turns_since_deep'], before)

    async def test_engine_mode_gives_no_hints(self):
        c = started()
        e = CompactionEngine('auto', config(rewrite_policy='auto', rewrite_interval_batches=1), fake_complete)
        for _ in range(3):
            batch(c, n=1, lines=50)
            self.assertNotIn('review_hint', await e.run(c, 1_000_000))

    async def test_no_body_changes_below_the_deep_trigger(self):
        c = started()
        for i in range(20):
            batch(c, n=1, lines=60)
            c.new('tool', K_TOOL_RESULT, str(i) + ' repeated material ' * 60)
        cfg = config(rewrite_policy='auto', hot_zone_tokens=100, deep_enabled=True)
        e = CompactionEngine('unchanged', cfg, fake_complete)
        before = c.to_messages()
        report = await e.run(c, 10_000_000)
        self.assertEqual(before, c.to_messages())
        self.assertFalse(report['deep'])

    async def test_deep_threshold_and_force(self):
        def conv():
            c = started()
            for _ in range(8):
                batch(c, n=1, lines=300)
            return c
        cfg = config(rewrite_policy='auto', deep_enabled=True, deep_trigger_ratio=.5,
                     deep_min_gain_tokens=100, deep_cooldown_turns=0)
        c = conv()
        report = await CompactionEngine('deep', cfg, fake_complete).run(c, int(c.total_tokens() * 1.5))
        self.assertTrue(report['deep'])
        self.assertEqual(c.compaction_state['last_reason'], 'pressure')
        self.assertEqual(c.compaction_state['batches'], 0)
        c = conv()
        report = await CompactionEngine('force', cfg, fake_complete).run(
            c, 100_000_000, force_deep=True, completed_batch=False)
        self.assertTrue(report['deep'])
        self.assertEqual(c.compaction_state['last_reason'], 'forced')

    def test_config_roundtrip_and_defaults(self):
        cfg = CompactionConfig.from_dict({})
        self.assertEqual(cfg.rewrite_interval_batches, 10)
        self.assertEqual(CompactionConfig.from_dict(cfg.to_dict()).rewrite_max_new_tokens, 20000)
        with self.assertRaises(ValueError):
            CompactionConfig.from_dict({'rewrite_pressure_ratio': float('nan')})

    def test_removed_settings_still_load(self):
        old = {'masking_enabled': True, 'dedup_enabled': True, 'significance_enabled': True,
               'mask_tiers': [2000, 1500], 'curate_gate': 'defer', 'tail_ratio_max': .15,
               'guidance': 'economics_hint', 'hot_zone_tokens': 40000}
        cfg = CompactionConfig.from_dict(old)
        self.assertEqual(cfg.hot_zone_tokens, 40000)
        self.assertEqual(cfg.guidance, 'default')
        self.assertNotIn('masking_enabled', cfg.to_dict())
        self.assertEqual(CompactionConfig.from_dict({'guidance': 'live'}).guidance, 'live')


if __name__ == '__main__':
    unittest.main(verbosity=2)
