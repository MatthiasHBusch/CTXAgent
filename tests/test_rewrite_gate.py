"""Cost gate for soft compaction cycles (config.rewrite_cost_gate).

Found by benchmarking (Context-Bench, 100 sequential questions, 1M window): every
'batch interval' cycle masked a few small results (~150-800 tokens freed) but changed
bytes deep inside the cached hot zone, so the provider re-wrote 30-75k tokens each
time -- cache writes were 3-5x those of Claude Code / Codex at no context pressure.
"""
import unittest

from ctxagent.compaction import CompactionEngine
from ctxagent.config import CompactionConfig
from ctxagent.segments import Conversation, K_SYSTEM, K_TOOL_RESULT, K_USER
from tests.test_compaction import fake_complete
from tests.test_context_controls import batch


def conv_with_batches(k=6, lines=400):
    c = Conversation()
    c.new('system', K_SYSTEM, 'sys', pinned=True)
    c.new('user', K_USER, 'go')
    for _ in range(k):
        batch(c, n=1, lines=lines)
    return c


def cfg(**kw):
    base = dict(rewrite_interval_batches=1, hot_zone_tokens=10_000_000, dedup_enabled=False,
                significance_enabled=False, deep_enabled=False, offload_enabled=False,
                protected_recent_observations=2, protected_recent_batches=1)
    base.update(kw)
    return CompactionConfig(**base)


class RewriteGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_soft_cycle_that_does_not_pay_is_undone_byte_exactly(self):
        c = conv_with_batches()
        before = c.to_messages()
        r = await CompactionEngine('g1', cfg(), fake_complete).run(c, 2_000_000)
        self.assertIn(r['schedule']['reason'], ('batch interval', 'large new context'))  # both soft
        gate = r['cost_gate']
        self.assertFalse(gate['keep'])
        self.assertGreater(gate['freed'], 0)
        self.assertGreater(gate['invalidated'], gate['freed'])
        self.assertEqual(r['freed'], 0)
        self.assertEqual(r['tiers'], [])
        self.assertEqual(c.to_messages(), before)            # cache-stable: nothing changed
        self.assertTrue(all(s.state == 'original' for s in c.segments if s.kind == K_TOOL_RESULT))
        # schedule still resets: no retry on every following batch
        self.assertEqual(c.compaction_state['batches'], 0)

    async def test_soft_cycle_that_pays_is_kept(self):
        c = conv_with_batches()
        before = c.to_messages()
        r = await CompactionEngine('g2', cfg(rewrite_cache_cost_ratio=1.5, rewrite_horizon_min=500,
                                               rewrite_horizon_max=500), fake_complete).run(c, 2_000_000)
        self.assertTrue(r['cost_gate']['keep'])
        self.assertGreater(r['freed'], 0)
        self.assertNotEqual(c.to_messages(), before)

    async def test_pressure_cycles_are_never_gated(self):
        c = conv_with_batches()
        window = int((c.total_tokens()) / 0.8)               # pressure ~0.8 >= rewrite_pressure_ratio
        r = await CompactionEngine('g3', cfg(rewrite_interval_batches=100), fake_complete).run(c, window)
        self.assertEqual(r['schedule']['reason'], 'pressure')
        self.assertNotIn('cost_gate', r)
        self.assertGreater(r['freed'], 0)

    async def test_gate_off_restores_legacy_behaviour(self):
        c = conv_with_batches()
        r = await CompactionEngine('g4', cfg(rewrite_cost_gate=False), fake_complete).run(c, 2_000_000)
        self.assertNotIn('cost_gate', r)
        self.assertGreater(r['freed'], 0)

    def test_config_validation(self):
        c = CompactionConfig.from_dict({'rewrite_cache_cost_ratio': 0.2, 'rewrite_horizon_min': 0,
                                        'rewrite_horizon_max': 10 ** 9})
        self.assertEqual(c.rewrite_cache_cost_ratio, 1.0)
        self.assertEqual(c.rewrite_horizon_min, 1)
        self.assertEqual(c.rewrite_horizon_max, 10000)
        with self.assertRaises(ValueError):
            CompactionConfig.from_dict({'rewrite_cost_gate': 'yes'})


if __name__ == '__main__':
    unittest.main()
