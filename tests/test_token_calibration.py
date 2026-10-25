"""Pressure calibration against provider-reported prompt tokens (AgentSession.token_calibration).

Found by benchmarking: on a 200k window Opus 5.5 reached 172k real prompt tokens while
the local estimate said 100k (pressure 0.60), so no compaction ever triggered.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ctxagent import agent, config, store
from ctxagent.config import SessionConfig
from ctxagent.segments import Conversation


def req(est, act, source='agent'):
    return {'source': source, 'usage': {'prompt_tokens': act},
            'cache_diagnostic': {'prompt_tokens_estimate': est}}


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [patch.object(config, 'DATA_DIR', root), patch.object(config, 'BLOB_DIR', root / 'blobs'),
                        patch.object(store, 'DB_PATH', root / 'chats.db')]
        for p in self.patches:
            p.start()
        store.init()
        cfg = SessionConfig(workdirs=[str(root)], context_window=200_000, max_output_tokens=20_000)
        self.s = agent.AgentSession(store.create_chat('cal', cfg.to_dict()), cfg, Conversation())

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_uncalibrated_until_measured(self):
        self.assertEqual(self.s.token_calibration(), 1.0)
        self.assertEqual(self.s.pressure_window(), self.s.cfg.effective_window())

    def test_median_of_recent_main_requests(self):
        self.s.usage['recent_requests'] = [req(10_000, 30_000)] + [req(10_000, 17_000)] * 3 + [
            req(10_000, 18_000), req(10_000, 99_000, source='summary'), req(500, 5_000)]
        self.assertAlmostEqual(self.s.token_calibration(), 1.7)  # last five agent rows >= 1000 est
        self.assertEqual(self.s.pressure_window(), int(self.s.cfg.effective_window() / 1.7))

    def test_clamped_and_switchable(self):
        self.s.usage['recent_requests'] = [req(10_000, 100_000)]
        self.assertEqual(self.s.token_calibration(), 3.0)
        self.s.usage['recent_requests'] = [req(10_000, 1_000)]
        self.assertEqual(self.s.token_calibration(), 0.8)
        self.s.cfg.compaction.calibrate_tokens = False
        self.assertEqual(self.s.token_calibration(), 1.0)

    def test_stats_report_calibrated_pressure(self):
        self.s.usage['recent_requests'] = [req(10_000, 20_000)]
        st = self.s.stats()
        self.assertEqual(st['token_calibration'], 2.0)


if __name__ == '__main__':
    unittest.main()
