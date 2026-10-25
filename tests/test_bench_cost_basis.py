"""normalize_cost_basis: flex-served requests are re-priced to list, including OpenAI's
long-context tier above 272k prompt tokens; list-billed and anomalous requests are handled."""
import unittest
from unittest import mock

from bench import analysis


def _row(wire, model="openai/gpt-6-astra", mode="prefer_flex"):
    return {"model": model, "pricing_mode": mode, "billed_cost": sum(w["billed"] for w in wire),
            "wire": wire, "turns": []}


def _w(turn, cache_read, output, billed, uncached=0, cache_write=0):
    return {"turn": turn, "uncached": uncached, "cache_write": cache_write, "cache_read": cache_read,
            "output": output, "billed": billed}


class NormalizeCostBasis(unittest.TestCase):
    # flat list prices: in 10, cache_write 12.5, cache_read 1, out 50 ($/M)
    def _cost(self, model, unc, cw, cr, out):
        return unc * 10e-6 + cw * 12.5e-6 + cr * 1e-6 + out * 50e-6

    def setUp(self):
        self.p = mock.patch("bench.pricing.cost", side_effect=self._cost)
        self.p.start()
        self.addCleanup(self.p.stop)

    def test_flex_request_doubles(self):
        flat = self._cost(None, 0, 0, 100_000, 1000)
        r = analysis.normalize_cost_basis(_row([_w(1, 100_000, 1000, flat / 2)]))
        self.assertAlmostEqual(r["billed_cost"], flat)
        self.assertEqual((r["n_flex_requests"], r["n_flex_longctx_requests"], r["n_price_anomalies"]), (1, 0, 0))
        self.assertEqual(r["cost_basis"], "list")

    def test_openai_long_context_on_flex_doubles(self):
        # >= 272k prompt: long-context tier (2x prompt, 1.5x output) at the 50 % flex discount
        # bills at ~flat-table level; the list price is still 2 x billed.
        cr, out = 300_000, 1000
        billed = (cr * 1e-6 * 2 + out * 50e-6 * 1.5) / 2
        r = analysis.normalize_cost_basis(_row([_w(5, cr, out, billed)]))
        self.assertAlmostEqual(r["billed_cost"], 2 * billed)
        self.assertEqual((r["n_flex_requests"], r["n_flex_longctx_requests"]), (1, 1))

    def test_list_billed_request_below_threshold_is_kept(self):
        # a non-OpenAI long-prompt surcharge (ratio >= 0.9, e.g. Haiku's 5x tier) is kept as billed
        flat = self._cost(None, 0, 0, 150_000, 100)
        r = analysis.normalize_cost_basis(_row([_w(2, 150_000, 100, flat * 5)], model="anthropic/claude-haiku-5.5"))
        self.assertAlmostEqual(r["billed_cost"], flat * 5)
        self.assertEqual((r["n_flex_requests"], r["n_flex_longctx_requests"], r["n_price_anomalies"]), (0, 0, 0))

    def test_long_prompt_at_list_for_non_openai_is_not_doubled(self):
        flat = self._cost(None, 0, 0, 300_000, 100)
        r = analysis.normalize_cost_basis(_row([_w(2, 300_000, 100, flat)], model="anthropic/claude-sonnet-5.5"))
        self.assertAlmostEqual(r["billed_cost"], flat)
        self.assertEqual(r["n_flex_longctx_requests"], 0)

    def test_anomaly_falls_back_to_flat(self):
        flat = self._cost(None, 0, 0, 10_000, 100)
        r = analysis.normalize_cost_basis(_row([_w(3, 10_000, 100, flat * 0.75)]))
        self.assertAlmostEqual(r["billed_cost"], flat)
        self.assertEqual(r["n_price_anomalies"], 1)

    def test_non_flex_session_untouched(self):
        r = analysis.normalize_cost_basis(_row([_w(1, 1000, 10, 0.5)], mode="standard"))
        self.assertNotIn("cost_basis", r)
        self.assertEqual(r["billed_cost"], 0.5)


if __name__ == "__main__":
    unittest.main()
