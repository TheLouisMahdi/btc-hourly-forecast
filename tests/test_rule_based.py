"""Offline tests for the rule-based paper engine."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rule_based import HOUR, buy, entry_signal, indicators, new_state, sell, step


def candles(count=300):
    return [{"time": i * HOUR, "open": 100 + i * .05,
             "high": 101 + i * .05, "low": 99 + i * .05,
             "close": 100 + i * .05, "volume": 100}
            for i in range(count)]


class RuleEngineTests(unittest.TestCase):
    def test_uses_prior_candles_for_channel(self):
        data = candles()
        features = indicators(data, 250)
        self.assertEqual(features["entry_high55"], max(b["high"] for b in data[195:250]))
        self.assertEqual(features["exit_low20"], min(b["low"] for b in data[230:250]))

    def test_does_not_enter_without_breakout(self):
        state = new_state()
        result = step(state, candles(), 115, 300 * HOUR)
        self.assertEqual(result["action"], "FLAT")
        self.assertEqual(state["qty"], 0)

    def test_transaction_costs_are_charged_both_sides(self):
        state = new_state()
        self.assertTrue(buy(state, 100, 1, 300 * HOUR, 301 * HOUR))
        self.assertLess(state["cash"], 10_000)
        sold = sell(state, 100, "TEST", 302 * HOUR, 301 * HOUR)
        self.assertLess(sold["pnl_usd"], 0)
        self.assertEqual(state["qty"], 0)
        self.assertAlmostEqual(state["cash"], 10_000 + sold["pnl_usd"], places=7)

    def test_idempotent_candle(self):
        state, data = new_state(), candles()
        step(state, data, 115, 300 * HOUR)
        result = step(state, data, 115, 300 * HOUR)
        self.assertEqual(result["action"], "UNCHANGED")
        self.assertEqual(len(state["history"]), 1)

    def test_skipped_candles_stop_new_orders(self):
        state = new_state()
        state["last_candle"] = 295 * HOUR
        result = step(state, candles(), 115, 300 * HOUR)
        self.assertEqual(result["action"], "FAIL_SAFE")
        self.assertEqual(state["last_candle"], 295 * HOUR)

    def test_open_position_exits_on_current_quote_below_stop(self):
        state = new_state()
        data = candles()
        self.assertTrue(buy(state, 110, 1, 298 * HOUR, 299 * HOUR))
        stop = state["stop"]
        result = step(state, data, stop - 1, 300 * HOUR)
        self.assertEqual(result["action"], "EXIT")
        self.assertEqual(result["reason"], "STOP_QUOTE")
        self.assertEqual(state["qty"], 0)
        self.assertEqual(len(state["trades"]), 1)

    def test_entry_signal_is_explicit(self):
        features = {"entry_high55": 100, "ema200": 90, "atr14": 2}
        self.assertTrue(entry_signal({"close": 105}, features))
        self.assertFalse(entry_signal({"close": 99}, features))


if __name__ == "__main__":
    unittest.main()
