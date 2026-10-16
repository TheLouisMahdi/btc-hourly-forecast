"""Offline causal and accounting tests for the independent regime shadow."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from regime_shadow import (
    HARD_STOP, HOUR, completed_four_hour_closes, features, new_state, step,
    wilder_rsi,
)


def rising(count=140):
    bars = []
    for i in range(count):
        close = 100 + i * .5
        bars.append({
            "time": i * HOUR, "open": close - .1, "high": close + .1,
            "low": close - .2, "close": close, "volume": 100,
        })
    return bars


def breakout(count=140):
    bars = rising(count)
    bars[-1]["close"] += 3
    bars[-1]["high"] = bars[-1]["close"] + .1
    return bars


class RegimeShadowTests(unittest.TestCase):
    def test_rsi_has_no_future_data_and_handles_flat_market(self):
        self.assertEqual(wilder_rsi([100.0] * 25), 50)
        self.assertEqual(wilder_rsi(list(range(25))), 100)
        bars = rising(140)
        completed = completed_four_hour_closes(bars)
        self.assertEqual(len(completed), 35)
        bars.append({"time": 140 * HOUR, "open": 170, "high": 170.1,
                     "low": 1, "close": 1, "volume": 100})
        self.assertEqual(completed_four_hour_closes(bars), completed)
        self.assertEqual(features(bars)["completed_4h_end"],
                         features(bars[:-1])["completed_4h_end"])

    def test_buy_uses_closed_hour_and_has_strict_cash_risk_cap(self):
        state, bars = new_state(), breakout()
        result = step(state, bars, bars[-1]["close"], len(bars) * HOUR)
        self.assertEqual(result["action"], "BUY")
        self.assertEqual(result["reason"], "BREAKOUT")
        self.assertGreater(result["features"]["rsi4h"], 70)
        self.assertLessEqual(state["qty"] * state["position"]["entry_price"] * HARD_STOP, 50.001)
        self.assertLess(state["cash"], 10000)
        self.assertGreaterEqual(state["cash"], 0)
        self.assertEqual(state["zone_entries"], 1)
        self.assertEqual(result["mode"], "PAPER_ONLY")

    def test_neutral_regime_stays_flat(self):
        bars = [{"time": i * HOUR, "open": 100, "high": 100.1,
                 "low": 99.9, "close": 100, "volume": 100} for i in range(140)]
        state = new_state()
        result = step(state, bars, 100, 140 * HOUR)
        self.assertEqual(result["reason"], "RSI_REGIME_OFF")
        self.assertEqual(state["qty"], 0)

    def test_duplicate_candle_cannot_buy_twice(self):
        state, bars = new_state(), breakout()
        step(state, bars, bars[-1]["close"], len(bars) * HOUR)
        before = copy.deepcopy(state)
        result = step(state, bars, bars[-1]["close"], len(bars) * HOUR + 900)
        self.assertEqual(result["action"], "UNCHANGED")
        self.assertEqual(state, before)

    def test_hard_stop_quote_closes_same_hour_with_costs(self):
        state, bars = new_state(), breakout()
        step(state, bars, bars[-1]["close"], len(bars) * HOUR)
        price = state["position"]["hard_stop"] - 1
        result = step(state, bars, price, len(bars) * HOUR + 900)
        self.assertEqual(result["reason"], "HARD_STOP_QUOTE")
        self.assertEqual(len(state["trades"]), 1)
        self.assertLess(state["trades"][0]["pnl_usd"], 0)
        self.assertEqual(state["qty"], 0)
        self.assertAlmostEqual(state["cash"], 10000 + state["trades"][0]["pnl_usd"])

    def test_soft_close_stop_waits_for_next_finished_hour(self):
        state, bars = new_state(), breakout()
        step(state, bars, bars[-1]["close"], len(bars) * HOUR)
        price = state["position"]["entry_price"] * .985
        next_bar = {"time": len(bars) * HOUR, "open": price + .1,
                    "high": price + .2, "low": price - .1,
                    "close": price, "volume": 100}
        self.assertGreater(next_bar["low"], state["position"]["hard_stop"])
        result = step(state, bars + [next_bar], price, (len(bars) + 1) * HOUR)
        self.assertEqual(result["reason"], "SOFT_CLOSE_STOP")
        self.assertEqual(len(state["trades"]), 1)

    def test_skipped_hour_exits_instead_of_replaying_entries(self):
        state, bars = new_state(), breakout()
        step(state, bars, bars[-1]["close"], len(bars) * HOUR)
        future = rising(145)
        result = step(state, future, future[-1]["close"], 145 * HOUR)
        self.assertEqual(result["reason"], "MISSED_CANDLE_EXIT")
        self.assertEqual(state["qty"], 0)
        self.assertEqual(len(state["trades"]), 1)

    def test_data_gap_rejected_before_account_mutation(self):
        state, bars = new_state(), rising()
        bad = bars[:]
        bad[50] = {**bad[50], "time": bad[50]["time"] + 50}
        before = copy.deepcopy(state)
        with self.assertRaisesRegex(ValueError, "Discontinuous"):
            step(state, bad, 170, len(bars) * HOUR)
        self.assertEqual(state, before)

    def test_max_entries_gate_prevents_reentry(self):
        bars = breakout()
        state = new_state()
        step(state, bars, bars[-1]["close"], 140 * HOUR)
        state["qty"] = 0.0
        state["position"] = None
        state["zone_entries"] = 3
        next_bar = {"time": 140 * HOUR, "open": 173, "high": 177.2,
                    "low": 172.9, "close": 177, "volume": 100}
        result = step(state, bars + [next_bar], 177, 141 * HOUR)
        self.assertEqual(result["reason"], "ZONE_ENTRY_LIMIT")
        self.assertEqual(state["qty"], 0)

    def test_time_exit_without_unbounded_holding(self):
        bars = breakout()
        state = new_state()
        step(state, bars, bars[-1]["close"], len(bars) * HOUR)
        last = bars[-1]["close"]
        for i in range(140, 165):
            close = last + (i - 139) * .3
            bars.append({"time": i * HOUR, "open": close - .1,
                         "high": close + .1, "low": close - .2,
                         "close": close, "volume": 100})
            result = step(state, bars, close, (i + 1) * HOUR)
            if result["action"] == "EXIT":
                break
        self.assertEqual(result["reason"], "TIME_EXIT")
        self.assertEqual(state["qty"], 0)


if __name__ == "__main__":
    unittest.main()
