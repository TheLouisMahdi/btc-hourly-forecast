"""Offline parity, gap and cost tests for the fair 4-strategy comparison."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_spot_strategies as comp


def fixtures():
    start_day = comp.epoch(comp.START_DAY)
    start_hour = comp.epoch(comp.START_HOUR)
    end = comp.epoch(comp.END)
    dbars = []
    for t in range(start_day, end, comp.DAY):
        i = (t - start_day) // comp.DAY
        price = 100 + i * 0.07
        dbars.append({"time": t, "open": price - 0.03,
                      "high": price + 1, "low": price - 1,
                      "close": price, "volume": 20.})
    hbars = []
    for t in range(start_hour, end, comp.HOUR):
        i = (t - start_hour) // comp.HOUR
        price = 130 + i * .005
        hbars.append({"time": t, "open": price - 0.003,
                      "high": price + .5, "low": price - .5,
                      "close": price, "volume": 5.})
    return hbars, dbars


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hours, cls.days = fixtures()

    def test_normalization_is_not_altered_by_fee_patches(self):
        start, end = comp.epoch("2024-02-01"), comp.epoch("2024-02-09")
        old_hour_fee, old_daily_fee = comp.hourly.FEE_BPS, comp.daily.FEE_BPS
        cheap = comp.run_sma(self.days, start, end, 15, 3)
        dear = comp.run_sma(self.days, start, end, 60, 3)
        self.assertGreater(cheap["ending_equity_usd"], dear["ending_equity_usd"])
        self.assertEqual((comp.hourly.FEE_BPS, comp.daily.FEE_BPS),
                         (old_hour_fee, old_daily_fee))
        self.assertLessEqual(cheap["ending_equity_usd"], 10_000 + 10_000)

    def test_same_days_for_sma_hourly_hold_cash(self):
        start, end = comp.epoch("2024-02-01"), comp.epoch("2024-02-09")
        rows = [
            comp.run_sma(self.days, start, end, 15, 3),
            comp.run_hourly(self.hours, start, end, 15, 3),
            comp.run_hold(self.days, start, end, 15, 3),
        ]
        for result in rows:
            self.assertIsInstance(result["net_return_pct"], float)
            self.assertGreaterEqual(result["ending_equity_usd"], 0)
            self.assertLessEqual(result["max_daily_drawdown_pct"], 100)
        self.assertEqual(comp.summarize(10_000, 10_000, [], [], 0, 1)
                         ["net_return_pct"], 0)

    def test_sma_uses_closed_prior_day_not_current(self):
        start, end = comp.epoch("2024-02-01"), comp.epoch("2024-02-04")
        seen = []
        real = comp.daily.step

        def verify(state, candles, price, now):
            seen.append((candles[-1]["time"], price, now))
            return real(state, candles, price, now)

        with patch.object(comp.daily, "step", side_effect=verify):
            comp.run_sma(self.days, start, end, 15, 3)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(prev == now - comp.DAY for prev, quote, now in seen))
        lookup = {b["time"]: b["open"] for b in self.days}
        self.assertTrue(all(quote == lookup[now] for prev, quote, now in seen))

    def test_hourly_gap_forces_isolated_comparable_windows(self):
        missing = comp.epoch("2025-10-25") + 16 * comp.HOUR
        hours = [x for x in self.hours
                 if not missing <= x["time"] < missing + 5 * comp.HOUR]
        w = comp.windows(hours, self.days)
        self.assertEqual(len([x for x in w if x["label"] == "2024"]), 1)
        later = [x for x in w if x["label"] == "2025"]
        self.assertEqual(len(later), 2)
        self.assertEqual(later[0]["end"], comp.epoch("2025-10-25"))
        self.assertGreaterEqual(later[1]["start"], comp.epoch("2025-12-15"))

    def test_cross_granularity_audit_reports_disagreement(self):
        sample = self.hours[:24]
        day = [{"time": sample[0]["time"],
                "open": sample[0]["open"], "high": sample[0]["high"],
                "low": sample[0]["low"], "close": sample[-1]["close"],
                "volume": 1.0}]
        report = comp.cross_granularity_audit(sample, day)
        self.assertEqual(report["full_days_compared"], 1)
        self.assertAlmostEqual(report["max_open_disagreement_bps"], 0)
        day[0]["close"] *= 1.01
        changed = comp.cross_granularity_audit(sample, day)
        self.assertEqual(changed["days_with_endpoint_disagreement_over_10bps"], 1)

    def test_zero_profit_factor_is_not_hidden_as_na(self):
        outcome = comp.summarize(10000, 9900, [9900],
                                  [{"pnl_usd": -100.0}], 1, 1)
        self.assertEqual(outcome["profit_factor"], 0.0)
        miniature = {"periods": [{"start": "2024-01-01", "end_exclusive": "2024-01-02",
                                   "days": 1, "cost_scenarios": [{
                                       "fee_bps_per_side": 15,
                                       "strategies": {"loser": outcome}}]}],
                     "missing_hour_intervals": [], "missing_daily_intervals": [],
                     "cross_granularity_audit": {}, "hourly_input_sha256": "x",
                     "daily_input_sha256": "y", "exclusions": []}
        rendered = comp.markdown(miniature)
        self.assertIn("| loser | -1.0 |", rendered)
        self.assertIn("| 1 | 0.0 |", rendered)

    def test_bad_daily_data_not_filled_or_traded(self):
        days = [x for x in self.days if x["time"] != comp.epoch("2025-08-01")]
        with self.assertRaisesRegex(ValueError, "Missing daily SMA200"):
            comp.windows(self.hours, days)

    def test_input_gaps_and_ohlc_invariants(self):
        comp.validate(self.hours, comp.HOUR)
        wrong = [dict(x) for x in self.hours]
        wrong[5]["low"] = wrong[5]["high"] + 2
        with self.assertRaisesRegex(ValueError, "Invalid Coinbase OHLCV"):
            comp.validate(wrong, comp.HOUR)
        self.assertEqual(comp.missing_intervals(self.hours, comp.HOUR), [])
        self.assertEqual(len(comp.missing_intervals(
            self.hours[:5] + self.hours[6:], comp.HOUR)), 1)

    def test_hold_is_marked_to_liquidation_and_cash_unchanged(self):
        start, end = comp.epoch("2024-02-01"), comp.epoch("2024-02-09")
        low = comp.run_hold(self.days, start, end, 15, 3)
        high = comp.run_hold(self.days, start, end, 60, 3)
        self.assertGreater(low["ending_equity_usd"], high["ending_equity_usd"])
        self.assertEqual(comp.summarize(10_000, 10_000, [], [], 0, 1)
                         ["ending_equity_usd"], 10_000)

    def test_reject_missing_history_and_negative_fees(self):
        with self.assertRaisesRegex(ValueError, "Expected complete hourly"):
            comp.compare(self.hours[1:], self.days, (15,), 3)
        with self.assertRaisesRegex(ValueError, "Invalid fees"):
            comp.compare(self.hours, self.days, (-1,), 3)

    def test_coinbase_pagination_and_no_current_bar(self):
        start = comp.epoch("2024-01-01")
        count = 500
        rows = [{"time": start + comp.DAY * k, "open": 100.,
                 "high": 101., "low": 99., "close": 100., "volume": 2.}
                for k in range(count)]
        calls = []
        def respond(endpoint, options):
            calls.append(options)
            st = int(datetime.fromisoformat(options["start"]).timestamp())
            en = int(datetime.fromisoformat(options["end"]).timestamp())
            return [[r["time"],r["low"],r["high"],r["open"],r["close"],r["volume"]]
                    for r in rows if st <= r["time"] < en]
        with patch.object(comp.hourly, "request_json", side_effect=respond), \
             patch.object(comp.time, "sleep"):
            retrieved = comp.fetch(start, start + count * comp.DAY, comp.DAY)
        self.assertEqual(retrieved, rows)
        self.assertTrue(all((datetime.fromisoformat(x["end"]) -
                             datetime.fromisoformat(x["start"])).total_seconds()
                            <= 288 * comp.DAY for x in calls))


if __name__ == "__main__":
    unittest.main()
