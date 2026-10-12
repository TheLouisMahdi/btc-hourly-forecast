"""Offline tests for the isolated Coinbase spot research harness."""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rule_based as engine
import spot_backtest as research


def hourly(count: int = 1240, start: int = 1704067200) -> list[dict]:
    bars = []
    for i in range(count):
        close = 100 + 0.01 * i + (5 if i >= 1210 else 0)
        bars.append({"time": start + i * engine.HOUR, "open": close,
                     "high": close + 0.5, "low": close - 0.5,
                     "close": close, "volume": 10.0})
    return bars


class SpotBacktestTests(unittest.TestCase):
    def test_validates_gaps_and_duplicates(self):
        bars = hourly()
        research.validate(bars)
        bars[500]["time"] += engine.HOUR
        with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
            research.validate(bars)

    def test_validates_malformed_ohlc(self):
        bars = hourly()
        bars[-1]["low"] = bars[-1]["high"] + 1
        with self.assertRaisesRegex(ValueError, "OHLCV"):
            research.validate(bars)

    def test_causal_next_open_cost_stress_and_runtime_unchanged(self):
        bars = hourly()
        original_costs = (engine.FEE_BPS, engine.SLIPPAGE_BPS)
        cheap = research.replay(bars, 1199, len(bars), 15, 3)
        expensive = research.replay(bars, 1199, len(bars), 60, 3)
        self.assertGreater(cheap["executions"], 0)
        self.assertGreater(cheap["equity_usd"], expensive["equity_usd"])
        self.assertEqual((engine.FEE_BPS, engine.SLIPPAGE_BPS), original_costs)

    def test_report_uses_fixed_2025_holdout_without_selection(self):
        bars = hourly(count=17544)
        with patch.object(research, "replay", return_value={"test": True}) as replay:
            report = research.report(bars, (15.0, 60.0))
        self.assertEqual(len(report["scenarios"]["2024"]), 2)
        self.assertEqual(len(report["scenarios"]["2025_holdout"]), 2)
        self.assertEqual(len(report["scenarios"]["full"]), 2)
        split = research.timestamp("2025-01-01")
        self.assertTrue(all(
            bars[call.args[1]]["time"] >= split
            for call in replay.call_args_list[2:4]
        ))
        self.assertTrue(all(
            bars[call.args[1]]["time"] < split
            for call in replay.call_args_list[:2]
        ))

    def test_no_partial_or_future_history_accepted(self):
        bars = hourly()
        with self.assertRaisesRegex(ValueError, "requested end"):
            research.validate(bars, start=bars[0]["time"],
                              end=bars[-1]["time"] + 2 * engine.HOUR)
        with self.assertRaisesRegex(ValueError, "Invalid evaluation"):
            research.replay(bars, 200, len(bars), 15, 3)

    def test_provider_windows_are_within_coinbase_limit(self):
        bars = hourly()
        calls = []
        def fake_request(path, params):
            calls.append(params)
            start = datetime.fromisoformat(params["start"]).timestamp()
            end = datetime.fromisoformat(params["end"]).timestamp()
            return [[b["time"], b["low"], b["high"], b["open"],
                     b["close"], b["volume"]]
                    for b in bars if start <= b["time"] < end]
        with (patch.object(engine, "request_json", side_effect=fake_request),
              patch.object(research.time, "sleep")):
            fetched = research.download_coinbase(
                bars[0]["time"], bars[-1]["time"] + engine.HOUR)
        self.assertEqual(fetched, bars)
        self.assertTrue(all(
            datetime.fromisoformat(c["end"]).timestamp()
            - datetime.fromisoformat(c["start"]).timestamp()
            <= 288 * engine.HOUR for c in calls
        ))


if __name__ == "__main__":
    unittest.main()
