"""Tests for replaying identical recorded entries with two exit engines."""
from __future__ import annotations

import unittest

from scripts.replay_recorded_exits import compare_replays, replay_entries


class RecordedExitReplayTests(unittest.TestCase):
    def _trade(self) -> dict:
        return {
            "trade_id": "frozen-entry",
            "status": "CLOSED",
            "event_type": "RESISTANCE_BREAKOUT_LONG",
            "direction": "LONG",
            "opened_at": "2026-01-01T01:04:00+00:00",
            "closed_at": "2026-01-01T03:00:00+00:00",
            "expires_at": "2026-01-02T01:04:00+00:00",
            "entry_price": 100.0,
            "initial_stop_price": 99.0,
            "current_stop_price": 100.5,
            "target_price": 105.0,
            "initial_risk_price": 1.0,
            "entry_atr": 1.0,
            "notional_usd": 1000.0,
            "risk_budget_usd": 10.0,
            "stress_execution_cost_bps": 10.0,
            "breakeven_trigger_r": 2.0,
            "trailing_trigger_r": 3.0,
            "trailing_atr_multiplier": 1.0,
            "max_favorable_r": 9.0,
            "outcome": "STOP",
            "realized_net_pnl_usd": -9.0,
        }

    def _candles(self) -> list[dict]:
        return [
            {
                "open_time": "2026-01-01T01:00:00Z",
                "open": 100.0,
                "high": 110.0,
                "low": 80.0,
                "close": 100.1,
            },
            {
                "open_time": "2026-01-01T02:00:00Z",
                "open": 100.2,
                "high": 100.5,
                "low": 98.8,
                "close": 99.5,
            },
            {
                "open_time": "2026-01-01T03:00:00Z",
                "open": 99.5,
                "high": 99.7,
                "low": 99.0,
                "close": 99.2,
            },
        ]

    def test_replay_resets_realized_trade_and_skips_partial_entry_hour(self) -> None:
        source = self._trade()
        result = replay_entries([source], self._candles())
        self.assertEqual(result["eligible_recorded_entries"], 1)
        reconstructed = result["trades"][0]
        self.assertEqual(reconstructed["replayed_outcome"], "STOP")
        self.assertEqual(reconstructed["replayed_exit_price"], 99.0)
        self.assertEqual(reconstructed["replayed_closed_at"], "2026-01-01T03:00:00+00:00")
        self.assertNotEqual(reconstructed["replayed_net_pnl_usd"], -9.0)
        self.assertEqual(source["current_stop_price"], 100.5)
        self.assertEqual(source["max_favorable_r"], 9.0)

    def test_compare_requires_identical_opportunities(self) -> None:
        baseline = {
            "trades": [
                {
                    "trade_id": "a",
                    "event_type": "NONE",
                    "resolved": True,
                    "replayed_net_pnl_usd": -5.0,
                    "replayed_outcome": "STOP",
                    "replayed_fill_reason": "INTRABAR_STOP",
                    "replayed_closed_at": "2026-01-01T03:00:00+00:00",
                    "original_recorded_net_pnl_usd": -5.0,
                }
            ]
        }
        candidate = {
            "trades": [
                dict(
                    baseline["trades"][0],
                    replayed_net_pnl_usd=-3.0,
                    replayed_closed_at="2026-01-01T02:00:00+00:00",
                )
            ]
        }
        result = compare_replays(baseline, candidate)
        self.assertEqual(result["paired_resolved"], 1)
        self.assertEqual(result["candidate_minus_baseline_net_pnl_usd"], 2.0)
        self.assertIn("NOT a full strategy backtest", result["warning"])
        candidate["trades"][0]["trade_id"] = "other"
        with self.assertRaises(ValueError):
            compare_replays(baseline, candidate)

    def test_missing_intrahour_history_is_never_silently_filled(self) -> None:
        candles = self._candles()
        candles[1]["open_time"] = "2026-01-01T04:00:00Z"
        with self.assertRaises(ValueError):
            replay_entries([self._trade()], candles)


if __name__ == "__main__":
    unittest.main()
