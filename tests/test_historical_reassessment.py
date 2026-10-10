"""Historical reassessment must never rewrite completed or open paper trades."""
from __future__ import annotations

import copy
import unittest

from scripts.build_historical_reassessment import build_reassessment


class HistoricalReassessmentTests(unittest.TestCase):
    def _trade(self, ident: str, event: str = "NONE", *, status: str = "CLOSED") -> dict:
        return {
            "trade_id": ident,
            "status": status,
            "event_type": event,
            "direction": "LONG",
            "opened_at": "2026-01-01T01:04:00+00:00",
            "closed_at": "2026-01-01T03:00:00+00:00",
            "entry_price": 100.0,
            "initial_stop_price": 99.0,
            "current_stop_price": 99.0,
            "target_price": 105.0,
            "initial_risk_price": 1.0,
            "entry_atr": 1.0,
            "expires_at": "2026-01-02T01:04:00+00:00",
            "notional_usd": 1000.0,
            "risk_budget_usd": 10.0,
            "stress_execution_cost_bps": 10.0,
            "breakeven_trigger_r": 2.0,
            "trailing_trigger_r": 3.0,
            "trailing_atr_multiplier": 1.0,
            "gross_aligned_return": 0.002,
            "realized_net_return": 0.001,
            "realized_net_pnl_usd": 1.0,
        }

    def _candles(self) -> list[dict]:
        return [
            {"open_time": "2026-01-01T01:00:00+00:00",
             "open": 100.0, "high": 107.0, "low": 90.0, "close": 100.0},
            {"open_time": "2026-01-01T02:00:00+00:00",
             "open": 100.0, "high": 100.5, "low": 98.9, "close": 99.8},
            {"open_time": "2026-01-01T03:00:00+00:00",
             "open": 99.8, "high": 100.1, "low": 99.2, "close": 99.6},
        ]

    def test_preserves_raw_records_and_tracks_replay_separately(self) -> None:
        trades = [self._trade("structural", "RESISTANCE_BREAKOUT_LONG"),
                  self._trade("model-only"), self._trade("open", status="OPEN")]
        original = copy.deepcopy(trades)
        report = build_reassessment(trades, self._candles())
        self.assertEqual(trades, original)
        self.assertEqual(report["original_closed_trades"]["closed_trades"], 2)
        self.assertEqual(report["historically_structural_subset"]["closed_trades"], 1)
        self.assertEqual(report["historically_profitable_but_now_blocked_entries"], 1)
        self.assertEqual(report["currently_open_trades_are_not_repriced"], ["open"])
        self.assertEqual(report["closed_trades_with_hourly_exit_replay"], 2)
        row = next(x for x in report["entries"] if x["trade_id"] == "model-only")
        self.assertEqual(row["original_net_pnl_usd"], 1.0)
        self.assertEqual(row["historical_policy_class"],
                         "MODEL_ONLY_ENTRY_BLOCKED_BY_NEW_POLICY")
        self.assertNotEqual(row["exit_replay_estimate"]["net_pnl_usd"], 1.0)

    def test_rollover_preserves_prior_estimates(self) -> None:
        row = self._trade("old")
        first = build_reassessment([row], self._candles())
        second = build_reassessment([row], [], previous=first)
        self.assertEqual(first["entries"][0]["exit_replay_estimate"],
                         second["entries"][0]["exit_replay_estimate"])
        self.assertIsNotNone(second["replay_warning"])
        self.assertEqual(second["closed_trades_with_hourly_exit_replay"], 1)

    def test_modified_source_does_not_keep_stale_replay(self) -> None:
        row = self._trade("old")
        first = build_reassessment([row], self._candles())
        changed = dict(row, realized_net_pnl_usd=9.0)
        next_report = build_reassessment([changed], [], previous=first)
        self.assertNotIn("exit_replay_estimate", next_report["entries"][0])
        self.assertEqual(next_report["closed_trades_with_hourly_exit_replay"], 0)


if __name__ == "__main__":
    unittest.main()
