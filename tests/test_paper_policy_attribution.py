"""Unit tests for the fixed-opportunity attribution report."""
from __future__ import annotations

import unittest

from scripts.compare_observed_paper_trades import (
    candle_coverage,
    compare,
    markdown_report,
    validate_ledger,
)


def trade(
    ident: str,
    event: str,
    pnl: float,
    close_hour: int,
    *,
    cost: float | None = 0.50,
) -> dict:
    result = {
        "trade_id": ident,
        "status": "CLOSED",
        "event_type": event,
        "opened_at": "2026-01-01T00:20:00+00:00",
        "closed_at": f"2026-01-01T{close_hour:02d}:00:00+00:00",
        "notional_usd": 1000.0,
        "gross_aligned_return": (pnl + 0.50) / 1000.0,
        "realized_net_return": pnl / 1000.0,
        "realized_net_pnl_usd": pnl,
    }
    if cost is not None:
        result["simulated_execution_cost_usd"] = cost
    return result


class PaperPolicyAttributionTests(unittest.TestCase):
    def test_same_opportunities_separate_retained_and_excluded(self) -> None:
        rows = [
            trade("loser", "NONE", -8.0, 3),
            trade("winner", "NONE", 2.0, 2, cost=None),
            trade("event", "RESISTANCE_BREAKOUT_LONG", 4.0, 4),
        ]
        report = compare(rows)
        original = report["original_recorded_opportunities"]
        retained = report["retained_structural_opportunities"]
        excluded = report["excluded_model_only_opportunities"]
        self.assertEqual(original["closed_trades"], 3)
        self.assertEqual(original["net_pnl_usd"], -2.0)
        self.assertEqual(original["execution_costs_usd"], 1.5)
        self.assertEqual(original["realized_close_only_max_drawdown_usd"], 8.0)
        self.assertEqual(retained["net_pnl_usd"], 4.0)
        self.assertEqual(retained["closed_trades"], 1)
        self.assertEqual(excluded["net_pnl_usd"], -6.0)
        self.assertEqual(report["excluded_profitable_trade_count"], 1)
        self.assertEqual(report["excluded_trade_ids"], ["winner", "loser"])
        self.assertIn("NOT_BACKTEST", report["analysis_type"])
        self.assertIn("not a backtest", markdown_report(report).lower())

    def test_profit_factor_without_losses_is_undefined(self) -> None:
        report = compare([trade("one", "SUPPORT_BREAKDOWN_SHORT", 1.0, 3)])
        self.assertIsNone(
            report["retained_structural_opportunities"]["profit_factor"]
        )

    def test_reject_duplicate_trade_ids_and_nonfinite_pnl(self) -> None:
        row = trade("dup", "NONE", 1.0, 2)
        with self.assertRaises(ValueError):
            validate_ledger([row, dict(row)])
        invalid = dict(row, trade_id="other", realized_net_pnl_usd=float("nan"))
        with self.assertRaises(ValueError):
            validate_ledger([invalid])

    def test_open_trades_do_not_count_as_closed_opportunities(self) -> None:
        open_row = trade("open", "NONE", 1.0, 2)
        open_row["status"] = "OPEN"
        report = compare([trade("closed", "NONE", -2.0, 2), open_row])
        self.assertEqual(report["original_recorded_opportunities"]["closed_trades"], 1)

    def test_partial_price_history_is_reported_not_fabricated(self) -> None:
        rows = [
            trade("covered", "RESISTANCE_BREAKOUT_LONG", 1.0, 2),
            dict(
                trade("missing", "NONE", -2.0, 3),
                opened_at="2025-12-31T20:00:00+00:00",
            ),
        ]
        candles = [
            {"open_time": "2026-01-01T00:00:00+00:00"},
            {"open_time": "2026-01-01T01:00:00+00:00"},
        ]
        coverage = candle_coverage(candles, validate_ledger(rows))
        self.assertEqual(coverage["hourly_candles"], 2)
        self.assertEqual(coverage["fully_covered_closed_trades"], 1)
        self.assertEqual(coverage["fully_covered_structural_trades"], 1)


if __name__ == "__main__":
    unittest.main()
