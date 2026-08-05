from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

from github_trade_dashboard import _position_ledger


class TradeDashboardTests(unittest.TestCase):
    def test_ledger_contains_only_long_and_short_positions(self) -> None:
        latest = {"price": 102.0}
        trades = [
            {
                "status": "OPEN",
                "direction": "LONG",
                "opened_at": "2026-01-01T01:10:00Z",
                "entry_price": 100.0,
                "target_price": 105.0,
                "initial_stop_price": 99.0,
                "current_stop_price": 99.5,
                "notional_usd": 1000.0,
                "risk_budget_usd": 10.0,
                "stress_execution_cost_bps": 10.0,
            },
            {
                "status": "WAIT",
                "direction": "NONE",
                "entry_price": 100.0,
            },
        ]
        document = _position_ledger(latest, trades)
        self.assertIn("LONG / SHORT position ledger", document)
        self.assertIn(">LONG<", document)
        self.assertIn("$+19.00", document)
        self.assertNotIn("Expected close", document)
        self.assertNotIn("Model range", document)
        self.assertNotIn(">NONE<", document)

    def test_closed_position_uses_immutable_realized_pnl(self) -> None:
        latest = {"price": 999.0}
        trades = [
            {
                "status": "CLOSED",
                "direction": "SHORT",
                "opened_at": "2026-01-01T01:10:00Z",
                "closed_at": "2026-01-01T04:00:00Z",
                "entry_price": 100.0,
                "target_price": 95.0,
                "initial_stop_price": 101.0,
                "exit_price": 96.0,
                "realized_net_pnl_usd": 38.0,
                "realized_net_return": 0.038,
                "realized_r": 3.8,
                "outcome": "TIME_EXIT_WIN",
            }
        ]
        document = _position_ledger(latest, trades)
        self.assertIn("$+38.00", document)
        self.assertIn("+3.80%", document)
        self.assertIn("+3.80R", document)
        self.assertIn("TIME EXIT WIN", document)
        self.assertNotIn("$999.00", document)


if __name__ == "__main__":
    unittest.main()
