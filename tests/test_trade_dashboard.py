from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

from github_trade_dashboard import _panel, _position_ledger


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

    def test_corrected_history_displays_corrected_win_loss(self) -> None:
        latest = {"price": 100.0}
        trades = [
            {
                "status": "CLOSED",
                "direction": "LONG",
                "opened_at": "2026-01-01T01:10:00Z",
                "closed_at": "2026-01-01T04:00:00Z",
                "entry_price": 100.0,
                "target_price": 105.0,
                "initial_stop_price": 99.0,
                "exit_price": 101.0,
                "realized_net_pnl_usd": 8.0,
                "realized_net_return": 0.008,
                "realized_r": 0.8,
                "outcome": "STOP",
                "historical_corrected_result": "WIN",
            }
        ]

        document = _position_ledger(latest, trades)

        self.assertIn(">WIN<", document)
        self.assertNotIn(">STOP<", document)

    def test_panel_concentrates_profit_first_position_evidence(self) -> None:
        latest = {
            "action": "LONG",
            "trade_plan": {
                "status": "ACTIONABLE",
                "entry_reference": 100.0,
                "target_price": 105.0,
                "stop_price": 99.0,
                "risk_budget_usd": 20.0,
                "risk_fraction": 0.02,
                "risk_score": 0.60,
                "policy_name": "AGGRESSIVE_STRUCTURAL_RISK_SCALED",
                "policy_version": 2,
                "qualification_passed": False,
                "direction_qualified": False,
                "soft_risk_flags": [
                    "MODEL_NOT_QUALIFIED",
                    "INSUFFICIENT_STRESS_NET_EDGE",
                ],
            },
            "trade_lifecycle_summary": {},
        }

        latest["trade_plan"].update(
            {
                "expected_value_usd": 6.0,
                "target_net_profit_usd": 18.0,
                "stop_net_loss_usd": -9.0,
                "adaptive_profit_probability": 0.64,
                "adaptive_loss_probability": 0.36,
                "adaptive_target_probability": 0.68,
                "adaptive_stop_probability": 0.22,
                "position_decision": "FAVORABLE",
                "position_decision_reason": (
                    "MULTIPLE_PROFIT_SIGNALS_ALIGNED"
                ),
                "position_metrics": {
                    "expected_value_to_risk": 0.30,
                    "net_reward_risk": 2.0,
                },
            }
        )

        document = _panel(latest, [])

        self.assertIn("Position decision", document)
        self.assertIn("LONG · FAVORABLE", document)
        self.assertIn("Expected value", document)
        self.assertIn("$+6.00", document)
        self.assertIn("+30.00% of risk budget", document)
        self.assertIn("Profit / loss", document)
        self.assertIn("$+18.00 / $-9.00", document)
        self.assertIn("2.00:1", document)
        self.assertIn("PROFIT 64.00% · LOSS 36.00%", document)
        self.assertIn("MODEL NOT QUALIFIED", document)


if __name__ == "__main__":
    unittest.main()
