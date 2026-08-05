from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

from github_structural_forecast import (
    _apply_risk_scaled_economics,
    open_trade_with_context,
)


class _Settings:
    def section(self, name: str):
        if name != "strategy":
            raise KeyError(name)
        return {
            "paper_only": True,
            "account_equity_usd": 1000.0,
            "risk_per_trade_fraction": 0.0125,
            "minimum_risk_per_trade_fraction": 0.005,
            "maximum_risk_per_trade_fraction": 0.03,
            "maximum_leverage": 5.0,
            "gap_risk_buffer_bps": 6.0,
        }


class AggressiveRiskContractTests(unittest.TestCase):
    def test_final_trade_economics_uses_one_gap_aware_risk_budget(self) -> None:
        plan = {
            "entry_reference": 100.0,
            "stop_percent": 0.01,
            "target_percent": 0.05,
            "stress_execution_cost_bps": 20.0,
            "risk_fraction": 0.025,
            "adaptive_target_probability": 0.55,
            "adaptive_stop_probability": 0.35,
            "adaptive_predicted_r": 1.5,
        }

        output = _apply_risk_scaled_economics(plan, _Settings())

        self.assertEqual(output["risk_fraction"], 0.025)
        self.assertEqual(output["risk_budget_usd"], 25.0)
        self.assertEqual(output["gap_risk_buffer_bps"], 6.0)
        self.assertGreater(output["notional_usd"], 0.0)
        self.assertLessEqual(
            output["modeled_total_risk_usd"],
            output["risk_budget_usd"] + 1e-9,
        )
        self.assertLessEqual(output["risk_budget_utilization"], 1.0 + 1e-12)
        self.assertGreater(output["target_net_profit_usd"], 0.0)
        self.assertLess(output["stop_net_loss_usd"], 0.0)

    def test_new_position_persists_policy_and_risk_contract(self) -> None:
        record = {
            "action": "LONG",
            "model_id": "directional-breakout-hourly-test",
            "event_id": "V5-TEST-LONG",
            "event_type": "RESISTANCE_BREAKOUT_LONG",
            "candle_time": "2026-08-05T10:00:00+00:00",
            "run_finished_at": "2026-08-05T11:01:00+00:00",
            "trade_plan": {
                "status": "ACTIONABLE",
                "entry_reference": 100.0,
                "stop_price": 99.0,
                "target_price": 105.0,
                "risk_reward": 5.0,
                "maximum_holding_hours": 72,
                "risk_budget_usd": 20.0,
                "risk_fraction": 0.02,
                "risk_score": 0.60,
                "risk_assessment": {"components": {"confidence": 0.5}},
                "policy_name": "AGGRESSIVE_STRUCTURAL_RISK_SCALED",
                "policy_version": 2,
                "risk_contract_version": 2,
                "entry_contract": "STRUCTURAL_EVENT_RISK_SCALED",
                "soft_risk_flags": ["MODEL_NOT_QUALIFIED"],
                "qualification_passed": False,
                "direction_qualified": False,
                "modeled_total_risk_usd": 20.0,
                "gap_risk_buffer_bps": 6.0,
                "label_execution_aligned": False,
                "label_entry_definition": "NEXT_HOURLY_OPEN",
                "runtime_entry_definition": "LIVE_QUOTE_AT_SIGNAL_RUN",
            },
        }

        trade = open_trade_with_context(record)

        self.assertIsNotNone(trade)
        assert trade is not None
        self.assertEqual(
            trade["policy_name"],
            "AGGRESSIVE_STRUCTURAL_RISK_SCALED",
        )
        self.assertEqual(trade["policy_version"], 2)
        self.assertEqual(trade["risk_contract_version"], 2)
        self.assertEqual(
            trade["entry_contract"],
            "STRUCTURAL_EVENT_RISK_SCALED",
        )
        self.assertEqual(trade["risk_fraction"], 0.02)
        self.assertEqual(trade["gap_risk_buffer_bps"], 6.0)
        self.assertFalse(trade["label_execution_aligned"])
        self.assertEqual(
            trade["label_entry_definition"],
            "NEXT_HOURLY_OPEN",
        )
        self.assertEqual(
            trade["soft_risk_flags"],
            ["MODEL_NOT_QUALIFIED"],
        )


if __name__ == "__main__":
    unittest.main()
