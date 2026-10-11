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
            "maximum_leverage": 40.0,
            "leverage_tiers": [10.0, 20.0, 40.0],
            "leverage_risk_score_thresholds": [0.35, 0.65],
            "leverage_liquidation_safety_factor": 0.70,
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
            "risk_score": 0.80,
            "adaptive_target_probability": 0.55,
            "adaptive_stop_probability": 0.35,
            "adaptive_predicted_r": 1.5,
            "direction": "LONG",
            "stop_price": 99.0,
            "target_price": 105.0,
        }

        output = _apply_risk_scaled_economics(plan, _Settings())

        self.assertEqual(output["risk_fraction"], 0.025)
        self.assertEqual(output["risk_budget_usd"], 25.0)
        self.assertEqual(output["gap_risk_buffer_bps"], 6.0)
        self.assertGreater(output["notional_usd"], 0.0)
        self.assertEqual(output["suggested_leverage"], 40.0)
        self.assertLessEqual(
            output["modeled_total_risk_usd"],
            output["risk_budget_usd"] + 1e-9,
        )
        self.assertLessEqual(output["risk_budget_utilization"], 1.0 + 1e-12)
        self.assertGreater(output["target_net_profit_usd"], 0.0)
        self.assertLess(output["stop_net_loss_usd"], 0.0)
        self.assertTrue(output["liquidation_safety_ok"])
        self.assertEqual(output["maintenance_margin_rate"], 0.004)
        self.assertAlmostEqual(
            output["estimated_liquidation_price"],
            100.0 * (1.0 - 1.0 / 40.0) / (1.0 - 0.004),
            places=9,
        )
        self.assertGreater(
            output["stop_to_liquidation_buffer_percent"],
            0.0,
        )

    def test_new_position_persists_policy_and_risk_contract(self) -> None:
        record = {
            "action": "LONG",
            "model_id": "directional-breakout-hourly-test",
            "event_id": "V5-TEST-LONG",
            "event_type": "RESISTANCE_BREAKOUT_LONG",
            "expected_net_edge_bps": 15.5,
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
                "economic_policy_available": False,
                "minimum_required_net_edge_bps": 12.0,
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
        self.assertFalse(trade["economic_policy_available"])
        self.assertEqual(trade["expected_net_edge_bps"], 15.5)
        self.assertEqual(trade["minimum_required_net_edge_bps"], 12.0)

    def test_missing_economic_entry_evidence_is_not_fabricated(self) -> None:
        record = {
            "action": "LONG",
            "candle_time": "2026-08-05T10:00:00+00:00",
            "trade_plan": {
                "status": "ACTIONABLE",
                "entry_reference": 100.0,
                "stop_price": 99.0,
                "target_price": 105.0,
            },
        }
        trade = open_trade_with_context(record)
        self.assertIsNotNone(trade)
        assert trade is not None
        self.assertNotIn("economic_policy_available", trade)
        self.assertNotIn("expected_net_edge_bps", trade)
        self.assertNotIn("minimum_required_net_edge_bps", trade)

    def test_nonfinite_expected_edge_is_not_saved(self) -> None:
        record = {
            "action": "LONG",
            "expected_net_edge_bps": float("nan"),
            "trade_plan": {
                "status": "ACTIONABLE",
                "entry_reference": 100.0,
                "stop_price": 99.0,
                "target_price": 105.0,
                "economic_policy_available": True,
            },
        }
        trade = open_trade_with_context(record)
        self.assertIsNotNone(trade)
        assert trade is not None
        self.assertTrue(trade["economic_policy_available"])
        self.assertNotIn("expected_net_edge_bps", trade)


if __name__ == "__main__":
    unittest.main()
