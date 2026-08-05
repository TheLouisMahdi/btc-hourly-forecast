from __future__ import annotations

import sys
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
        }


def test_final_trade_economics_uses_decision_risk_fraction() -> None:
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

    assert output["risk_fraction"] == 0.025
    assert output["risk_budget_usd"] == 25.0
    assert output["notional_usd"] > 0.0
    assert output["target_net_profit_usd"] > 0.0
    assert output["stop_net_loss_usd"] < 0.0


def test_new_position_persists_policy_and_risk_contract() -> None:
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
            "policy_name": "AGGRESSIVE_STRUCTURAL_RISK_SCALED",
            "policy_version": 2,
            "risk_contract_version": 2,
            "entry_contract": "STRUCTURAL_EVENT_RISK_SCALED",
            "soft_risk_flags": ["MODEL_NOT_QUALIFIED"],
            "qualification_passed": False,
            "direction_qualified": False,
        },
    }

    trade = open_trade_with_context(record)

    assert trade is not None
    assert trade["policy_name"] == "AGGRESSIVE_STRUCTURAL_RISK_SCALED"
    assert trade["policy_version"] == 2
    assert trade["risk_contract_version"] == 2
    assert trade["entry_contract"] == "STRUCTURAL_EVENT_RISK_SCALED"
    assert trade["risk_fraction"] == 0.02
    assert trade["soft_risk_flags"] == ["MODEL_NOT_QUALIFIED"]
