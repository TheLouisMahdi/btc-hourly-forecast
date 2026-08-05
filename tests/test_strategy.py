from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from btc_ema_trader.strategy import make_decision


def _row(**overrides):
    values = {
        "close": 70_000.0,
        "atr": 500.0,
        "atr_pct": 0.007,
        "is_event": 1,
        "event_id": "V5-TEST-LONG",
        "event_type": "RESISTANCE_BREAKOUT_LONG",
        "event_direction": 1,
        "event_score": 0.30,
        "event_scale_hours": 168,
        "breakout_source": "RESISTANCE_168H",
        "breakout_level": 69_900.0,
        "breakout_invalidation_level": 69_300.0,
        "regime": "STRUCTURE_UP",
        "regime_code": 1.0,
        "news_shock": 0,
        "adx": 22.0,
        "rsi_centered": 0.10,
        "volume_z_24": 0.4,
    }
    values.update(overrides)
    return pd.Series(values)


def _prediction(
    *,
    success: float = 0.50,
    tradeability: float = 0.50,
    event_return: float = 0.0,
):
    return {
        "direction": "UP",
        "trade_direction": "UP",
        "confidence": success,
        "agreement": 1.0,
        "expected_return": event_return,
        "expected_event_aligned_return": event_return,
        "selected_horizon": 3,
        "qualified_trade_horizons": [],
        "probabilities": {1: 0.52, 3: 0.55, 6: 0.53, 12: 0.51},
        "continuation": {1: 0.50, 3: success, 6: 0.48, 12: 0.46},
        "tradeability": {1: 0.50, 3: tradeability, 6: 0.48, 12: 0.46},
        "event_returns": {1: 0.0, 3: event_return, 6: -0.001, 12: -0.002},
        "returns": {1: 0.0, 3: event_return, 6: -0.001, 12: -0.002},
        "absolute_event_returns": {
            1: 0.0,
            3: event_return,
            6: -0.001,
            12: -0.002,
        },
    }


def _unqualified_bundle():
    return SimpleNamespace(
        qualification={
            "passed": False,
            "qualified_directions": {"LONG": [], "SHORT": []},
            "economic_policy": {"LONG": {}, "SHORT": {}},
            "economic_stress_cost_bps": 21.0,
        }
    )


def _qualified_bundle():
    return SimpleNamespace(
        qualification={
            "passed": True,
            "qualified_directions": {"LONG": [3], "SHORT": []},
            "economic_stress_cost_bps": 15.0,
            "economic_policy": {
                "LONG": {
                    "3": {
                        "success_probability": 0.58,
                        "tradeability_probability": 0.56,
                        "minimum_event_score": 0.10,
                        "minimum_predicted_stress_edge_bps": 0.0,
                    }
                },
                "SHORT": {},
            },
        }
    )


def test_unqualified_negative_edge_event_still_opens_with_scaled_risk(settings):
    decision = make_decision(
        _row(),
        _prediction(success=0.50, tradeability=0.50, event_return=0.0),
        _unqualified_bundle(),
        settings,
    )

    assert decision.action == "LONG"
    assert decision.blockers == []
    assert "MODEL_NOT_QUALIFIED" in decision.trade_plan["soft_risk_flags"]
    assert (
        "SELECTED_DIRECTION_NOT_QUALIFIED"
        in decision.trade_plan["soft_risk_flags"]
    )
    assert "INSUFFICIENT_STRESS_NET_EDGE" in decision.trade_plan["soft_risk_flags"]
    assert decision.trade_plan["policy_name"] == (
        "AGGRESSIVE_STRUCTURAL_RISK_SCALED"
    )
    assert decision.trade_plan["policy_version"] == 2
    assert 0.005 <= decision.trade_plan["risk_fraction"] <= 0.03
    assert decision.trade_plan["risk_budget_usd"] > 0.0


def test_strong_qualified_event_receives_more_risk_than_weak_event(settings):
    weak = make_decision(
        _row(),
        _prediction(success=0.50, tradeability=0.50, event_return=0.0),
        _unqualified_bundle(),
        settings,
    )
    strong = make_decision(
        _row(event_score=0.95, volume_z_24=2.0, adx=34.0),
        _prediction(success=0.78, tradeability=0.76, event_return=0.005),
        _qualified_bundle(),
        settings,
    )

    assert strong.action == "LONG"
    assert strong.blockers == []
    assert strong.trade_plan["soft_risk_flags"] == []
    assert strong.trade_plan["risk_fraction"] > weak.trade_plan["risk_fraction"]
    assert strong.trade_plan["risk_fraction"] <= 0.03
    assert strong.trade_plan["risk_score"] > weak.trade_plan["risk_score"]


def test_no_event_remains_a_hard_blocker(settings):
    decision = make_decision(
        _row(
            is_event=0,
            event_direction=0,
            event_type="NONE",
            event_id=None,
            breakout_level=None,
            breakout_invalidation_level=None,
        ),
        _prediction(success=0.80, tradeability=0.80, event_return=0.01),
        _qualified_bundle(),
        settings,
    )

    assert decision.action == "WAIT"
    assert "NO_NEW_STRUCTURE_BREAKOUT" in decision.blockers


def test_missing_invalidation_remains_a_hard_blocker(settings):
    decision = make_decision(
        _row(breakout_invalidation_level=None),
        _prediction(success=0.80, tradeability=0.80, event_return=0.01),
        _qualified_bundle(),
        settings,
    )

    assert decision.action == "WAIT"
    assert "INVALIDATION_LEVEL_UNAVAILABLE" in decision.blockers


def test_unhealthy_market_data_remains_a_hard_blocker(settings):
    decision = make_decision(
        _row(),
        _prediction(success=0.80, tradeability=0.80, event_return=0.01),
        _qualified_bundle(),
        settings,
        data_health={
            "candles_ok": False,
            "quote_ok": True,
            "provider_mismatch": False,
            "model_stale": False,
            "news_stale": False,
        },
    )

    assert decision.action == "WAIT"
    assert "CANDLE_DATA_UNHEALTHY" in decision.blockers
