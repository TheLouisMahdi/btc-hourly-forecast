from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .costs import execution_cost_breakdown
from .economic_validation import apply_calibration
from .model import HourlyModelBundle

STRUCTURAL_EVENTS = {
    "RESISTANCE_BREAKOUT_LONG",
    "TRIANGLE_BREAKOUT_LONG",
    "SUPPORT_BREAKDOWN_SHORT",
    "TRIANGLE_BREAKDOWN_SHORT",
}


@dataclass(frozen=True)
class Decision:
    forecast_direction: str
    action: str
    confidence: float
    tradeability_probability: float
    expected_return: float
    expected_net_edge_bps: float
    selected_horizon: int
    probabilities: dict[int, float]
    tradeability: dict[int, float]
    returns: dict[int, float]
    blockers: list[str]
    trade_plan: dict[str, Any]


def make_decision(
    latest_row: pd.Series,
    prediction: dict[str, Any],
    bundle: HourlyModelBundle,
    settings: Settings,
    data_health: dict[str, Any] | None = None,
    recent_signal_count: int = 0,
    hours_since_last_signal: float | None = None,
    event_already_traded: bool = False,
) -> Decision:
    cfg = settings.section("strategy")
    qualification = bundle.qualification or {}
    forecast_direction = str(prediction["direction"])
    trade_direction = str(
        prediction.get("trade_direction", forecast_direction)
    )

    is_event = int(latest_row.get("is_event", 0)) == 1
    event_direction = int(latest_row.get("event_direction", 0))
    direction_name = (
        "LONG"
        if event_direction > 0
        else "SHORT"
        if event_direction < 0
        else "NONE"
    )
    event_type = str(latest_row.get("event_type", "NONE"))
    event_score = float(latest_row.get("event_score", 0.0))
    breakout_level = _finite(latest_row.get("breakout_level"))
    invalidation_level = _finite(
        latest_row.get("breakout_invalidation_level")
    )

    qualified_by_direction = qualification.get(
        "qualified_directions", {}
    )
    qualified_horizons = {
        int(horizon)
        for horizon in qualified_by_direction.get(direction_name, [])
    }
    policy_by_direction = qualification.get("economic_policy", {})
    direction_policies = policy_by_direction.get(direction_name, {})

    selected_horizon, confidence, tradeability_probability, policy = (
        _select_economic_horizon(
            prediction=prediction,
            qualified_horizons=qualified_horizons,
            policies=direction_policies,
            event_score=event_score,
        )
    )
    event_aligned_return = _mapping_value(
        prediction.get("event_returns", {}),
        selected_horizon,
        float(prediction.get("expected_event_aligned_return", 0.0)),
    )
    expected_return = (
        event_aligned_return
        if event_direction >= 0
        else -event_aligned_return
    )
    base_costs = execution_cost_breakdown(cfg)
    stress_cost_bps = float(
        qualification.get(
            "economic_stress_cost_bps",
            base_costs["stress_cost_bps"],
        )
    )
    net_edge_bps = event_aligned_return * 10_000.0 - stress_cost_bps
    minimum_edge_bps = float(
        policy.get(
            "minimum_predicted_stress_edge_bps",
            cfg.get("minimum_net_edge_bps", 8.0),
        )
    )
    minimum_event_score = float(
        policy.get(
            "minimum_event_score",
            cfg.get("minimum_event_score", 0.30),
        )
    )

    blockers: list[str] = []
    if not bool(qualification.get("passed", False)):
        blockers.append("MODEL_NOT_ECONOMICALLY_QUALIFIED")
    if is_event and selected_horizon not in qualified_horizons:
        blockers.append("SELECTED_DIRECTION_NOT_QUALIFIED")
    if is_event and not policy:
        blockers.append("ECONOMIC_POLICY_UNAVAILABLE")

    if not is_event or event_direction == 0:
        blockers.append("NO_NEW_STRUCTURE_BREAKOUT")
    elif event_type not in STRUCTURAL_EVENTS:
        blockers.append("UNSUPPORTED_STRUCTURE_EVENT")
    if is_event and event_score < minimum_event_score:
        blockers.append("WEAK_BREAKOUT_STRUCTURE")
    if is_event and breakout_level is None:
        blockers.append("BREAKOUT_LEVEL_UNAVAILABLE")
    if is_event and invalidation_level is None:
        blockers.append("INVALIDATION_LEVEL_UNAVAILABLE")
    if event_already_traded:
        blockers.append("EVENT_ALREADY_TRADED")
    if str(latest_row.get("regime", "UNKNOWN")) == "UNKNOWN":
        blockers.append("REGIME_UNKNOWN")

    expected_trade_direction = (
        "UP"
        if event_direction > 0
        else "DOWN"
        if event_direction < 0
        else trade_direction
    )
    if is_event and trade_direction != expected_trade_direction:
        blockers.append("EVENT_DIRECTION_MISMATCH")
    if event_direction < 0 and not bool(cfg.get("allow_short", False)):
        blockers.append("SHORT_EXECUTION_VENUE_NOT_ENABLED")

    minimum_confidence = float(
        policy.get(
            "success_probability",
            _per_horizon(
                cfg.get("minimum_confidence", {}),
                selected_horizon,
                0.60,
            ),
        )
    )
    if confidence < minimum_confidence:
        blockers.append("LOW_BREAKOUT_SUCCESS_PROBABILITY")
    minimum_tradeability = float(
        policy.get(
            "tradeability_probability",
            _per_horizon(
                cfg.get("minimum_tradeability_probability", {}),
                selected_horizon,
                0.58,
            ),
        )
    )
    if tradeability_probability < minimum_tradeability:
        blockers.append("LOW_TRADEABILITY_PROBABILITY")
    if net_edge_bps < minimum_edge_bps:
        blockers.append("INSUFFICIENT_STRESS_NET_EDGE")

    atr_pct = float(latest_row.get("atr_pct", np.nan))
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        blockers.append("ATR_UNAVAILABLE")
    elif atr_pct * 100 > float(cfg.get("maximum_atr_percent", 4.0)):
        blockers.append("VOLATILITY_SHOCK")
    if int(latest_row.get("news_shock", 0)) == 1 and bool(
        cfg.get("block_during_news_shock", True)
    ):
        blockers.append("NEWS_SHOCK_BLOCK")
    if recent_signal_count >= int(cfg.get("maximum_daily_signals", 3)):
        blockers.append("DAILY_SIGNAL_LIMIT")
    cooldown = float(cfg.get("cooldown_hours_after_signal", 3))
    if (
        hours_since_last_signal is not None
        and hours_since_last_signal < cooldown
    ):
        blockers.append("SIGNAL_COOLDOWN")

    if data_health:
        if not data_health.get("candles_ok", True):
            blockers.append("CANDLE_DATA_UNHEALTHY")
        if not data_health.get("quote_ok", True):
            blockers.append("QUOTE_STALE")
        if data_health.get("provider_mismatch", False):
            blockers.append("PROVIDER_MISMATCH")
        if data_health.get("model_stale", False):
            blockers.append("MODEL_STALE")
        if data_health.get("news_stale", False) and bool(
            cfg.get("require_fresh_news", False)
        ):
            blockers.append("NEWS_STALE")

    action = (
        "LONG"
        if event_direction > 0
        else "SHORT"
        if event_direction < 0
        else "WAIT"
    )
    if blockers:
        action = "WAIT"
    trade_plan = build_trade_plan(
        latest_row,
        expected_trade_direction,
        prediction,
        settings,
        action,
        selected_horizon=selected_horizon,
        expected_aligned_return=event_aligned_return,
        stress_cost_bps=stress_cost_bps,
        minimum_edge_bps=minimum_edge_bps,
        calibrated_success=confidence,
        calibrated_tradeability=tradeability_probability,
    )
    return Decision(
        forecast_direction=forecast_direction,
        action=action,
        confidence=confidence,
        tradeability_probability=tradeability_probability,
        expected_return=expected_return,
        expected_net_edge_bps=float(net_edge_bps),
        selected_horizon=selected_horizon,
        probabilities={
            int(key): float(value)
            for key, value in prediction["probabilities"].items()
        },
        tradeability={
            int(key): float(value)
            for key, value in prediction["tradeability"].items()
        },
        returns={
            int(key): float(value)
            for key, value in prediction.get(
                "absolute_event_returns",
                prediction.get("returns", {}),
            ).items()
        },
        blockers=blockers,
        trade_plan=trade_plan,
    )


def _select_economic_horizon(
    prediction: dict[str, Any],
    qualified_horizons: set[int],
    policies: dict[str, Any],
    event_score: float,
) -> tuple[int, float, float, dict[str, Any]]:
    fallback = int(prediction.get("selected_horizon", 1))
    candidates = sorted(qualified_horizons) or [fallback]
    best: tuple[float, int, float, float, dict[str, Any]] | None = None
    for horizon in candidates:
        policy = policies.get(str(horizon), policies.get(horizon, {}))
        policy = policy if isinstance(policy, dict) else {}
        raw_success = _mapping_value(
            prediction.get("continuation", {}), horizon, 0.5
        )
        raw_tradeability = _mapping_value(
            prediction.get("tradeability", {}), horizon, 0.5
        )
        success = apply_calibration(
            raw_success, policy.get("success_calibration")
        )
        tradeability = apply_calibration(
            raw_tradeability,
            policy.get("tradeability_calibration"),
        )
        gross_return = _mapping_value(
            prediction.get("event_returns", {}), horizon, 0.0
        )
        edge_floor = float(
            policy.get("minimum_predicted_stress_edge_bps", 0.0)
        )
        score = (
            success
            * tradeability
            * max(gross_return * 10_000.0 - edge_floor, 0.0)
            * (0.5 + max(event_score, 0.0))
        )
        candidate = (score, horizon, success, tradeability, policy)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return int(best[1]), float(best[2]), float(best[3]), dict(best[4])


def build_trade_plan(
    row: pd.Series,
    direction: str,
    prediction: dict[str, Any],
    settings: Settings,
    action: str,
    *,
    selected_horizon: int | None = None,
    expected_aligned_return: float | None = None,
    stress_cost_bps: float | None = None,
    minimum_edge_bps: float | None = None,
    calibrated_success: float | None = None,
    calibrated_tradeability: float | None = None,
) -> dict[str, Any]:
    cfg = settings.section("strategy")
    price = float(row["close"])
    atr = float(
        row.get(
            "atr",
            price * float(cfg.get("minimum_stop_percent", 0.0035)),
        )
    )
    invalidation = _finite(row.get("breakout_invalidation_level"))
    generic_stop_pct = float(cfg.get("stop_atr_multiplier", 1.10)) * atr / price
    if direction == "UP" and invalidation is not None and invalidation < price:
        structural_stop_pct = (price - invalidation) / price
    elif direction == "DOWN" and invalidation is not None and invalidation > price:
        structural_stop_pct = (invalidation - price) / price
    else:
        structural_stop_pct = generic_stop_pct
    stop_pct = float(
        np.clip(
            max(structural_stop_pct, generic_stop_pct * 0.65),
            float(cfg.get("minimum_stop_percent", 0.0035)),
            float(cfg.get("maximum_stop_percent", 0.018)),
        )
    )
    selected_horizon = int(
        selected_horizon
        if selected_horizon is not None
        else prediction["selected_horizon"]
    )
    predicted_move = abs(
        float(
            expected_aligned_return
            if expected_aligned_return is not None
            else prediction.get("expected_event_aligned_return", 0.0)
        )
    )
    target_pct = max(
        float(cfg.get("target_r_multiple", 1.80)) * stop_pct,
        predicted_move,
    )
    if direction == "UP":
        stop_price = price * (1 - stop_pct)
        target_price = price * (1 + target_pct)
    else:
        stop_price = price * (1 + stop_pct)
        target_price = price * (1 - target_pct)

    account = float(cfg.get("account_equity_usd", 1000.0))
    risk_budget = account * float(cfg.get("risk_per_trade_fraction", 0.0020))
    costs = execution_cost_breakdown(cfg)
    effective_stress_bps = float(
        stress_cost_bps
        if stress_cost_bps is not None
        else costs["stress_cost_bps"]
    )
    cost_buffer = effective_stress_bps / 10_000.0
    gap_buffer = float(cfg.get("gap_risk_buffer_bps", 6.0)) / 10_000.0
    effective_risk_pct = stop_pct + cost_buffer + gap_buffer
    quantity_btc = risk_budget / max(price * effective_risk_pct, 1e-9)
    notional = quantity_btc * price
    leverage = min(
        float(cfg.get("maximum_leverage", 3.0)),
        max(1.0, notional / max(account, 1e-9)),
    )
    predicted_gross_bps = predicted_move * 10_000.0
    predicted_net_bps = predicted_gross_bps - effective_stress_bps
    return {
        "status": "ACTIONABLE" if action in {"LONG", "SHORT"} else "BLOCKED",
        "event_id": row.get("event_id"),
        "event_type": row.get("event_type", "NONE"),
        "event_score": float(row.get("event_score", 0.0)),
        "event_scale_hours": int(row.get("event_scale_hours", 0)),
        "breakout_source": row.get("breakout_source", "NONE"),
        "breakout_level": row.get("breakout_level"),
        "invalidation_level": row.get("breakout_invalidation_level"),
        "triangle_type": row.get("triangle_type", "NONE"),
        "regime": row.get("regime", "UNKNOWN"),
        "trade_direction_source": "ECONOMIC_DIRECTIONAL_BREAKOUT",
        "entry_reference": price,
        "entry_definition": "OPEN_OF_NEXT_1H_CANDLE",
        "entry_style": str(cfg.get("entry_order_style", "maker")).upper(),
        "exit_style": str(cfg.get("exit_order_style", "taker")).upper(),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "stop_percent": stop_pct,
        "target_percent": target_pct,
        "risk_reward": float(target_pct / stop_pct),
        "risk_budget_usd": risk_budget,
        "quantity_btc": float(quantity_btc),
        "notional_usd": float(notional),
        "suggested_leverage": float(leverage),
        "maximum_holding_hours": selected_horizon,
        "base_execution_cost_bps": costs["base_cost_bps"],
        "stress_execution_cost_bps": effective_stress_bps,
        "minimum_required_net_edge_bps": minimum_edge_bps,
        "predicted_gross_move_bps": predicted_gross_bps,
        "predicted_stress_net_edge_bps": predicted_net_bps,
        "calibrated_success_probability": calibrated_success,
        "calibrated_tradeability_probability": calibrated_tradeability,
        "paper_only": bool(cfg.get("paper_only", True)),
    }


def _mapping_value(
    value: Any,
    horizon: int,
    default: float,
) -> float:
    if not isinstance(value, dict):
        return float(default)
    return float(value.get(horizon, value.get(str(horizon), default)))


def _per_horizon(value: Any, horizon: int, default: float) -> float:
    if isinstance(value, dict):
        return float(value.get(horizon, value.get(str(horizon), default)))
    return default if value is None else float(value)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
