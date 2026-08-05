from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .costs import execution_cost_breakdown
from .economic_validation import apply_calibration
from .model import HourlyModelBundle

POLICY_NAME = "AGGRESSIVE_STRUCTURAL_RISK_SCALED"
POLICY_VERSION = 2
RISK_CONTRACT_VERSION = 2

STRUCTURAL_EVENTS = {
    "RESISTANCE_BREAKOUT_LONG",
    "TRIANGLE_BREAKOUT_LONG",
    "SUPPORT_BREAKDOWN_SHORT",
    "TRIANGLE_BREAKDOWN_SHORT",
}

SOFT_RISK_FLAGS = {
    "MODEL_NOT_QUALIFIED",
    "SELECTED_DIRECTION_NOT_QUALIFIED",
    "ECONOMIC_POLICY_UNAVAILABLE",
    "WEAK_BREAKOUT_STRUCTURE",
    "LOW_BREAKOUT_SUCCESS_PROBABILITY",
    "LOW_TRADEABILITY_PROBABILITY",
    "INSUFFICIENT_STRESS_NET_EDGE",
    "VOLATILITY_SHOCK",
    "NEWS_SHOCK",
    "DAILY_SIGNAL_LIMIT",
    "SIGNAL_COOLDOWN",
    "MODEL_STALE",
    "NEWS_STALE",
    "REGIME_UNKNOWN",
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
            cfg.get("minimum_net_edge_bps", 0.0),
        )
    )
    minimum_event_score = float(
        policy.get(
            "minimum_event_score",
            cfg.get("minimum_event_score", 0.10),
        )
    )

    hard_blockers: list[str] = []
    soft_risk_flags: list[str] = []

    qualification_passed = bool(qualification.get("passed", False))
    direction_qualified = selected_horizon in qualified_horizons
    if not qualification_passed:
        soft_risk_flags.append("MODEL_NOT_QUALIFIED")
    if is_event and not direction_qualified:
        soft_risk_flags.append("SELECTED_DIRECTION_NOT_QUALIFIED")
    if is_event and not policy:
        soft_risk_flags.append("ECONOMIC_POLICY_UNAVAILABLE")

    if not is_event or event_direction == 0:
        hard_blockers.append("NO_NEW_STRUCTURE_BREAKOUT")
    elif event_type not in STRUCTURAL_EVENTS:
        hard_blockers.append("UNSUPPORTED_STRUCTURE_EVENT")
    if is_event and event_score < minimum_event_score:
        soft_risk_flags.append("WEAK_BREAKOUT_STRUCTURE")
    if is_event and breakout_level is None:
        hard_blockers.append("BREAKOUT_LEVEL_UNAVAILABLE")
    if is_event and invalidation_level is None:
        hard_blockers.append("INVALIDATION_LEVEL_UNAVAILABLE")
    if event_already_traded:
        hard_blockers.append("EVENT_ALREADY_TRADED")
    if str(latest_row.get("regime", "UNKNOWN")) == "UNKNOWN":
        soft_risk_flags.append("REGIME_UNKNOWN")

    expected_trade_direction = (
        "UP"
        if event_direction > 0
        else "DOWN"
        if event_direction < 0
        else trade_direction
    )
    if is_event and trade_direction != expected_trade_direction:
        hard_blockers.append("EVENT_DIRECTION_MISMATCH")
    if event_direction < 0 and not bool(cfg.get("allow_short", False)):
        hard_blockers.append("SHORT_EXECUTION_VENUE_NOT_ENABLED")

    minimum_confidence = float(
        policy.get(
            "success_probability",
            _per_horizon(
                cfg.get("minimum_confidence", {}),
                selected_horizon,
                0.50,
            ),
        )
    )
    if confidence < minimum_confidence:
        soft_risk_flags.append("LOW_BREAKOUT_SUCCESS_PROBABILITY")
    minimum_tradeability = float(
        policy.get(
            "tradeability_probability",
            _per_horizon(
                cfg.get("minimum_tradeability_probability", {}),
                selected_horizon,
                0.50,
            ),
        )
    )
    if tradeability_probability < minimum_tradeability:
        soft_risk_flags.append("LOW_TRADEABILITY_PROBABILITY")
    if net_edge_bps < minimum_edge_bps:
        soft_risk_flags.append("INSUFFICIENT_STRESS_NET_EDGE")

    atr_pct = float(latest_row.get("atr_pct", np.nan))
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        hard_blockers.append("ATR_UNAVAILABLE")
    elif atr_pct * 100 > float(cfg.get("maximum_atr_percent", 6.0)):
        soft_risk_flags.append("VOLATILITY_SHOCK")
    if int(latest_row.get("news_shock", 0)) == 1:
        soft_risk_flags.append("NEWS_SHOCK")
    if recent_signal_count >= int(cfg.get("maximum_daily_signals", 12)):
        soft_risk_flags.append("DAILY_SIGNAL_LIMIT")
    cooldown = float(cfg.get("cooldown_hours_after_signal", 0))
    if (
        hours_since_last_signal is not None
        and hours_since_last_signal < cooldown
    ):
        soft_risk_flags.append("SIGNAL_COOLDOWN")

    if data_health:
        if not data_health.get("candles_ok", True):
            hard_blockers.append("CANDLE_DATA_UNHEALTHY")
        if not data_health.get("quote_ok", True):
            hard_blockers.append("QUOTE_STALE")
        if data_health.get("provider_mismatch", False):
            hard_blockers.append("PROVIDER_MISMATCH")
        if data_health.get("model_stale", False):
            soft_risk_flags.append("MODEL_STALE")
        if data_health.get("news_stale", False):
            soft_risk_flags.append("NEWS_STALE")

    hard_blockers = list(dict.fromkeys(hard_blockers))
    soft_risk_flags = [
        item
        for item in dict.fromkeys(soft_risk_flags)
        if item in SOFT_RISK_FLAGS
    ]

    risk_assessment = _risk_assessment(
        cfg=cfg,
        event_score=event_score,
        confidence=confidence,
        tradeability_probability=tradeability_probability,
        net_edge_bps=net_edge_bps,
        minimum_edge_bps=minimum_edge_bps,
        qualification_passed=qualification_passed,
        direction_qualified=direction_qualified,
        economic_policy_available=bool(policy),
        regime=str(latest_row.get("regime", "UNKNOWN")),
        event_direction=event_direction,
        soft_risk_flags=soft_risk_flags,
    )

    action = (
        "LONG"
        if event_direction > 0
        else "SHORT"
        if event_direction < 0
        else "WAIT"
    )
    if hard_blockers:
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
        risk_assessment=risk_assessment,
    )
    trade_plan.update(
        {
            "decision_mode": POLICY_NAME,
            "policy_name": POLICY_NAME,
            "policy_version": POLICY_VERSION,
            "risk_contract_version": RISK_CONTRACT_VERSION,
            "entry_contract": "STRUCTURAL_EVENT_RISK_SCALED",
            "soft_risk_flags": soft_risk_flags,
            "ignored_soft_blockers": soft_risk_flags,
            "hard_blockers": hard_blockers,
            "qualification_passed": qualification_passed,
            "direction_qualified": direction_qualified,
            "economic_policy_available": bool(policy),
        }
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
        blockers=hard_blockers,
        trade_plan=trade_plan,
    )


def _select_economic_horizon(
    prediction: dict[str, Any],
    qualified_horizons: set[int],
    policies: dict[str, Any],
    event_score: float,
) -> tuple[int, float, float, dict[str, Any]]:
    fallback = int(prediction.get("selected_horizon", 1))
    available: set[int] = set()
    for key in ("continuation", "tradeability", "event_returns"):
        value = prediction.get(key, {})
        if isinstance(value, dict):
            for horizon in value:
                try:
                    available.add(int(horizon))
                except (TypeError, ValueError):
                    continue
    structural_horizons = sorted(h for h in available if h > 1)
    candidates = (
        sorted(qualified_horizons)
        or structural_horizons
        or sorted(available)
        or [fallback]
    )
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
            raw_tradeability, policy.get("tradeability_calibration")
        )
        gross_return = _mapping_value(
            prediction.get("event_returns", {}), horizon, 0.0
        )
        edge_floor = float(
            policy.get("minimum_predicted_stress_edge_bps", 0.0)
        )
        relative_edge_bps = gross_return * 10_000.0 - edge_floor
        economic_weight = float(
            np.clip(0.75 + relative_edge_bps / 80.0, 0.25, 1.50)
        )
        score = (
            success
            * tradeability
            * (0.5 + max(event_score, 0.0))
            * economic_weight
        )
        candidate = (score, horizon, success, tradeability, policy)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return int(best[1]), float(best[2]), float(best[3]), dict(best[4])


def _risk_assessment(
    *,
    cfg: dict[str, Any],
    event_score: float,
    confidence: float,
    tradeability_probability: float,
    net_edge_bps: float,
    minimum_edge_bps: float,
    qualification_passed: bool,
    direction_qualified: bool,
    economic_policy_available: bool,
    regime: str,
    event_direction: int,
    soft_risk_flags: list[str],
) -> dict[str, Any]:
    base_fraction = float(cfg.get("risk_per_trade_fraction", 0.0125))
    minimum_fraction = float(
        cfg.get(
            "minimum_risk_per_trade_fraction",
            max(0.001, base_fraction * 0.40),
        )
    )
    maximum_fraction = float(
        cfg.get(
            "maximum_risk_per_trade_fraction",
            max(minimum_fraction, base_fraction * 2.40),
        )
    )
    if maximum_fraction < minimum_fraction:
        minimum_fraction, maximum_fraction = (
            maximum_fraction,
            minimum_fraction,
        )

    event_component = float(np.clip(event_score, 0.0, 1.0))
    confidence_component = float(
        np.clip((confidence - 0.48) / 0.24, 0.0, 1.0)
    )
    tradeability_component = float(
        np.clip((tradeability_probability - 0.45) / 0.30, 0.0, 1.0)
    )
    qualification_component = float(
        0.35
        + 0.25 * float(qualification_passed)
        + 0.25 * float(direction_qualified)
        + 0.15 * float(economic_policy_available)
    )
    edge_lower = float(cfg.get("risk_edge_lower_bps", -20.0))
    edge_upper = float(cfg.get("risk_edge_upper_bps", 20.0))
    if edge_upper <= edge_lower:
        edge_upper = edge_lower + 1.0
    edge_reference = net_edge_bps - minimum_edge_bps
    edge_component = float(
        np.clip(
            (edge_reference - edge_lower) / (edge_upper - edge_lower),
            0.0,
            1.0,
        )
    )
    regime_component = _regime_alignment_score(regime, event_direction)

    components = {
        "event_quality": event_component,
        "confidence": confidence_component,
        "tradeability": tradeability_component,
        "qualification": qualification_component,
        "economic_edge": edge_component,
        "regime_alignment": regime_component,
    }
    risk_score = float(
        0.30 * event_component
        + 0.18 * confidence_component
        + 0.16 * tradeability_component
        + 0.14 * qualification_component
        + 0.14 * edge_component
        + 0.08 * regime_component
    )

    penalties = {
        "WEAK_BREAKOUT_STRUCTURE": 0.92,
        "LOW_BREAKOUT_SUCCESS_PROBABILITY": 0.95,
        "LOW_TRADEABILITY_PROBABILITY": 0.95,
        "VOLATILITY_SHOCK": 0.88,
        "NEWS_SHOCK": 0.92,
        "DAILY_SIGNAL_LIMIT": 0.90,
        "SIGNAL_COOLDOWN": 0.92,
        "MODEL_STALE": 0.85,
        "NEWS_STALE": 0.96,
        "REGIME_UNKNOWN": 0.92,
    }
    penalty_multiplier = 1.0
    applied_penalties: dict[str, float] = {}
    for flag in soft_risk_flags:
        factor = penalties.get(flag)
        if factor is None:
            continue
        penalty_multiplier *= factor
        applied_penalties[flag] = factor
    risk_score = float(np.clip(risk_score * penalty_multiplier, 0.0, 1.0))
    risk_fraction = float(
        np.clip(
            minimum_fraction
            + risk_score * (maximum_fraction - minimum_fraction),
            minimum_fraction,
            maximum_fraction,
        )
    )
    return {
        "risk_score": risk_score,
        "risk_fraction": risk_fraction,
        "minimum_risk_fraction": minimum_fraction,
        "maximum_risk_fraction": maximum_fraction,
        "components": components,
        "penalty_multiplier": float(penalty_multiplier),
        "applied_penalties": applied_penalties,
        "edge_reference_bps": float(edge_reference),
    }


def _regime_alignment_score(regime: str, event_direction: int) -> float:
    name = str(regime or "UNKNOWN").upper()
    if name == "UNKNOWN":
        return 0.50
    bullish = any(token in name for token in ("UP", "BULL", "LONG"))
    bearish = any(token in name for token in ("DOWN", "BEAR", "SHORT"))
    if event_direction > 0:
        if bullish:
            return 1.0
        if bearish:
            return 0.25
    elif event_direction < 0:
        if bearish:
            return 1.0
        if bullish:
            return 0.25
    return 0.60


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
    risk_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = settings.section("strategy")
    price = float(row["close"])
    atr = float(
        row.get(
            "atr",
            price * float(cfg.get("minimum_stop_percent", 0.0035)),
        )
    )
    selected_horizon = int(
        selected_horizon
        if selected_horizon is not None
        else prediction["selected_horizon"]
    )
    invalidation = _finite(row.get("breakout_invalidation_level"))
    fallback_stop_pct = (
        float(cfg.get("stop_atr_multiplier", 0.75))
        * atr
        / max(price, 1e-9)
    )
    if direction == "UP" and invalidation is not None and invalidation < price:
        stop_price = invalidation
        stop_pct = (price - invalidation) / price
    elif direction == "DOWN" and invalidation is not None and invalidation > price:
        stop_price = invalidation
        stop_pct = (invalidation - price) / price
    else:
        stop_pct = fallback_stop_pct
        stop_price = price * (
            1 - stop_pct if direction == "UP" else 1 + stop_pct
        )
    stop_pct = float(
        np.clip(
            stop_pct,
            float(cfg.get("minimum_stop_percent", 0.0025)),
            float(cfg.get("maximum_stop_percent", 0.025)),
        )
    )
    stop_price = price * (
        1 - stop_pct if direction == "UP" else 1 + stop_pct
    )
    base_reward_r = float(cfg.get("target_r_multiple", 5.0))
    target_pct = stop_pct * base_reward_r
    target_price = price * (
        1 + target_pct if direction == "UP" else 1 - target_pct
    )
    predicted_move = abs(
        float(
            expected_aligned_return
            if expected_aligned_return is not None
            else prediction.get("expected_event_aligned_return", 0.0)
        )
    )
    account = float(cfg.get("account_equity_usd", 1000.0))
    assessment = risk_assessment or {}
    risk_fraction = float(
        assessment.get(
            "risk_fraction",
            cfg.get("risk_per_trade_fraction", 0.0125),
        )
    )
    risk_budget = account * risk_fraction
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
    notional = min(
        quantity_btc * price,
        account * float(cfg.get("maximum_leverage", 5.0)),
    )
    quantity_btc = notional / price
    leverage = min(
        float(cfg.get("maximum_leverage", 5.0)),
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
        "regime_code": float(row.get("regime_code", 0.0)),
        "trade_direction_source": "AGGRESSIVE_STRUCTURAL_BREAKOUT",
        "entry_reference": price,
        "entry_reference_kind": "CURRENT_CLOSE_PROXY",
        "entry_definition": "PAPER_MARKET_ORDER_AT_SIGNAL_RUN",
        "entry_style": str(cfg.get("entry_order_style", "market")).upper(),
        "exit_style": str(cfg.get("exit_order_style", "taker")).upper(),
        "entry_atr": atr,
        "atr_pct": float(row.get("atr_pct", atr / max(price, 1e-9))),
        "adx": float(row.get("adx", 0.0)),
        "rsi_centered": float(row.get("rsi_centered", 0.0)),
        "volume_z_24": float(row.get("volume_z_24", 0.0)),
        "stop_price": float(stop_price),
        "target_price": float(target_price),
        "stop_percent": float(stop_pct),
        "target_percent": float(target_pct),
        "target_atr": float(target_pct * price / max(atr, 1e-9)),
        "risk_reward": float(base_reward_r),
        "label_execution_aligned": bool(invalidation is not None),
        "risk_score": float(assessment.get("risk_score", 0.0)),
        "risk_fraction": risk_fraction,
        "risk_assessment": assessment,
        "risk_budget_usd": risk_budget,
        "quantity_btc": float(quantity_btc),
        "notional_usd": float(notional),
        "suggested_leverage": float(leverage),
        "maximum_holding_hours": int(
            settings.section("trade_lifecycle").get(
                "base_maximum_holding_hours", 72
            )
        ),
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
