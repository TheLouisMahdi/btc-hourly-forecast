from __future__ import annotations

from typing import Any

import numpy as np


def select_leverage(
    strategy: dict[str, Any],
    risk_score: float,
    modeled_risk_fraction: float = 0.0,
) -> float:
    tiers = sorted(
        {
            float(value)
            for value in strategy.get("leverage_tiers", [10.0, 20.0, 40.0])
            if float(value) > 0
        }
    )
    maximum = max(1.0, float(strategy.get("maximum_leverage", 40.0)))
    tiers = [tier for tier in tiers if tier <= maximum]
    if not tiers:
        return maximum

    thresholds = strategy.get(
        "leverage_risk_score_thresholds",
        [0.35, 0.65],
    )
    low = float(thresholds[0]) if len(thresholds) > 0 else 0.35
    high = float(thresholds[1]) if len(thresholds) > 1 else 0.65
    score = float(np.clip(risk_score, 0.0, 1.0))

    if score < low:
        candidate = tiers[0]
    elif score < high:
        candidate = tiers[min(1, len(tiers) - 1)]
    else:
        candidate = tiers[-1]

    if modeled_risk_fraction > 0:
        safety = float(
            strategy.get("leverage_liquidation_safety_factor", 0.70)
        )
        maintenance_margin_rate = float(
            strategy.get("maintenance_margin_rate", 0.004)
        )
        safe = [
            tier
            for tier in tiers
            if tier <= candidate
            and modeled_risk_fraction
            <= safety
            * _minimum_liquidation_distance_fraction(
                tier,
                maintenance_margin_rate,
            )
        ]
        if safe:
            candidate = safe[-1]
        else:
            candidate = tiers[0]

    return float(candidate)


def apply_risk_scaled_economics(
    plan: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    """Recalculate final position economics from one canonical risk budget.

    Position size is bounded by stop distance, stress execution cost and the
    configured gap-risk buffer. The gap buffer affects sizing but is not
    reported as a realized execution fee.
    """
    output = dict(plan)
    try:
        entry = float(output["entry_reference"])
        stop_pct = abs(float(output["stop_percent"]))
        target_pct = abs(float(output["target_percent"]))
    except (KeyError, TypeError, ValueError):
        return output
    if entry <= 0 or stop_pct <= 0 or target_pct <= 0:
        return output

    strategy = settings.section("strategy")
    account = float(strategy.get("account_equity_usd", 1000.0))
    minimum_fraction = float(
        strategy.get("minimum_risk_per_trade_fraction", 0.005)
    )
    maximum_fraction = float(
        strategy.get("maximum_risk_per_trade_fraction", 0.03)
    )
    if maximum_fraction < minimum_fraction:
        minimum_fraction, maximum_fraction = (
            maximum_fraction,
            minimum_fraction,
        )
    risk_fraction = float(
        np.clip(
            float(
                output.get(
                    "risk_fraction",
                    strategy.get("risk_per_trade_fraction", 0.0125),
                )
            ),
            minimum_fraction,
            maximum_fraction,
        )
    )
    risk_budget = account * risk_fraction

    stress_bps = float(output.get("stress_execution_cost_bps", 0.0))
    gap_bps = float(strategy.get("gap_risk_buffer_bps", 0.0))
    cost_fraction = max(0.0, stress_bps) / 10_000.0
    gap_fraction = max(0.0, gap_bps) / 10_000.0
    modeled_risk_fraction = stop_pct + cost_fraction + gap_fraction

    leverage = select_leverage(
        strategy,
        float(output.get("risk_score", 0.0)),
        modeled_risk_fraction,
    )
    quantity = risk_budget / max(entry * modeled_risk_fraction, 1e-12)
    notional = min(quantity * entry, account * leverage)
    quantity = notional / entry
    margin = notional / leverage

    execution_cost = notional * cost_fraction
    modeled_total_risk = notional * modeled_risk_fraction
    target_gross = notional * target_pct
    stop_gross = notional * stop_pct
    target_net = target_gross - execution_cost
    stop_net = -(stop_gross + execution_cost)

    predicted_r = float(output.get("adaptive_predicted_r", 0.0))
    p_target = float(output.get("adaptive_target_probability", 0.5))
    p_stop = float(output.get("adaptive_stop_probability", 0.4))
    p_expiry = max(0.0, 1.0 - p_target - p_stop)
    expiry_net = notional * predicted_r * stop_pct - execution_cost
    expected_value = (
        p_target * target_net
        + p_stop * stop_net
        + p_expiry * expiry_net
    )

    direction = _position_direction(output, entry)
    maintenance_margin_rate = float(
        strategy.get("maintenance_margin_rate", 0.004)
    )
    liquidation_price = estimated_isolated_liquidation_price(
        entry,
        leverage,
        maintenance_margin_rate,
        direction,
    )
    liquidation_distance = (
        abs(entry - liquidation_price) / entry
        if liquidation_price is not None
        else None
    )
    stop_price = _positive_or_none(output.get("stop_price"))
    stop_to_liquidation_buffer = None
    if liquidation_price is not None and stop_price is not None:
        if direction == "LONG":
            stop_to_liquidation_buffer = (stop_price - liquidation_price) / entry
        elif direction == "SHORT":
            stop_to_liquidation_buffer = (liquidation_price - stop_price) / entry

    safety_factor = float(
        strategy.get("leverage_liquidation_safety_factor", 0.70)
    )
    liquidation_safety_ok = (
        liquidation_distance is not None
        and modeled_risk_fraction <= liquidation_distance * safety_factor
    )

    output.update(
        {
            "risk_fraction": risk_fraction,
            "risk_budget_usd": float(risk_budget),
            "modeled_risk_fraction": float(modeled_risk_fraction),
            "modeled_total_risk_usd": float(modeled_total_risk),
            "risk_budget_utilization": float(
                modeled_total_risk / max(risk_budget, 1e-12)
            ),
            "gap_risk_buffer_bps": float(gap_bps),
            "quantity_btc": float(quantity),
            "notional_usd": float(notional),
            "suggested_leverage": leverage,
            "margin_required_usd": float(margin),
            "round_trip_stress_cost_usd": float(execution_cost),
            "target_gross_profit_usd": float(target_gross),
            "target_net_profit_usd": float(target_net),
            "stop_gross_loss_usd": float(-stop_gross),
            "stop_net_loss_usd": float(stop_net),
            "profit_margin_usd": float(target_net),
            "target_margin_roi": float(target_net / max(margin, 1e-12)),
            "stop_margin_roi": float(stop_net / max(margin, 1e-12)),
            "expected_value_usd": float(expected_value),
            "execution_venue": str(
                strategy.get("execution_venue", "BINANCE_USDM_BTCUSDT")
            ),
            "margin_mode": str(strategy.get("margin_mode", "isolated")).upper(),
            "maintenance_margin_rate": float(maintenance_margin_rate),
            "estimated_liquidation_price": liquidation_price,
            "liquidation_distance_percent": liquidation_distance,
            "stop_to_liquidation_buffer_percent": stop_to_liquidation_buffer,
            "liquidation_safety_ok": bool(liquidation_safety_ok),
            "liquidation_model": (
                "ISOLATED_LINEAR_FIRST_MAINTENANCE_BRACKET_APPROX"
                if liquidation_price is not None
                else "UNAVAILABLE"
            ),
        }
    )
    return output


def estimated_isolated_liquidation_price(
    entry: float,
    leverage: float,
    maintenance_margin_rate: float,
    direction: str | None,
) -> float | None:
    """Approximate linear isolated-margin liquidation before liquidation fees.

    This matches the first maintenance-margin bracket model where maintenance
    amount is zero. Exchange liquidation is ultimately driven by mark price.
    """
    if entry <= 0 or leverage <= 0:
        return None
    mmr = float(np.clip(maintenance_margin_rate, 0.0, 0.99))
    side = str(direction or "").upper()
    if side == "LONG":
        denominator = 1.0 - mmr
        return float(entry * (1.0 - 1.0 / leverage) / denominator)
    if side == "SHORT":
        denominator = 1.0 + mmr
        return float(entry * (1.0 + 1.0 / leverage) / denominator)
    return None


def _minimum_liquidation_distance_fraction(
    leverage: float,
    maintenance_margin_rate: float,
) -> float:
    long_price = estimated_isolated_liquidation_price(
        1.0,
        leverage,
        maintenance_margin_rate,
        "LONG",
    )
    short_price = estimated_isolated_liquidation_price(
        1.0,
        leverage,
        maintenance_margin_rate,
        "SHORT",
    )
    distances = [
        abs(1.0 - value)
        for value in (long_price, short_price)
        if value is not None and value > 0
    ]
    return min(distances) if distances else 1.0 / max(leverage, 1.0)


def _position_direction(
    plan: dict[str, Any],
    entry: float,
) -> str | None:
    direction = str(
        plan.get("direction")
        or plan.get("position_side")
        or ""
    ).upper()
    if direction in {"LONG", "SHORT"}:
        return direction
    target = _positive_or_none(plan.get("target_price"))
    if target is None:
        return None
    return "LONG" if target > entry else "SHORT"


def _positive_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
