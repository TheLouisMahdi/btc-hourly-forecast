from __future__ import annotations

from typing import Any

import numpy as np

from .costs import runtime_cost_breakdown


def estimate_isolated_liquidation_price(
    strategy: dict[str, Any],
    *,
    entry: float,
    direction: str,
    leverage: float,
    quantity: float = 1.0,
) -> float | None:
    """Estimate Binance-style USD-M isolated liquidation for configured tier.

    The estimate assumes one-way isolated margin, initial margin equal to
    entry notional / leverage, and the configured maintenance-margin tier.
    Actual liquidation is exchange/Mark-Price driven and may differ.
    """
    entry = float(entry)
    leverage = float(leverage)
    quantity = abs(float(quantity))
    if entry <= 0 or leverage <= 0 or quantity <= 0:
        return None
    if str(strategy.get("margin_mode", "ISOLATED")).upper() != "ISOLATED":
        return None

    mmr = max(0.0, float(strategy.get("maintenance_margin_rate", 0.004)))
    maintenance_amount = max(
        0.0,
        float(strategy.get("maintenance_amount_usd", 0.0)),
    )
    maintenance_per_unit = maintenance_amount / quantity
    side = str(direction).upper()
    if side in {"LONG", "UP"}:
        denominator = 1.0 - mmr
        if denominator <= 0:
            return None
        price = (
            entry * (1.0 - 1.0 / leverage) - maintenance_per_unit
        ) / denominator
    elif side in {"SHORT", "DOWN"}:
        denominator = 1.0 + mmr
        price = (
            entry * (1.0 + 1.0 / leverage) + maintenance_per_unit
        ) / denominator
    else:
        return None
    return float(price) if np.isfinite(price) and price > 0 else None


def liquidation_distance_fraction(
    strategy: dict[str, Any],
    *,
    entry: float,
    direction: str,
    leverage: float,
) -> float | None:
    price = estimate_isolated_liquidation_price(
        strategy,
        entry=entry,
        direction=direction,
        leverage=leverage,
    )
    if price is None:
        return None
    return abs(float(price) - float(entry)) / max(float(entry), 1e-12)


def select_leverage(
    strategy: dict[str, Any],
    risk_score: float,
    modeled_risk_fraction: float = 0.0,
    *,
    entry: float | None = None,
    direction: str | None = None,
) -> float:
    tiers = sorted(
        {
            float(value)
            for value in strategy.get(
                "leverage_tiers",
                [10.0, 20.0, 40.0],
            )
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

    if modeled_risk_fraction <= 0:
        return float(candidate)

    safety = float(
        np.clip(
            strategy.get("leverage_liquidation_safety_factor", 0.70),
            0.05,
            0.95,
        )
    )
    safe: list[float] = []
    for tier in tiers:
        if tier > candidate:
            continue
        distance: float | None = None
        if entry is not None and direction is not None:
            distance = liquidation_distance_fraction(
                strategy,
                entry=float(entry),
                direction=str(direction),
                leverage=tier,
            )
        if distance is None:
            distance = 1.0 / tier
        if modeled_risk_fraction <= safety * distance:
            safe.append(tier)
    return float(safe[-1] if safe else tiers[0])


def apply_risk_scaled_economics(
    plan: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    """Recalculate final paper position economics from one risk budget."""
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
    account = float(
        output.get(
            "paper_account_equity_usd",
            strategy.get("account_equity_usd", 1000.0),
        )
    )
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

    holding_hours = float(output.get("maximum_holding_hours", 0.0))
    runtime_costs = runtime_cost_breakdown(strategy, holding_hours)
    configured_stress = float(output.get("stress_execution_cost_bps", 0.0))
    stress_bps = max(
        configured_stress,
        float(runtime_costs["runtime_stress_cost_bps"]),
    )
    base_cost_bps = float(runtime_costs["runtime_base_cost_bps"])
    gap_bps = float(strategy.get("gap_risk_buffer_bps", 0.0))
    cost_fraction = max(0.0, stress_bps) / 10_000.0
    gap_fraction = max(0.0, gap_bps) / 10_000.0
    modeled_risk_fraction = stop_pct + cost_fraction + gap_fraction
    direction = str(
        output.get("position_direction")
        or output.get("direction")
        or "NONE"
    )

    leverage = select_leverage(
        strategy,
        float(output.get("risk_score", 0.0)),
        modeled_risk_fraction,
        entry=entry,
        direction=direction,
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
    heuristic_expected_value = (
        p_target * target_net
        + p_stop * stop_net
        + p_expiry * expiry_net
    )
    minimum_samples = int(
        settings.section("trade_lifecycle").get(
            "minimum_online_samples",
            20,
        )
    )
    samples_seen = int(output.get("adaptive_samples_seen", 0) or 0)
    calibrated = samples_seen >= minimum_samples
    expected_value = heuristic_expected_value if calibrated else None

    liquidation_price = estimate_isolated_liquidation_price(
        strategy,
        entry=entry,
        direction=direction,
        leverage=leverage,
        quantity=max(quantity, 1e-12),
    )
    liquidation_distance = (
        None
        if liquidation_price is None
        else abs(liquidation_price - entry) / entry
    )
    stop_to_liquidation_ratio = (
        None
        if not liquidation_distance
        else modeled_risk_fraction / liquidation_distance
    )
    first_tier_max = float(
        strategy.get("first_tier_max_notional_usd", 300_000.0)
    )
    liquidation_status = (
        "TIER1_ISOLATED_ESTIMATE"
        if liquidation_price is not None and notional <= first_tier_max
        else "ESTIMATE_UNAVAILABLE"
    )

    output.update(
        {
            "paper_account_equity_usd": float(account),
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
            "base_execution_cost_bps": float(runtime_costs["base_cost_bps"]),
            "stress_execution_cost_bps": float(stress_bps),
            "projected_funding_bps": float(
                runtime_costs["projected_funding_bps"]
            ),
            "round_trip_stress_cost_usd": float(execution_cost),
            "target_gross_profit_usd": float(target_gross),
            "target_net_profit_usd": float(target_net),
            "stop_gross_loss_usd": float(-stop_gross),
            "stop_net_loss_usd": float(stop_net),
            "profit_margin_usd": float(target_net),
            "target_margin_roi": float(target_net / max(margin, 1e-12)),
            "stop_margin_roi": float(stop_net / max(margin, 1e-12)),
            "heuristic_expected_value_usd": float(
                heuristic_expected_value
            ),
            "expected_value_usd": (
                None if expected_value is None else float(expected_value)
            ),
            "expected_value_status": (
                "CALIBRATED"
                if calibrated
                else "UNCALIBRATED_HEURISTIC"
            ),
            "maintenance_margin_rate": float(
                strategy.get("maintenance_margin_rate", 0.004)
            ),
            "estimated_liquidation_price": liquidation_price,
            "liquidation_distance_percent": (
                None
                if liquidation_distance is None
                else float(liquidation_distance)
            ),
            "stop_to_liquidation_ratio": (
                None
                if stop_to_liquidation_ratio is None
                else float(stop_to_liquidation_ratio)
            ),
            "liquidation_estimate_status": liquidation_status,
            "margin_mode": str(
                strategy.get("margin_mode", "ISOLATED")
            ).upper(),
        }
    )
    return output
