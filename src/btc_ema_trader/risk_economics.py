from __future__ import annotations

from typing import Any

import numpy as np


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

    maximum_leverage = float(strategy.get("maximum_leverage", 5.0))
    quantity = risk_budget / max(entry * modeled_risk_fraction, 1e-12)
    notional = min(quantity * entry, account * maximum_leverage)
    quantity = notional / entry
    leverage = float(
        np.clip(
            notional / max(account, 1e-12),
            1.0,
            maximum_leverage,
        )
    )
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
        }
    )
    return output
