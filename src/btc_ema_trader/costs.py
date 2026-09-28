from __future__ import annotations

import math
from typing import Any


def execution_cost_breakdown(strategy_cfg: dict[str, Any]) -> dict[str, float]:
    """Return the configured round-trip trading costs in basis points."""
    entry_style = str(strategy_cfg.get("entry_order_style", "maker")).lower()
    exit_style = str(strategy_cfg.get("exit_order_style", "taker")).lower()
    maker = float(strategy_cfg.get("maker_fee_bps", 2.0))
    taker = float(strategy_cfg.get("taker_fee_bps", 5.0))
    entry_fee = maker if entry_style == "maker" else taker
    exit_fee = maker if exit_style == "maker" else taker
    entry_slippage = float(strategy_cfg.get("entry_slippage_bps", 1.5))
    exit_slippage = float(strategy_cfg.get("exit_slippage_bps", 2.5))
    static_funding = float(strategy_cfg.get("funding_buffer_bps", 0.0))
    base = (
        entry_fee
        + exit_fee
        + entry_slippage
        + exit_slippage
        + static_funding
    )
    stress_multiplier = float(strategy_cfg.get("stress_cost_multiplier", 1.5))
    return {
        "entry_fee_bps": entry_fee,
        "exit_fee_bps": exit_fee,
        "entry_slippage_bps": entry_slippage,
        "exit_slippage_bps": exit_slippage,
        "funding_buffer_bps": static_funding,
        "base_cost_bps": base,
        "stress_cost_bps": base * stress_multiplier,
        "profit_buffer_bps": float(
            strategy_cfg.get("minimum_profit_buffer_bps", 8.0)
        ),
    }


def projected_funding_bps(
    strategy_cfg: dict[str, Any],
    holding_hours: float,
) -> float:
    """Return a conservative funding-cost allowance for a paper position.

    The configured rate is a cost buffer, not a forecast of the next funding
    payment. It intentionally assumes that every possible funding timestamp
    during the holding window is adverse to the position.
    """
    hours = max(0.0, float(holding_hours))
    interval = max(
        1.0,
        float(strategy_cfg.get("funding_interval_hours", 8.0)),
    )
    rate = max(
        0.0,
        float(
            strategy_cfg.get(
                "funding_rate_buffer_bps_per_interval",
                0.0,
            )
        ),
    )
    if hours <= 0.0 or rate <= 0.0:
        return 0.0
    intervals = int(math.ceil(hours / interval))
    return float(intervals * rate)


def runtime_cost_breakdown(
    strategy_cfg: dict[str, Any],
    holding_hours: float,
) -> dict[str, float]:
    """Add a holding-period funding allowance to execution costs."""
    base = execution_cost_breakdown(strategy_cfg)
    projected_funding = projected_funding_bps(
        strategy_cfg,
        holding_hours,
    )
    return {
        **base,
        "projected_funding_bps": projected_funding,
        "runtime_base_cost_bps": (
            float(base["base_cost_bps"]) + projected_funding
        ),
        "runtime_stress_cost_bps": (
            float(base["stress_cost_bps"]) + projected_funding
        ),
    }
