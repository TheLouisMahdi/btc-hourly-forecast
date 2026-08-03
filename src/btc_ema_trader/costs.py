from __future__ import annotations

from typing import Any


def execution_cost_breakdown(strategy_cfg: dict[str, Any]) -> dict[str, float]:
    """Return a single source of truth for round-trip execution costs in basis points."""
    entry_style = str(strategy_cfg.get("entry_order_style", "maker")).lower()
    exit_style = str(strategy_cfg.get("exit_order_style", "taker")).lower()
    maker = float(strategy_cfg.get("maker_fee_bps", 2.0))
    taker = float(strategy_cfg.get("taker_fee_bps", 5.0))
    entry_fee = maker if entry_style == "maker" else taker
    exit_fee = maker if exit_style == "maker" else taker
    entry_slippage = float(strategy_cfg.get("entry_slippage_bps", 1.5))
    exit_slippage = float(strategy_cfg.get("exit_slippage_bps", 2.5))
    funding_buffer = float(strategy_cfg.get("funding_buffer_bps", 0.0))
    base = entry_fee + exit_fee + entry_slippage + exit_slippage + funding_buffer
    stress_multiplier = float(strategy_cfg.get("stress_cost_multiplier", 1.5))
    return {
        "entry_fee_bps": entry_fee,
        "exit_fee_bps": exit_fee,
        "entry_slippage_bps": entry_slippage,
        "exit_slippage_bps": exit_slippage,
        "funding_buffer_bps": funding_buffer,
        "base_cost_bps": base,
        "stress_cost_bps": base * stress_multiplier,
        "profit_buffer_bps": float(strategy_cfg.get("minimum_profit_buffer_bps", 8.0)),
    }
