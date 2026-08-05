from __future__ import annotations

from typing import Any


ACTIVE_POSITION_CONTRACT = "ACTIVE_TARGET_STOP_POSITION"


def build_active_position_plan(
    trade: dict[str, Any],
) -> dict[str, Any]:
    """Return one normalized management plan for the currently open position.

    The hourly candidate plan must never masquerade as the active position.
    This projection contains only values frozen at entry or updated by the
    position lifecycle itself.
    """
    direction = str(trade.get("direction") or "").upper()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError("An active position must be LONG or SHORT")
    if str(trade.get("status") or "").upper() != "OPEN":
        raise ValueError("Only an OPEN position can create a management plan")

    entry = _positive(trade.get("entry_price"), "entry_price")
    target = _positive(trade.get("target_price"), "target_price")
    current_stop = _positive(
        trade.get("current_stop_price", trade.get("initial_stop_price")),
        "current_stop_price",
    )
    initial_stop = _positive(
        trade.get("initial_stop_price", current_stop),
        "initial_stop_price",
    )

    return {
        "status": "MANAGING_OPEN_TRADE",
        "contract_type": ACTIVE_POSITION_CONTRACT,
        "direction": direction,
        "trade_id": trade.get("trade_id"),
        "model_id": trade.get("model_id"),
        "event_id": trade.get("event_id"),
        "event_type": trade.get("event_type", "NONE"),
        "entry_reference": entry,
        "entry_reference_kind": "FROZEN_POSITION_ENTRY",
        "entry_definition": trade.get(
            "entry_definition",
            "PAPER_MARKET_ORDER_AT_SIGNAL_RUN",
        ),
        "opened_at": trade.get("opened_at"),
        "signal_candle_time": trade.get("signal_candle_time"),
        "target_price": target,
        "stop_price": current_stop,
        "current_stop_price": current_stop,
        "initial_stop_price": initial_stop,
        "risk_reward": trade.get("risk_reward"),
        "maximum_holding_hours": trade.get("maximum_holding_hours"),
        "expires_at": trade.get("expires_at"),
        "breakeven_trigger_r": trade.get("breakeven_trigger_r"),
        "trailing_trigger_r": trade.get("trailing_trigger_r"),
        "trailing_atr_multiplier": trade.get("trailing_atr_multiplier"),
        "breakeven_armed": bool(trade.get("breakeven_armed", False)),
        "trailing_armed": bool(trade.get("trailing_armed", False)),
        "max_favorable_r": trade.get("max_favorable_r"),
        "max_adverse_r": trade.get("max_adverse_r"),
        "entry_atr": trade.get("entry_atr"),
        "quantity_btc": trade.get("quantity_btc"),
        "notional_usd": trade.get("notional_usd"),
        "margin_required_usd": trade.get("margin_required_usd"),
        "risk_budget_usd": trade.get("risk_budget_usd"),
        "stress_execution_cost_bps": trade.get(
            "stress_execution_cost_bps"
        ),
        "target_net_profit_usd": trade.get("target_net_profit_usd"),
        "stop_net_loss_usd": trade.get("stop_net_loss_usd"),
        "expected_value_usd": trade.get("expected_value_usd"),
        "target_margin_roi": trade.get("target_margin_roi"),
        "adaptive_target_probability": trade.get(
            "adaptive_target_probability"
        ),
        "adaptive_stop_probability": trade.get(
            "adaptive_stop_probability"
        ),
        "entry_feature_names": list(trade.get("entry_feature_names", [])),
        "entry_feature_vector": list(trade.get("entry_feature_vector", [])),
        "event_candle_context": trade.get("event_candle_context"),
        "candle_context_contract": trade.get("candle_context_contract"),
        "candle_context_complete": bool(
            trade.get("candle_context_complete", False)
        ),
        "paper_only": True,
        "decision_mode": "ACTIVE_POSITION_MANAGEMENT",
    }


def _positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not number > 0:
        raise ValueError(f"{name} must be a positive number")
    return number
