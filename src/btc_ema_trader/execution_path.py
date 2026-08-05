from __future__ import annotations

from typing import Any

import pandas as pd

from .config import Settings

EXECUTION_PATH_CONTRACT = "FIRST_FULL_HOURLY_CANDLE_AFTER_ENTRY"


def install_execution_path_contract(trade: dict[str, Any]) -> dict[str, Any]:
    """Freeze the first candle that can be evaluated after the paper entry."""
    opened_at = _utc(
        trade.get("opened_at")
        or trade.get("signal_candle_time")
        or pd.Timestamp.now(tz="UTC")
    )
    first_open = first_full_candle_open(opened_at)
    trade["execution_path_contract"] = EXECUTION_PATH_CONTRACT
    trade["first_evaluable_candle_open"] = first_open.isoformat()
    trade["partial_entry_candle_used_for_barriers"] = False
    return trade


def first_full_candle_open(opened_at: Any) -> pd.Timestamp:
    timestamp = _utc(opened_at)
    boundary = timestamp.floor("h")
    if timestamp == boundary:
        return boundary
    return boundary + pd.Timedelta(hours=1)


def resolve_open_trades_after_entry(
    trades: list[dict[str, Any]],
    candles: pd.DataFrame,
    settings: Settings,
) -> int:
    """Resolve positions only with fully observable post-entry hourly candles."""
    if candles.empty:
        return 0

    # Reuse the lifecycle's barrier and accounting functions while replacing
    # only the timestamp selection contract.
    from . import trade_lifecycle as lifecycle

    frame = candles.copy().sort_values("open_time").reset_index(drop=True)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    resolved = 0
    for trade in trades:
        if trade.get("status") != "OPEN":
            continue
        if not trade.get("first_evaluable_candle_open"):
            install_execution_path_contract(trade)
        first_open = _utc(trade["first_evaluable_candle_open"])
        expiry = _utc(trade.get("expires_at"))
        relevant = frame.loc[frame["open_time"] >= first_open]
        if relevant.empty:
            continue

        for _, candle in relevant.iterrows():
            candle_time = _utc(candle["open_time"])
            event = lifecycle._evaluate_candle(trade, candle, settings)
            if event is not None:
                lifecycle._close_trade(
                    trade,
                    exit_price=event["exit_price"],
                    outcome=event["outcome"],
                    closed_at=candle_time + pd.Timedelta(hours=1),
                )
                resolved += 1
                break
            lifecycle._update_dynamic_stop(trade, candle)
            if candle_time + pd.Timedelta(hours=1) >= expiry:
                lifecycle._close_trade(
                    trade,
                    exit_price=float(candle["close"]),
                    outcome="TIME_EXIT",
                    closed_at=candle_time + pd.Timedelta(hours=1),
                )
                resolved += 1
                break
    return resolved


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
