from __future__ import annotations

from typing import Any

import pandas as pd

from .config import Settings

EXECUTION_PATH_CONTRACT = "FIRST_FULL_MINUTE_AFTER_ENTRY"


def install_execution_path_contract(trade: dict[str, Any]) -> dict[str, Any]:
    """Freeze the first fully post-entry minute that can hit a barrier."""
    opened_at = _utc(
        trade.get("opened_at")
        or trade.get("signal_candle_time")
        or pd.Timestamp.now(tz="UTC")
    )
    previous = trade.get("execution_path_contract")
    first_open = first_full_minute_open(opened_at)
    if previous and previous != EXECUTION_PATH_CONTRACT:
        trade["execution_path_migrated_from"] = str(previous)
    trade["execution_path_contract"] = EXECUTION_PATH_CONTRACT
    trade["execution_bar_minutes"] = 1
    trade["first_evaluable_candle_open"] = first_open.isoformat()
    trade["partial_entry_candle_used_for_barriers"] = False
    trade["label_execution_aligned"] = False
    trade["label_entry_definition"] = "NEXT_HOURLY_OPEN"
    trade["runtime_entry_definition"] = str(
        trade.get("entry_reference_kind") or "LIVE_QUOTE_AT_SIGNAL_RUN"
    )
    trade["execution_alignment_status"] = "MINUTE_BAR_CAUSAL_APPROXIMATION"
    return trade


def first_full_minute_open(opened_at: Any) -> pd.Timestamp:
    timestamp = _utc(opened_at)
    boundary = timestamp.floor("min")
    if timestamp == boundary:
        return boundary
    return boundary + pd.Timedelta(minutes=1)


def first_full_candle_open(opened_at: Any) -> pd.Timestamp:
    """Backward-compatible hourly helper retained for external callers."""
    timestamp = _utc(opened_at)
    boundary = timestamp.floor("h")
    if timestamp == boundary:
        return boundary
    return boundary + pd.Timedelta(hours=1)


def resolve_open_trades_after_entry(
    trades: list[dict[str, Any]],
    candles: pd.DataFrame,
    settings: Settings,
    *,
    bar_minutes: int | None = None,
) -> int:
    """Resolve positions on fully observable post-entry candles."""
    if candles.empty:
        return 0

    from . import trade_lifecycle as lifecycle

    frame = candles.copy().sort_values("open_time").reset_index(drop=True)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    duration = _bar_duration(frame, bar_minutes)
    resolved = 0
    for trade in trades:
        if trade.get("status") != "OPEN":
            continue
        if trade.get("execution_path_contract") != EXECUTION_PATH_CONTRACT:
            install_execution_path_contract(trade)
        elif not trade.get("first_evaluable_candle_open"):
            install_execution_path_contract(trade)

        first_open = _utc(trade["first_evaluable_candle_open"])
        expiry = _utc(trade.get("expires_at"))
        relevant = frame.loc[frame["open_time"] >= first_open]
        if relevant.empty:
            continue

        for _, candle in relevant.iterrows():
            candle_time = _utc(candle["open_time"])
            candle_close = candle_time + duration
            event = lifecycle._evaluate_candle(trade, candle, settings)
            if event is not None:
                lifecycle._close_trade(
                    trade,
                    exit_price=event["exit_price"],
                    outcome=event["outcome"],
                    closed_at=candle_close,
                )
                resolved += 1
                break
            lifecycle._update_dynamic_stop(trade, candle)
            if candle_close >= expiry:
                lifecycle._close_trade(
                    trade,
                    exit_price=float(candle["close"]),
                    outcome="TIME_EXIT",
                    closed_at=candle_close,
                )
                resolved += 1
                break
    return resolved


def _bar_duration(
    frame: pd.DataFrame,
    explicit_minutes: int | None,
) -> pd.Timedelta:
    if explicit_minutes is not None:
        return pd.Timedelta(minutes=max(1, int(explicit_minutes)))
    times = pd.DatetimeIndex(frame["open_time"])
    if len(times) >= 2:
        differences = (
            times.to_series()
            .diff()
            .dropna()
        )
        positive = differences.loc[differences > pd.Timedelta(0)]
        if not positive.empty:
            return positive.min()
    return pd.Timedelta(minutes=1)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
