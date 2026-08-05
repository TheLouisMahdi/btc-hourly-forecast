from __future__ import annotations

from typing import Any

import pandas as pd

from .forecast_contract import build_next_candle_forecast

STRICT_CONTRACT_VERSION = 3
DEFAULT_SETTLEMENT_DELAY_SECONDS = 90


def build_strict_next_candle_forecast(
    record: dict[str, Any],
    model_metrics: dict[str, Any] | None,
    recent_candles: pd.DataFrame,
    history: list[dict[str, Any]],
    interval_probability: float = 0.80,
) -> dict[str, Any]:
    """Build a non-retroactive forecast for exactly one future candle close.

    `candle_time` is the source candle OPEN time. The forecast is created only
    after that source candle closes and before the target candle closes. This
    makes the economic meaning unambiguous: one hour from forecast creation to
    the next scheduled close, even though the target close is two hours after
    the source candle OPEN timestamp.
    """
    source_open = _utc(record["candle_time"])
    source_close = source_open + pd.Timedelta(hours=1)
    target_open = source_close
    target_close = target_open + pd.Timedelta(hours=1)
    created_at = _utc(
        record.get("run_finished_at")
        or record.get("created_at")
        or pd.Timestamp.now(tz="UTC")
    )

    if created_at < source_close:
        raise ValueError(
            "Forecast creation precedes the source candle close; only closed "
            "candles may create a forecast"
        )
    if created_at >= target_close:
        raise ValueError(
            "Target candle already closed before forecast creation; refusing "
            "a retroactive next-candle forecast"
        )

    source_row = _source_candle(recent_candles, source_open)
    if source_row is None:
        raise ValueError(
            "The exact source candle is unavailable in recent closed candles"
        )
    record_price = float(record.get("price") or 0.0)
    source_close_price = float(source_row["close"])
    tolerance = max(0.01, abs(source_close_price) * 1e-8)
    if record_price <= 0 or abs(record_price - source_close_price) > tolerance:
        raise ValueError(
            "Record price does not match the exact source candle close"
        )

    contract = build_next_candle_forecast(
        record,
        model_metrics,
        recent_candles,
        history,
        interval_probability=interval_probability,
    )
    settlement_not_before = target_close + pd.Timedelta(
        seconds=DEFAULT_SETTLEMENT_DELAY_SECONDS
    )
    contract.update(
        {
            "contract_version": STRICT_CONTRACT_VERSION,
            "forecast_created_at": created_at.isoformat(),
            "forecast_horizon_seconds": int(
                (target_close - created_at).total_seconds()
            ),
            "source_close_age_seconds": int(
                (created_at - source_close).total_seconds()
            ),
            "settlement_not_before": settlement_not_before.isoformat(),
            "timing_status": "EXACT_NEXT_CLOSED_CANDLE",
            "source_close_verified": True,
            "retroactive_forecast": False,
        }
    )
    return contract


def _source_candle(
    recent_candles: pd.DataFrame,
    source_open: pd.Timestamp,
) -> dict[str, float] | None:
    if recent_candles.empty or "open_time" not in recent_candles:
        return None
    frame = recent_candles.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    match = frame.loc[frame["open_time"] == source_open]
    if match.empty:
        return None
    row = match.iloc[-1]
    try:
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
