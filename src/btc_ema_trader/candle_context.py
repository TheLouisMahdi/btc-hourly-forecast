from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CONTEXT_CONTRACT = "EVENT_AND_TWO_PREVIOUS_CLOSED_CANDLES"
CONTEXT_PREFIX = "candle_ctx_"


def attach_causal_candle_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach event candle plus two prior closed-candle features.

    All model inputs are causal. Lag 0 is the event/source candle, while lag 1
    and lag 2 are the two immediately preceding closed candles. Future candles
    are never used as predictors.
    """
    if frame.empty:
        return frame.copy()

    output = frame.copy().sort_values("open_time").reset_index(drop=True)
    open_ = pd.to_numeric(output["open"], errors="coerce")
    high = pd.to_numeric(output["high"], errors="coerce")
    low = pd.to_numeric(output["low"], errors="coerce")
    close = pd.to_numeric(output["close"], errors="coerce")
    atr = pd.to_numeric(output.get("atr"), errors="coerce").replace(0, np.nan)
    previous_close = close.shift(1)

    body = close - open_
    candle_range = high - low
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    close_location = (close - low) / candle_range.replace(0, np.nan)
    volume_z = pd.to_numeric(
        output.get("volume_z_24", pd.Series(0.0, index=output.index)),
        errors="coerce",
    ).fillna(0.0)

    base = {
        "open_gap_atr": (open_ - previous_close) / atr,
        "close_change_atr": (close - previous_close) / atr,
        "body_atr": body / atr,
        "range_atr": candle_range / atr,
        "upper_wick_atr": upper_wick / atr,
        "lower_wick_atr": lower_wick / atr,
        "close_location": close_location,
        "volume_z_24": volume_z,
    }
    for lag in range(3):
        for name, values in base.items():
            output[f"{CONTEXT_PREFIX}lag{lag}_{name}"] = values.shift(lag)

    atr_current = atr
    three_body = body + body.shift(1) + body.shift(2)
    three_range = candle_range + candle_range.shift(1) + candle_range.shift(2)
    three_upper = upper_wick + upper_wick.shift(1) + upper_wick.shift(2)
    three_lower = lower_wick + lower_wick.shift(1) + lower_wick.shift(2)

    output[f"{CONTEXT_PREFIX}3bar_net_close_atr"] = (
        close - open_.shift(2)
    ) / atr_current
    output[f"{CONTEXT_PREFIX}3bar_body_sum_atr"] = three_body / atr_current
    output[f"{CONTEXT_PREFIX}3bar_range_sum_atr"] = three_range / atr_current
    output[f"{CONTEXT_PREFIX}3bar_upper_wick_sum_atr"] = three_upper / atr_current
    output[f"{CONTEXT_PREFIX}3bar_lower_wick_sum_atr"] = three_lower / atr_current
    output[f"{CONTEXT_PREFIX}3bar_wick_pressure"] = (
        three_lower - three_upper
    ) / three_range.replace(0, np.nan)
    output[f"{CONTEXT_PREFIX}3bar_volume_slope"] = volume_z - volume_z.shift(2)
    output[f"{CONTEXT_PREFIX}3bar_higher_high_count"] = (
        (high > high.shift(1)).astype(float)
        + (high.shift(1) > high.shift(2)).astype(float)
    )
    output[f"{CONTEXT_PREFIX}3bar_lower_low_count"] = (
        (low < low.shift(1)).astype(float)
        + (low.shift(1) < low.shift(2)).astype(float)
    )
    output[f"{CONTEXT_PREFIX}3bar_bullish_count"] = (
        (close > open_).astype(float).rolling(3, min_periods=3).sum()
    )
    output[f"{CONTEXT_PREFIX}3bar_close_location_mean"] = (
        close_location.rolling(3, min_periods=3).mean()
    )
    return output.replace([np.inf, -np.inf], np.nan)


def candle_context_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith(CONTEXT_PREFIX)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def extract_candle_context(
    candles: pd.DataFrame,
    source_open_time: Any,
    *,
    previous_bars: int = 2,
) -> dict[str, Any]:
    """Return auditable raw OHLCV and shadow values for the causal context."""
    if candles.empty:
        return {
            "contract": CONTEXT_CONTRACT,
            "source_open_time": _utc(source_open_time).isoformat(),
            "bars": [],
            "complete": False,
        }

    frame = candles.copy().sort_values("open_time").drop_duplicates(
        "open_time", keep="last"
    )
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    source_time = _utc(source_open_time)
    source_matches = frame.index[frame["open_time"] == source_time].tolist()
    if not source_matches:
        return {
            "contract": CONTEXT_CONTRACT,
            "source_open_time": source_time.isoformat(),
            "bars": [],
            "complete": False,
        }

    source_index = int(source_matches[-1])
    positional_index = frame.index.get_loc(source_index)
    if isinstance(positional_index, slice):
        positional_index = positional_index.stop - 1
    start = max(0, int(positional_index) - int(previous_bars))
    window = frame.iloc[start : int(positional_index) + 1]
    roles = [
        f"PREVIOUS_{len(window) - position - 1}"
        if position < len(window) - 1
        else "EVENT"
        for position in range(len(window))
    ]

    bars: list[dict[str, Any]] = []
    for role, (_, row) in zip(roles, window.iterrows()):
        open_price = _number(row.get("open"))
        high_price = _number(row.get("high"))
        low_price = _number(row.get("low"))
        close_price = _number(row.get("close"))
        if None in {open_price, high_price, low_price, close_price}:
            continue
        assert open_price is not None
        assert high_price is not None
        assert low_price is not None
        assert close_price is not None
        candle_range = max(0.0, high_price - low_price)
        body = close_price - open_price
        upper_wick = max(0.0, high_price - max(open_price, close_price))
        lower_wick = max(0.0, min(open_price, close_price) - low_price)
        bars.append(
            {
                "role": role,
                "open_time": _utc(row["open_time"]).isoformat(),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": _number(row.get("volume")),
                "body": body,
                "body_percent": body / open_price if open_price > 0 else None,
                "range": candle_range,
                "range_percent": (
                    candle_range / close_price if close_price > 0 else None
                ),
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "upper_wick_percent": (
                    upper_wick / close_price if close_price > 0 else None
                ),
                "lower_wick_percent": (
                    lower_wick / close_price if close_price > 0 else None
                ),
                "close_location": (
                    (close_price - low_price) / candle_range
                    if candle_range > 0
                    else 0.5
                ),
            }
        )

    return {
        "contract": CONTEXT_CONTRACT,
        "source_open_time": source_time.isoformat(),
        "bars": bars,
        "complete": len(bars) == previous_bars + 1,
        "future_bars_used_as_features": False,
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
