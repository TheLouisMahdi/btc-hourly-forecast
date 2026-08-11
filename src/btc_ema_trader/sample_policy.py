from __future__ import annotations

from bisect import bisect_left, insort
from collections import deque
from math import floor
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Settings

WEEKDAY_MAX = 4

_LABEL_PREFIXES = (
    "target_",
    "future_",
    "entry_",
    "breakout_success_",
    "breakout_hold_",
    "false_breakout_",
    "neutral_breakout_",
    "tradeable_",
    "event_gross_return_",
    "event_net_return_",
    "event_mfe_",
    "event_mae_",
    "event_hold_ratio_",
    "event_target_first_",
    "event_stop_first_",
    "event_continuation_",
)


def apply_model_calendar(candles: pd.DataFrame) -> pd.DataFrame:
    """Keep Monday-Friday UTC candles before feature engineering."""
    if candles.empty:
        return candles.copy()
    output = candles.copy()
    output["open_time"] = pd.to_datetime(output["open_time"], utc=True)
    output = output.loc[output["open_time"].dt.dayofweek <= WEEKDAY_MAX]
    return (
        output.sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
        .reset_index(drop=True)
    )


def attach_sample_policy(
    frame: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    cfg = settings.section("sample_policy")
    liquidity = cfg.get("low_liquidity", {})
    if not isinstance(liquidity, dict):
        liquidity = {}
    enabled = bool(liquidity.get("enabled", True))
    lookback = max(1, int(liquidity.get("eligible_lookback", 480)))
    quantile = float(np.clip(liquidity.get("quantile", 0.10), 0.0, 1.0))
    if not enabled:
        output = frame.copy()
        output["model_weekday_eligible"] = True
        output["model_dollar_volume"] = _dollar_volume(output)
        output["model_liquidity_threshold"] = np.nan
        output["model_low_liquidity"] = False
        output["model_sample_eligible"] = True
        return output
    return mark_liquidity_eligibility(
        frame,
        lookback=lookback,
        quantile=quantile,
    )


def mark_liquidity_eligibility(
    frame: pd.DataFrame,
    *,
    lookback: int = 480,
    quantile: float = 0.10,
) -> pd.DataFrame:
    """Mark low liquidity from previous eligible weekday candles only."""
    output = frame.copy()
    if output.empty:
        output["model_weekday_eligible"] = pd.Series(dtype=bool)
        output["model_dollar_volume"] = pd.Series(dtype=float)
        output["model_liquidity_threshold"] = pd.Series(dtype=float)
        output["model_low_liquidity"] = pd.Series(dtype=bool)
        output["model_sample_eligible"] = pd.Series(dtype=bool)
        return output

    lookback = max(1, int(lookback))
    quantile = float(np.clip(quantile, 0.0, 1.0))
    times = pd.to_datetime(output["open_time"], utc=True)
    weekday = times.dt.dayofweek <= WEEKDAY_MAX
    dollar_volume = _dollar_volume(output)

    previous_eligible: deque[float] = deque()
    sorted_eligible: list[float] = []
    thresholds = np.full(len(output), np.nan, dtype=float)
    low_liquidity = np.zeros(len(output), dtype=bool)

    for index, value in enumerate(dollar_volume.to_numpy(dtype=float)):
        if not bool(weekday.iloc[index]):
            continue
        if not np.isfinite(value) or value <= 0.0:
            low_liquidity[index] = True
            continue

        if len(previous_eligible) >= lookback:
            threshold = _sorted_quantile(sorted_eligible, quantile)
            thresholds[index] = threshold
            if value < threshold:
                low_liquidity[index] = True
                continue

        previous_eligible.append(value)
        insort(sorted_eligible, value)
        if len(previous_eligible) > lookback:
            removed = previous_eligible.popleft()
            position = bisect_left(sorted_eligible, removed)
            sorted_eligible.pop(position)

    output["model_weekday_eligible"] = weekday.to_numpy(dtype=bool)
    output["model_dollar_volume"] = dollar_volume.to_numpy(dtype=float)
    output["model_liquidity_threshold"] = thresholds
    output["model_low_liquidity"] = low_liquidity
    output["model_sample_eligible"] = (
        weekday.to_numpy(dtype=bool) & ~low_liquidity
    )
    return output


def suppress_ineligible_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Prevent an ineligible source candle from becoming a model event."""
    output = frame.copy()
    if "model_sample_eligible" not in output:
        return output
    eligible = output["model_sample_eligible"].fillna(False).astype(bool)
    blocked = ~eligible
    if not blocked.any() or "event_direction" not in output:
        return output

    zero_columns = (
        "event_direction",
        "event_score",
        "event_scale_hours",
        "breakout_confirmed",
        "is_event",
        "aligned_body_atr",
        "aligned_close_quality",
        "aligned_regime",
        "aligned_rsi",
        "aligned_ema168_slope",
    )
    nan_columns = (
        "breakout_level",
        "breakout_invalidation_level",
        "breakout_distance_atr",
        "breakout_level_age_bars",
        "breakout_line_slope_atr",
        "breakout_line_r2",
    )
    for column in zero_columns:
        if column in output:
            output.loc[blocked, column] = 0
    for column in nan_columns:
        if column in output:
            output.loc[blocked, column] = np.nan
    if "breakout_level_touches" in output:
        output.loc[blocked, "breakout_level_touches"] = 0
    for column in ("event_type", "breakout_source"):
        if column in output:
            output.loc[blocked, column] = "NONE"
    for column in ("event_id", "event_diversity_key"):
        if column in output:
            output.loc[blocked, column] = None

    direction = pd.to_numeric(
        output["event_direction"], errors="coerce"
    ).fillna(0).astype(int).to_numpy()
    bars_since: list[float] = []
    active_direction: list[int] = []
    last_index: int | None = None
    last_direction = 0
    for index, value in enumerate(direction):
        if value != 0:
            last_index = index
            last_direction = int(value)
        bars_since.append(
            np.nan if last_index is None else float(index - last_index)
        )
        active_direction.append(last_direction)
    output["bars_since_event"] = bars_since
    output["active_event_direction"] = active_direction
    if "regime_code" in output:
        regime = pd.to_numeric(
            output["regime_code"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        output["event_continuation_bias"] = regime * direction
    return output


def invalidate_ineligible_labels(
    frame: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Invalidate labels crossing weekends, gaps, or low-liquidity candles."""
    output = frame.copy()
    for raw_horizon in horizons:
        horizon = int(raw_horizon)
        valid = horizon_path_eligible(output, horizon)
        output[f"model_path_eligible_h{horizon}"] = valid.to_numpy(dtype=bool)
        suffix = f"_h{horizon}"
        for column in output.columns:
            if column.endswith(suffix) and column.startswith(_LABEL_PREFIXES):
                output.loc[~valid, column] = np.nan
    return output


def horizon_path_eligible(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Require an exact hourly path whose source and all future rows are eligible."""
    horizon = max(1, int(horizon))
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    times = pd.to_datetime(frame["open_time"], utc=True)
    if "model_sample_eligible" in frame:
        eligible = frame["model_sample_eligible"].fillna(False).astype(bool)
    else:
        eligible = times.dt.dayofweek <= WEEKDAY_MAX

    valid = eligible.copy()
    for offset in range(1, horizon + 1):
        valid &= eligible.shift(-offset, fill_value=False)
    target_time = times.shift(-horizon)
    valid &= target_time.eq(times + pd.Timedelta(hours=horizon))
    return valid.fillna(False).astype(bool)


def filter_history_for_model_policy(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [item for item in history if history_record_eligible(item)]


def history_record_eligible(item: dict[str, Any]) -> bool:
    explicit = item.get("model_sample_eligible")
    if explicit is False or item.get("model_low_liquidity") is True:
        return False
    source = item.get("candle_time")
    if source and not _weekday_timestamp(source):
        return False
    contract = item.get("next_candle_forecast")
    if isinstance(contract, dict):
        target = contract.get("target_open_time")
        if target and not _weekday_timestamp(target):
            return False
    return True


def _dollar_volume(frame: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    volume = pd.to_numeric(frame.get("volume"), errors="coerce")
    fallback = close * volume
    if "quote_volume" not in frame:
        return fallback.astype(float)
    quote = pd.to_numeric(frame["quote_volume"], errors="coerce")
    return quote.where(quote > 0.0, fallback).astype(float)


def _sorted_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * quantile
    lower = int(floor(position))
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return float(values[lower] + fraction * (values[upper] - values[lower]))


def _weekday_timestamp(value: Any) -> bool:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.dayofweek) <= WEEKDAY_MAX
