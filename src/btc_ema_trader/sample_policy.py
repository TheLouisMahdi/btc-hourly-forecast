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
    """Preserve the full UTC market calendar for feature engineering."""
    if candles.empty:
        return candles.copy()
    output = candles.copy()
    output["open_time"] = pd.to_datetime(output["open_time"], utc=True)
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

    return mark_liquidity_eligibility(
        frame,
        lookback=max(1, int(liquidity.get("eligible_lookback", 480))),
        quantile=float(
            np.clip(liquidity.get("quantile", 0.10), 0.0, 1.0)
        ),
        liquidity_enabled=bool(liquidity.get("enabled", True)),
        weekend_weight=float(
            np.clip(cfg.get("weekend_weight_multiplier", 0.25), 0.0, 1.0)
        ),
        low_liquidity_weight=float(
            np.clip(
                liquidity.get("sample_weight_multiplier", 0.35),
                0.0,
                1.0,
            )
        ),
        minimum_weight=float(
            np.clip(cfg.get("minimum_weight_multiplier", 0.15), 0.0, 1.0)
        ),
    )


def mark_liquidity_eligibility(
    frame: pd.DataFrame,
    *,
    lookback: int = 480,
    quantile: float = 0.10,
    liquidity_enabled: bool = True,
    weekend_weight: float = 0.25,
    low_liquidity_weight: float = 0.35,
    minimum_weight: float = 0.15,
) -> pd.DataFrame:
    """Mark weak samples causally and assign a reduced training weight."""
    output = frame.copy()
    if output.empty:
        output["model_weekday_eligible"] = pd.Series(dtype=bool)
        output["model_weekend"] = pd.Series(dtype=bool)
        output["model_dollar_volume"] = pd.Series(dtype=float)
        output["model_liquidity_threshold"] = pd.Series(dtype=float)
        output["model_low_liquidity"] = pd.Series(dtype=bool)
        output["model_sample_eligible"] = pd.Series(dtype=bool)
        output["model_sample_weight_multiplier"] = pd.Series(dtype=float)
        return output

    lookback = max(1, int(lookback))
    quantile = float(np.clip(quantile, 0.0, 1.0))
    weekend_weight = float(np.clip(weekend_weight, 0.0, 1.0))
    low_liquidity_weight = float(
        np.clip(low_liquidity_weight, 0.0, 1.0)
    )
    minimum_weight = float(np.clip(minimum_weight, 0.0, 1.0))

    times = pd.to_datetime(output["open_time"], utc=True)
    weekday = times.dt.dayofweek <= WEEKDAY_MAX
    weekend = ~weekday
    dollar_volume = _dollar_volume(output)

    previous_reference: deque[float] = deque()
    sorted_reference: list[float] = []
    thresholds = np.full(len(output), np.nan, dtype=float)
    low_liquidity = np.zeros(len(output), dtype=bool)

    for index, value in enumerate(dollar_volume.to_numpy(dtype=float)):
        valid_volume = bool(np.isfinite(value) and value > 0.0)
        if liquidity_enabled and len(previous_reference) >= lookback:
            threshold = _sorted_quantile(sorted_reference, quantile)
            thresholds[index] = threshold
            low_liquidity[index] = (
                not valid_volume or bool(value < threshold)
            )
        elif liquidity_enabled and not valid_volume:
            low_liquidity[index] = True

        if (
            bool(weekday.iloc[index])
            and valid_volume
            and not bool(low_liquidity[index])
        ):
            previous_reference.append(value)
            insort(sorted_reference, value)
            if len(previous_reference) > lookback:
                removed = previous_reference.popleft()
                position = bisect_left(sorted_reference, removed)
                sorted_reference.pop(position)

    multiplier = np.ones(len(output), dtype=float)
    multiplier[weekend.to_numpy(dtype=bool)] *= weekend_weight
    multiplier[low_liquidity] *= low_liquidity_weight
    multiplier = np.clip(multiplier, minimum_weight, 1.0)

    output["model_weekday_eligible"] = weekday.to_numpy(dtype=bool)
    output["model_weekend"] = weekend.to_numpy(dtype=bool)
    output["model_dollar_volume"] = dollar_volume.to_numpy(dtype=float)
    output["model_liquidity_threshold"] = thresholds
    output["model_low_liquidity"] = low_liquidity
    output["model_sample_eligible"] = True
    output["model_sample_weight_multiplier"] = multiplier
    return output


def suppress_ineligible_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain all real events; weak samples are handled by sample weights."""
    return frame.copy()


def invalidate_ineligible_labels(
    frame: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Invalidate only labels that cross a real missing-candle gap."""
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
    """Require an exact uninterrupted hourly path for each target."""
    horizon = max(1, int(horizon))
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    times = pd.to_datetime(frame["open_time"], utc=True)
    target_time = times.shift(-horizon)
    valid = target_time.eq(times + pd.Timedelta(hours=horizon))
    return valid.fillna(False).astype(bool)


def filter_history_for_model_policy(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep weekend and low-liquidity forecasts in evaluation history."""
    return list(history)


def history_record_eligible(item: dict[str, Any]) -> bool:
    """All real forecast records remain eligible for evaluation."""
    return bool(isinstance(item, dict))


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
