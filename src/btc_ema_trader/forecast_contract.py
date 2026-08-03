from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INTERVAL_PROBABILITY = 0.80
MINIMUM_RESIDUAL_SAMPLES = 20
MAXIMUM_RESIDUAL_SAMPLES = 240


@dataclass(frozen=True)
class NextCandleForecast:
    contract_version: int
    target: str
    interval_probability: float
    source_open_time: str
    source_close_time: str
    target_open_time: str
    target_close_time: str
    reference_close: float
    median_return: float
    likely_return_low: float
    likely_return_high: float
    median_close: float
    likely_close_low: float
    likely_close_high: float
    probability_up: float
    probability_down: float
    direction: str
    direction_confidence: float
    scenario: str
    interval_method: str
    calibration_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_next_candle_forecast(
    record: dict[str, Any],
    model_metrics: dict[str, Any] | None,
    recent_candles: pd.DataFrame,
    history: list[dict[str, Any]],
    interval_probability: float = DEFAULT_INTERVAL_PROBABILITY,
) -> dict[str, Any]:
    source_open_time = _utc(record["candle_time"])
    source_close_time = source_open_time + pd.Timedelta(hours=1)
    target_open_time = source_close_time
    target_close_time = target_open_time + pd.Timedelta(hours=1)
    reference_close = _finite(record.get("price"), 0.0)
    if reference_close <= 0:
        raise ValueError("A positive source candle close is required")

    probability_up = float(
        np.clip(_mapping_value(record.get("probabilities"), 1, 0.5), 1e-4, 1 - 1e-4)
    )
    median_return = _mapping_value(record.get("returns"), 1, 0.0)
    residuals = _resolved_residuals(history)

    if len(residuals) >= MINIMUM_RESIDUAL_SAMPLES:
        alpha = (1.0 - interval_probability) / 2.0
        lower_error, upper_error = np.quantile(
            np.asarray(residuals[-MAXIMUM_RESIDUAL_SAMPLES:], dtype=float),
            [alpha, 1.0 - alpha],
        )
        likely_return_low = median_return + float(lower_error)
        likely_return_high = median_return + float(upper_error)
        interval_method = "EMPIRICAL_PREQUENTIAL_RESIDUAL"
        calibration_samples = min(len(residuals), MAXIMUM_RESIDUAL_SAMPLES)
    else:
        return_mae = _model_return_mae(model_metrics)
        market_half_range = _market_half_range(recent_candles)
        robust_error = max(return_mae, market_half_range, 0.0015)
        normal_multiplier = 1.2815515655446004
        half_width = normal_multiplier * robust_error
        likely_return_low = median_return - half_width
        likely_return_high = median_return + half_width
        interval_method = "MODEL_MAE_AND_MARKET_RANGE_FALLBACK"
        calibration_samples = len(residuals)

    minimum_half_width = max(_market_half_range(recent_candles) * 0.35, 0.0010)
    center = median_return
    current_half_width = max(
        center - likely_return_low,
        likely_return_high - center,
    )
    if current_half_width < minimum_half_width:
        likely_return_low = center - minimum_half_width
        likely_return_high = center + minimum_half_width

    likely_return_low, likely_return_high = sorted(
        (float(likely_return_low), float(likely_return_high))
    )
    median_close = reference_close * (1.0 + median_return)
    likely_close_low = max(0.0, reference_close * (1.0 + likely_return_low))
    likely_close_high = max(likely_close_low, reference_close * (1.0 + likely_return_high))

    if probability_up >= 0.58:
        direction = "UP"
        scenario = "BULLISH_BIAS"
    elif probability_up <= 0.42:
        direction = "DOWN"
        scenario = "BEARISH_BIAS"
    else:
        direction = "RANGE"
        scenario = "RANGE_BIAS"
    direction_confidence = max(probability_up, 1.0 - probability_up)

    return NextCandleForecast(
        contract_version=1,
        target="NEXT_CLOSED_1H_CANDLE",
        interval_probability=float(interval_probability),
        source_open_time=source_open_time.isoformat(),
        source_close_time=source_close_time.isoformat(),
        target_open_time=target_open_time.isoformat(),
        target_close_time=target_close_time.isoformat(),
        reference_close=float(reference_close),
        median_return=float(median_return),
        likely_return_low=float(likely_return_low),
        likely_return_high=float(likely_return_high),
        median_close=float(median_close),
        likely_close_low=float(likely_close_low),
        likely_close_high=float(likely_close_high),
        probability_up=float(probability_up),
        probability_down=float(1.0 - probability_up),
        direction=direction,
        direction_confidence=float(direction_confidence),
        scenario=scenario,
        interval_method=interval_method,
        calibration_samples=int(calibration_samples),
    ).to_dict()


def _resolved_residuals(history: list[dict[str, Any]]) -> list[float]:
    residuals: list[float] = []
    for item in history:
        if item.get("prediction_result") not in {"IN_RANGE", "OUT_OF_RANGE"}:
            continue
        contract = item.get("next_candle_forecast")
        if not isinstance(contract, dict):
            continue
        actual_return = _optional_finite(item.get("actual_close_return"))
        predicted_return = _optional_finite(contract.get("median_return"))
        if actual_return is None or predicted_return is None:
            continue
        residuals.append(actual_return - predicted_return)
    return residuals[-MAXIMUM_RESIDUAL_SAMPLES:]


def _model_return_mae(metrics: dict[str, Any] | None) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    horizon = metrics.get("1", metrics.get(1, {}))
    if not isinstance(horizon, dict):
        return 0.0
    return max(0.0, _finite(horizon.get("return_mae"), 0.0))


def _market_half_range(candles: pd.DataFrame) -> float:
    if candles.empty or not {"high", "low", "close"}.issubset(candles.columns):
        return 0.0
    close = pd.to_numeric(candles["close"], errors="coerce")
    high = pd.to_numeric(candles["high"], errors="coerce")
    low = pd.to_numeric(candles["low"], errors="coerce")
    values = ((high - low) / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0
    return max(0.0, float(values.tail(168).quantile(0.70)) / 2.0)


def _mapping_value(value: Any, key: int, default: float) -> float:
    if not isinstance(value, dict):
        return float(default)
    return _finite(value.get(key, value.get(str(key), default)), default)


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _optional_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
