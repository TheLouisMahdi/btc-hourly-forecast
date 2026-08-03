from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

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
    signal_strength: str
    scenario: str
    interval_method: str
    calibration_samples: int
    forecast_source: str
    batch_probability_up: float
    online_probability_up: float
    direction_blend_weight: float
    batch_return: float
    online_return: float
    return_blend_weight: float
    return_direction_consistent: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def attach_close_based_general_labels(
    frame: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    output = frame.copy()
    source_close = pd.to_numeric(
        output["close"],
        errors="coerce",
    )
    for raw_horizon in horizons:
        horizon = int(raw_horizon)
        future_close = source_close.shift(-horizon)
        close_return = (
            future_close / source_close.replace(0, np.nan) - 1.0
        )
        output[f"future_return_h{horizon}"] = close_return
        output[f"target_up_h{horizon}"] = (
            close_return > 0
        ).astype(float)
        output.loc[
            close_return.isna(),
            f"target_up_h{horizon}",
        ] = np.nan
    return output


def build_next_candle_forecast(
    record: dict[str, Any],
    model_metrics: dict[str, Any] | None,
    recent_candles: pd.DataFrame,
    history: list[dict[str, Any]],
    interval_probability: float = DEFAULT_INTERVAL_PROBABILITY,
) -> dict[str, Any]:
    del recent_candles
    source_open_time = _utc(record["candle_time"])
    source_close_time = source_open_time + pd.Timedelta(hours=1)
    target_open_time = source_close_time
    target_close_time = target_open_time + pd.Timedelta(hours=1)
    reference_close = _finite(record.get("price"), 0.0)
    if reference_close <= 0:
        raise ValueError("A positive source candle close is required")

    adaptive = record.get("price_forecast_model")
    if not isinstance(adaptive, dict):
        adaptive = {}
    probability_source = record.get(
        "general_probabilities",
        record.get("probabilities"),
    )
    return_source = record.get(
        "general_return_estimates",
        record.get("returns"),
    )
    probability_up = float(
        np.clip(
            _mapping_value(probability_source, 1, 0.5),
            0.05,
            0.95,
        )
    )
    median_return = float(
        np.clip(
            _mapping_value(return_source, 1, 0.0),
            -0.05,
            0.05,
        )
    )

    batch_probability_up = _finite(
        adaptive.get("batch_probability_up"),
        probability_up,
    )
    online_probability_up = _finite(
        adaptive.get("online_probability_up"),
        probability_up,
    )
    direction_blend_weight = float(
        np.clip(
            _finite(adaptive.get("direction_blend_weight"), 0.0),
            0.0,
            1.0,
        )
    )
    batch_return = _finite(
        adaptive.get("batch_return"),
        median_return,
    )
    online_return = _finite(
        adaptive.get("online_return"),
        median_return,
    )
    return_blend_weight = float(
        np.clip(
            _finite(adaptive.get("return_blend_weight"), 0.0),
            0.0,
            1.0,
        )
    )
    forecast_source = str(
        adaptive.get("source") or "BATCH_CHAMPION"
    )

    residuals = _resolved_residuals(history)
    if len(residuals) >= MINIMUM_RESIDUAL_SAMPLES:
        alpha = (1.0 - interval_probability) / 2.0
        lower_error, upper_error = np.quantile(
            np.asarray(
                residuals[-MAXIMUM_RESIDUAL_SAMPLES:],
                dtype=float,
            ),
            [alpha, 1.0 - alpha],
        )
        likely_return_low = median_return + float(lower_error)
        likely_return_high = median_return + float(upper_error)
        interval_method = "LIVE_PREQUENTIAL_MODEL_RESIDUALS"
        calibration_samples = min(
            len(residuals),
            MAXIMUM_RESIDUAL_SAMPLES,
        )
    else:
        walk_forward = _walk_forward_interval(model_metrics)
        if walk_forward is not None:
            lower_error, upper_error, samples = walk_forward
            likely_return_low = median_return + lower_error
            likely_return_high = median_return + upper_error
            interval_method = "WALK_FORWARD_MODEL_RESIDUALS"
            calibration_samples = samples
        else:
            return_mae = max(
                _model_return_mae(model_metrics),
                0.0005,
            )
            half_width = 1.2815515655446004 * return_mae
            likely_return_low = median_return - half_width
            likely_return_high = median_return + half_width
            interval_method = "MODEL_RETURN_ERROR_FALLBACK"
            calibration_samples = 0

    return_mae = max(
        _model_return_mae(model_metrics),
        0.0005,
    )
    minimum_half_width = max(return_mae * 0.35, 0.0004)
    current_half_width = max(
        median_return - likely_return_low,
        likely_return_high - median_return,
    )
    if current_half_width < minimum_half_width:
        likely_return_low = median_return - minimum_half_width
        likely_return_high = median_return + minimum_half_width

    likely_return_low, likely_return_high = sorted(
        (float(likely_return_low), float(likely_return_high))
    )
    median_close = reference_close * (1.0 + median_return)
    likely_close_low = max(
        0.0,
        reference_close * (1.0 + likely_return_low),
    )
    likely_close_high = max(
        likely_close_low,
        reference_close * (1.0 + likely_return_high),
    )

    direction = "UP" if probability_up >= 0.5 else "DOWN"
    direction_confidence = max(probability_up, 1.0 - probability_up)
    if direction_confidence >= 0.67:
        signal_strength = "HIGH"
    elif direction_confidence >= 0.58:
        signal_strength = "MODERATE"
    else:
        signal_strength = "LOW"
    scenario = "BULLISH_BIAS" if direction == "UP" else "BEARISH_BIAS"
    return_direction_consistent = (
        median_return >= 0 if direction == "UP" else median_return < 0
    )

    return NextCandleForecast(
        contract_version=2,
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
        signal_strength=signal_strength,
        scenario=scenario,
        interval_method=interval_method,
        calibration_samples=int(calibration_samples),
        forecast_source=forecast_source,
        batch_probability_up=float(batch_probability_up),
        online_probability_up=float(online_probability_up),
        direction_blend_weight=float(direction_blend_weight),
        batch_return=float(batch_return),
        online_return=float(online_return),
        return_blend_weight=float(return_blend_weight),
        return_direction_consistent=bool(return_direction_consistent),
    ).to_dict()


def _resolved_residuals(
    history: list[dict[str, Any]],
) -> list[float]:
    residuals: list[float] = []
    for item in history:
        interval_result = item.get("interval_result")
        if interval_result not in {"IN_RANGE", "OUT_OF_RANGE"}:
            if item.get("prediction_result") not in {
                "IN_RANGE",
                "OUT_OF_RANGE",
            }:
                continue
        contract = item.get("next_candle_forecast")
        if not isinstance(contract, dict):
            continue
        actual_return = _optional_finite(
            item.get("actual_close_return")
        )
        predicted_return = _optional_finite(
            contract.get("median_return")
        )
        if actual_return is None or predicted_return is None:
            continue
        residuals.append(actual_return - predicted_return)
    return residuals[-MAXIMUM_RESIDUAL_SAMPLES:]


def _walk_forward_interval(
    metrics: dict[str, Any] | None,
) -> tuple[float, float, int] | None:
    horizon = _horizon_metrics(metrics)
    if horizon is None:
        return None
    lower = _optional_finite(
        horizon.get("close_interval_residual_low")
    )
    upper = _optional_finite(
        horizon.get("close_interval_residual_high")
    )
    samples = int(
        _finite(
            horizon.get("close_interval_oof_samples"),
            0.0,
        )
    )
    if lower is None or upper is None or samples <= 0:
        return None
    lower, upper = sorted((lower, upper))
    return float(lower), float(upper), samples


def _model_return_mae(
    metrics: dict[str, Any] | None,
) -> float:
    horizon = _horizon_metrics(metrics)
    if horizon is None:
        return 0.0
    return max(
        0.0,
        _finite(horizon.get("return_mae"), 0.0),
    )


def _horizon_metrics(
    metrics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(metrics, dict):
        return None
    horizon = metrics.get("1", metrics.get(1, {}))
    return horizon if isinstance(horizon, dict) else None


def _mapping_value(
    value: Any,
    key: int,
    default: float,
) -> float:
    if not isinstance(value, dict):
        return float(default)
    return _finite(
        value.get(key, value.get(str(key), default)),
        default,
    )


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
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
