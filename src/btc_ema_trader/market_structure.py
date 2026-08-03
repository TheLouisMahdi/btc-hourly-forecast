from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Settings


@dataclass(frozen=True)
class StructureColumns:
    resistance: str = "structure_resistance"
    support: str = "structure_support"
    breakout_level: str = "breakout_level"
    invalidation_level: str = "breakout_invalidation_level"
    event_type: str = "event_type"
    event_direction: str = "event_direction"
    event_score: str = "event_score"


def build_market_structure(
    frame: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    cfg = settings.section("structure")
    output = frame.copy().reset_index(drop=True)
    high = pd.to_numeric(output["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(output["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(output["close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(output["atr"], errors="coerce").to_numpy(dtype=float)
    body_atr = pd.to_numeric(output["body_atr"], errors="coerce").to_numpy(dtype=float)
    close_location = pd.to_numeric(
        output["close_location"], errors="coerce"
    ).to_numpy(dtype=float)
    volume_z = pd.to_numeric(
        output["volume_z_24"], errors="coerce"
    ).to_numpy(dtype=float)

    left = int(cfg.get("pivot_left_bars", 3))
    right = int(cfg.get("pivot_right_bars", 3))
    high_pivots, low_pivots = confirmed_pivots(
        high,
        low,
        left=left,
        right=right,
    )

    scales = tuple(
        sorted(
            {
                int(value)
                for value in cfg.get(
                    "lookback_hours",
                    [48, 120, 240, 480],
                )
                if int(value) > left + right + 4
            }
        )
    )
    if not scales:
        scales = (48, 120, 240)

    n = len(output)
    result: dict[str, np.ndarray] = {}
    for scale in scales:
        prefix = f"structure_{scale}h"
        result[f"{prefix}_resistance"] = np.full(n, np.nan)
        result[f"{prefix}_support"] = np.full(n, np.nan)
        result[f"{prefix}_resistance_slope_atr"] = np.full(n, np.nan)
        result[f"{prefix}_support_slope_atr"] = np.full(n, np.nan)
        result[f"{prefix}_resistance_r2"] = np.full(n, np.nan)
        result[f"{prefix}_support_r2"] = np.full(n, np.nan)
        result[f"{prefix}_resistance_touches"] = np.zeros(n, dtype=float)
        result[f"{prefix}_support_touches"] = np.zeros(n, dtype=float)
        result[f"{prefix}_width_atr"] = np.full(n, np.nan)

    resistance = np.full(n, np.nan)
    support = np.full(n, np.nan)
    resistance_strength = np.zeros(n, dtype=float)
    support_strength = np.zeros(n, dtype=float)
    resistance_age = np.full(n, np.nan)
    support_age = np.full(n, np.nan)

    triangle_type = np.full(n, "NONE", dtype=object)
    triangle_quality = np.zeros(n, dtype=float)
    triangle_upper = np.full(n, np.nan)
    triangle_lower = np.full(n, np.nan)
    triangle_contraction = np.zeros(n, dtype=float)
    triangle_apex_bars = np.full(n, np.nan)

    max_pivots = int(cfg.get("maximum_pivots_per_line", 8))
    touch_atr = float(cfg.get("level_touch_tolerance_atr", 0.30))
    triangle_scale = int(cfg.get("triangle_lookback_hours", 120))

    for index in range(n):
        if index < min(scales):
            continue
        current_atr = atr[index]
        reference_price = close[index - 1] if index > 0 else close[index]
        if not np.isfinite(current_atr) or current_atr <= 0:
            continue
        candidates_resistance: list[tuple[float, float, float]] = []
        candidates_support: list[tuple[float, float, float]] = []

        for scale in scales:
            prefix = f"structure_{scale}h"
            start = max(0, index - scale)
            available_highs = [
                pivot
                for pivot in high_pivots
                if start <= pivot.confirmed_at <= index - 1
            ][-max_pivots:]
            available_lows = [
                pivot
                for pivot in low_pivots
                if start <= pivot.confirmed_at <= index - 1
            ][-max_pivots:]
            high_line = fit_pivot_line(
                available_highs,
                index,
                current_atr,
            )
            low_line = fit_pivot_line(
                available_lows,
                index,
                current_atr,
            )

            static_high = _window_max(high, start, index)
            static_low = _window_min(low, start, index)
            dynamic_high = high_line.value if high_line else np.nan
            dynamic_low = low_line.value if low_line else np.nan

            scale_resistance = _nearest_level_above(
                reference_price,
                [dynamic_high, static_high],
            )
            scale_support = _nearest_level_below(
                reference_price,
                [dynamic_low, static_low],
            )
            result[f"{prefix}_resistance"][index] = scale_resistance
            result[f"{prefix}_support"][index] = scale_support

            if high_line:
                result[f"{prefix}_resistance_slope_atr"][index] = (
                    high_line.slope_atr
                )
                result[f"{prefix}_resistance_r2"][index] = high_line.r2
            if low_line:
                result[f"{prefix}_support_slope_atr"][index] = (
                    low_line.slope_atr
                )
                result[f"{prefix}_support_r2"][index] = low_line.r2

            high_touches = count_level_touches(
                high[start:index],
                scale_resistance,
                current_atr * touch_atr,
            )
            low_touches = count_level_touches(
                low[start:index],
                scale_support,
                current_atr * touch_atr,
            )
            result[f"{prefix}_resistance_touches"][index] = high_touches
            result[f"{prefix}_support_touches"][index] = low_touches
            if np.isfinite(scale_resistance) and np.isfinite(scale_support):
                result[f"{prefix}_width_atr"][index] = (
                    scale_resistance - scale_support
                ) / current_atr

            scale_weight = float(np.log1p(scale))
            if np.isfinite(scale_resistance):
                candidates_resistance.append(
                    (
                        scale_resistance,
                        scale_weight,
                        float(high_touches),
                    )
                )
            if np.isfinite(scale_support):
                candidates_support.append(
                    (
                        scale_support,
                        scale_weight,
                        float(low_touches),
                    )
                )

        chosen_resistance = choose_composite_level(
            reference_price,
            candidates_resistance,
            direction=1,
        )
        chosen_support = choose_composite_level(
            reference_price,
            candidates_support,
            direction=-1,
        )
        resistance[index] = chosen_resistance.value
        support[index] = chosen_support.value
        resistance_strength[index] = chosen_resistance.strength
        support_strength[index] = chosen_support.strength
        resistance_age[index] = level_age(
            high,
            chosen_resistance.value,
            current_atr * touch_atr,
            index,
            max(scales),
        )
        support_age[index] = level_age(
            low,
            chosen_support.value,
            current_atr * touch_atr,
            index,
            max(scales),
        )

        triangle = detect_triangle(
            index=index,
            high_pivots=high_pivots,
            low_pivots=low_pivots,
            atr=current_atr,
            price=reference_price,
            lookback=triangle_scale,
            max_pivots=max_pivots,
            cfg=cfg,
        )
        if triangle:
            triangle_type[index] = triangle.pattern
            triangle_quality[index] = triangle.quality
            triangle_upper[index] = triangle.upper
            triangle_lower[index] = triangle.lower
            triangle_contraction[index] = triangle.contraction
            triangle_apex_bars[index] = triangle.apex_bars

    for name, values in result.items():
        output[name] = values
    output["structure_resistance"] = resistance
    output["structure_support"] = support
    output["resistance_strength"] = resistance_strength
    output["support_strength"] = support_strength
    output["resistance_age_bars"] = resistance_age
    output["support_age_bars"] = support_age
    output["distance_to_resistance_atr"] = (
        resistance - close
    ) / np.where(atr > 0, atr, np.nan)
    output["distance_to_support_atr"] = (
        close - support
    ) / np.where(atr > 0, atr, np.nan)
    output["triangle_type"] = triangle_type
    output["triangle_code"] = pd.Series(triangle_type).map(
        {
            "NONE": 0,
            "SYMMETRICAL": 1,
            "ASCENDING": 2,
            "DESCENDING": -2,
        }
    ).fillna(0).astype(int)
    output["triangle_quality"] = triangle_quality
    output["triangle_upper"] = triangle_upper
    output["triangle_lower"] = triangle_lower
    output["triangle_contraction"] = triangle_contraction
    output["triangle_apex_bars"] = triangle_apex_bars
    output["triangle_width_atr"] = (
        triangle_upper - triangle_lower
    ) / np.where(atr > 0, atr, np.nan)

    event = detect_breakout_events(
        output,
        cfg,
    )
    for column, values in event.items():
        output[column] = values
    return output


@dataclass(frozen=True)
class Pivot:
    pivot_at: int
    confirmed_at: int
    value: float


@dataclass(frozen=True)
class PivotLine:
    value: float
    slope: float
    slope_atr: float
    r2: float
    points: int


@dataclass(frozen=True)
class CompositeLevel:
    value: float
    strength: float


@dataclass(frozen=True)
class Triangle:
    pattern: str
    upper: float
    lower: float
    quality: float
    contraction: float
    apex_bars: float


def confirmed_pivots(
    high: np.ndarray,
    low: np.ndarray,
    left: int,
    right: int,
) -> tuple[list[Pivot], list[Pivot]]:
    high_pivots: list[Pivot] = []
    low_pivots: list[Pivot] = []
    for confirmed_at in range(left + right, len(high)):
        pivot_at = confirmed_at - right
        start = pivot_at - left
        stop = confirmed_at + 1
        high_window = high[start:stop]
        low_window = low[start:stop]
        candidate_high = high[pivot_at]
        candidate_low = low[pivot_at]
        if np.isfinite(candidate_high) and np.isfinite(high_window).any():
            if candidate_high >= np.nanmax(high_window):
                high_pivots.append(
                    Pivot(pivot_at, confirmed_at, float(candidate_high))
                )
        if np.isfinite(candidate_low) and np.isfinite(low_window).any():
            if candidate_low <= np.nanmin(low_window):
                low_pivots.append(
                    Pivot(pivot_at, confirmed_at, float(candidate_low))
                )
    return high_pivots, low_pivots


def fit_pivot_line(
    pivots: Iterable[Pivot],
    current_index: int,
    atr: float,
) -> PivotLine | None:
    points = list(pivots)
    if len(points) < 2 or not np.isfinite(atr) or atr <= 0:
        return None
    x = np.asarray([point.pivot_at for point in points], dtype=float)
    y = np.asarray([point.value for point in points], dtype=float)
    if np.ptp(x) <= 0 or not np.isfinite(y).all():
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    residual = y - fitted
    total = y - np.mean(y)
    denominator = float(np.sum(total**2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 0.0
    return PivotLine(
        value=float(slope * current_index + intercept),
        slope=float(slope),
        slope_atr=float(slope / atr),
        r2=float(np.clip(r2, 0.0, 1.0)),
        points=len(points),
    )


def detect_triangle(
    index: int,
    high_pivots: list[Pivot],
    low_pivots: list[Pivot],
    atr: float,
    price: float,
    lookback: int,
    max_pivots: int,
    cfg: dict[str, Any],
) -> Triangle | None:
    start = max(0, index - lookback)
    highs = [
        pivot
        for pivot in high_pivots
        if start <= pivot.confirmed_at <= index - 1
    ][-max_pivots:]
    lows = [
        pivot
        for pivot in low_pivots
        if start <= pivot.confirmed_at <= index - 1
    ][-max_pivots:]
    if len(highs) < 2 or len(lows) < 2:
        return None
    upper = fit_pivot_line(highs, index, atr)
    lower = fit_pivot_line(lows, index, atr)
    if upper is None or lower is None:
        return None
    width_now = upper.value - lower.value
    past_index = max(start, index - max(12, lookback // 2))
    upper_past = upper.value - upper.slope * (index - past_index)
    lower_past = lower.value - lower.slope * (index - past_index)
    width_past = upper_past - lower_past
    if width_now <= 0 or width_past <= 0 or not np.isfinite(price) or price <= 0:
        return None
    contraction = 1.0 - width_now / width_past
    minimum_contraction = float(cfg.get("triangle_minimum_contraction", 0.15))
    maximum_width_pct = float(cfg.get("triangle_maximum_width_percent", 0.08))
    minimum_r2 = float(cfg.get("triangle_minimum_line_r2", 0.25))
    if (
        contraction < minimum_contraction
        or width_now / price > maximum_width_pct
        or upper.r2 < minimum_r2
        or lower.r2 < minimum_r2
    ):
        return None

    minimum_slope = float(cfg.get("triangle_minimum_slope_atr", 0.005))
    flat_slope = float(cfg.get("triangle_flat_slope_atr", 0.018))
    pattern = "NONE"
    if upper.slope_atr < -minimum_slope and lower.slope_atr > minimum_slope:
        pattern = "SYMMETRICAL"
    elif abs(upper.slope_atr) <= flat_slope and lower.slope_atr > minimum_slope:
        pattern = "ASCENDING"
    elif upper.slope_atr < -minimum_slope and abs(lower.slope_atr) <= flat_slope:
        pattern = "DESCENDING"
    if pattern == "NONE":
        return None

    slope_gap = lower.slope - upper.slope
    apex_bars = width_now / slope_gap if slope_gap > 0 else np.nan
    touches_quality = min(1.0, (len(highs) + len(lows)) / 8.0)
    fit_quality = (upper.r2 + lower.r2) / 2.0
    contraction_quality = float(np.clip(contraction / 0.55, 0.0, 1.0))
    quality = float(
        np.clip(
            0.35 * fit_quality
            + 0.30 * touches_quality
            + 0.35 * contraction_quality,
            0.0,
            1.0,
        )
    )
    return Triangle(
        pattern=pattern,
        upper=float(upper.value),
        lower=float(lower.value),
        quality=quality,
        contraction=float(contraction),
        apex_bars=float(apex_bars) if np.isfinite(apex_bars) else np.nan,
    )


def detect_breakout_events(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    n = len(frame)
    close = frame["close"].to_numpy(dtype=float)
    atr = frame["atr"].to_numpy(dtype=float)
    body_atr = frame["body_atr"].to_numpy(dtype=float)
    close_location = frame["close_location"].to_numpy(dtype=float)
    volume_z = frame["volume_z_24"].to_numpy(dtype=float)
    resistance = frame["structure_resistance"].to_numpy(dtype=float)
    support = frame["structure_support"].to_numpy(dtype=float)
    resistance_strength = frame["resistance_strength"].to_numpy(dtype=float)
    support_strength = frame["support_strength"].to_numpy(dtype=float)
    triangle_upper = frame["triangle_upper"].to_numpy(dtype=float)
    triangle_lower = frame["triangle_lower"].to_numpy(dtype=float)
    triangle_quality = frame["triangle_quality"].to_numpy(dtype=float)
    triangle_type = frame["triangle_type"].astype(str).to_numpy()

    event_type = np.full(n, "NONE", dtype=object)
    event_direction = np.zeros(n, dtype=int)
    event_score = np.zeros(n, dtype=float)
    event_id = np.full(n, None, dtype=object)
    breakout_level = np.full(n, np.nan)
    invalidation_level = np.full(n, np.nan)
    breakout_distance_atr = np.full(n, np.nan)
    breakout_source = np.full(n, "NONE", dtype=object)
    breakout_confirmed = np.zeros(n, dtype=int)

    buffer_atr = float(cfg.get("breakout_buffer_atr", 0.10))
    maximum_extension = float(cfg.get("breakout_maximum_extension_atr", 1.50))
    minimum_body = float(cfg.get("breakout_minimum_body_atr", 0.20))
    minimum_volume = float(cfg.get("breakout_minimum_volume_z", -0.25))
    long_close_location = float(cfg.get("long_minimum_close_location", 0.65))
    short_close_location = float(cfg.get("short_maximum_close_location", 0.35))
    triangle_min_quality = float(cfg.get("triangle_minimum_quality", 0.45))
    cooldown = int(cfg.get("event_cooldown_hours", 4))
    last_event_by_direction: dict[int, int] = {}
    sequence = 0

    for index in range(n):
        if not np.isfinite(atr[index]) or atr[index] <= 0:
            continue
        triangle_long = (
            triangle_type[index] in {"SYMMETRICAL", "ASCENDING"}
            and triangle_quality[index] >= triangle_min_quality
            and np.isfinite(triangle_upper[index])
        )
        triangle_short = (
            triangle_type[index] in {"SYMMETRICAL", "DESCENDING"}
            and triangle_quality[index] >= triangle_min_quality
            and np.isfinite(triangle_lower[index])
        )
        long_levels = [resistance[index]]
        short_levels = [support[index]]
        if triangle_long:
            long_levels.append(triangle_upper[index])
        if triangle_short:
            short_levels.append(triangle_lower[index])
        long_level = _nearest_level_below_or_equal(close[index], long_levels)
        short_level = _nearest_level_above_or_equal(close[index], short_levels)
        long_distance = (
            (close[index] - long_level) / atr[index]
            if np.isfinite(long_level)
            else np.nan
        )
        short_distance = (
            (short_level - close[index]) / atr[index]
            if np.isfinite(short_level)
            else np.nan
        )
        long_break = (
            np.isfinite(long_distance)
            and buffer_atr <= long_distance <= maximum_extension
            and body_atr[index] >= minimum_body
            and close_location[index] >= long_close_location
            and volume_z[index] >= minimum_volume
        )
        short_break = (
            np.isfinite(short_distance)
            and buffer_atr <= short_distance <= maximum_extension
            and body_atr[index] <= -minimum_body
            and close_location[index] <= short_close_location
            and volume_z[index] >= minimum_volume
        )
        direction = 1 if long_break else -1 if short_break else 0
        if direction == 0:
            continue
        previous_event = last_event_by_direction.get(direction)
        if previous_event is not None and index - previous_event <= cooldown:
            continue
        sequence += 1
        triangle_break = (
            direction > 0
            and triangle_long
            and np.isclose(long_level, triangle_upper[index], rtol=0, atol=atr[index] * 0.15)
        ) or (
            direction < 0
            and triangle_short
            and np.isclose(short_level, triangle_lower[index], rtol=0, atol=atr[index] * 0.15)
        )
        if direction > 0:
            level = long_level
            strength = resistance_strength[index]
            source = "TRIANGLE" if triangle_break else "DYNAMIC_RESISTANCE"
            kind = "TRIANGLE_BREAKOUT_LONG" if triangle_break else "RESISTANCE_BREAKOUT_LONG"
            distance = long_distance
            invalidation = level - atr[index] * float(
                cfg.get("breakout_invalidation_atr", 0.35)
            )
            close_quality = close_location[index]
        else:
            level = short_level
            strength = support_strength[index]
            source = "TRIANGLE" if triangle_break else "DYNAMIC_SUPPORT"
            kind = "TRIANGLE_BREAKDOWN_SHORT" if triangle_break else "SUPPORT_BREAKDOWN_SHORT"
            distance = short_distance
            invalidation = level + atr[index] * float(
                cfg.get("breakout_invalidation_atr", 0.35)
            )
            close_quality = 1.0 - close_location[index]
        quality = float(
            np.clip(
                0.25 * np.clip(strength / 8.0, 0.0, 1.0)
                + 0.20 * np.clip(abs(body_atr[index]) / 1.5, 0.0, 1.0)
                + 0.15 * np.clip((volume_z[index] + 0.5) / 3.0, 0.0, 1.0)
                + 0.15 * np.clip(close_quality, 0.0, 1.0)
                + 0.15 * np.clip(1.0 - distance / maximum_extension, 0.0, 1.0)
                + 0.10 * (triangle_quality[index] if triangle_break else 0.0),
                0.0,
                1.0,
            )
        )
        timestamp = pd.Timestamp(frame.iloc[index]["open_time"]).strftime(
            "%Y%m%dT%H%MZ"
        )
        event_type[index] = kind
        event_direction[index] = direction
        event_score[index] = quality
        event_id[index] = (
            f"BRK{sequence:05d}-{timestamp}-"
            f"{'LONG' if direction > 0 else 'SHORT'}"
        )
        breakout_level[index] = level
        invalidation_level[index] = invalidation
        breakout_distance_atr[index] = distance
        breakout_source[index] = source
        breakout_confirmed[index] = 1
        last_event_by_direction[direction] = index

    return {
        "event_type": event_type,
        "event_direction": event_direction,
        "event_score": event_score,
        "event_id": event_id,
        "is_event": (event_direction != 0).astype(int),
        "breakout_level": breakout_level,
        "breakout_invalidation_level": invalidation_level,
        "breakout_distance_atr": breakout_distance_atr,
        "breakout_source": breakout_source,
        "breakout_confirmed": breakout_confirmed,
    }


def count_level_touches(
    values: np.ndarray,
    level: float,
    tolerance: float,
) -> int:
    if not np.isfinite(level) or not np.isfinite(tolerance) or tolerance <= 0:
        return 0
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return 0
    return int(np.sum(np.abs(valid - level) <= tolerance))


def choose_composite_level(
    reference_price: float,
    candidates: list[tuple[float, float, float]],
    direction: int,
) -> CompositeLevel:
    valid = [item for item in candidates if np.isfinite(item[0])]
    if not valid:
        return CompositeLevel(np.nan, 0.0)
    if direction > 0:
        eligible = [item for item in valid if item[0] >= reference_price]
        pool = eligible or valid
        selected = min(pool, key=lambda item: abs(item[0] - reference_price))
    else:
        eligible = [item for item in valid if item[0] <= reference_price]
        pool = eligible or valid
        selected = min(pool, key=lambda item: abs(item[0] - reference_price))
    nearby = [
        item
        for item in valid
        if abs(item[0] - selected[0])
        <= max(abs(reference_price) * 0.0025, 1e-9)
    ]
    strength = sum(item[1] * (1.0 + min(item[2], 6.0) / 6.0) for item in nearby)
    return CompositeLevel(float(selected[0]), float(strength))


def level_age(
    values: np.ndarray,
    level: float,
    tolerance: float,
    current_index: int,
    lookback: int,
) -> float:
    if not np.isfinite(level) or tolerance <= 0:
        return np.nan
    start = max(0, current_index - lookback)
    for index in range(current_index - 1, start - 1, -1):
        if np.isfinite(values[index]) and abs(values[index] - level) <= tolerance:
            return float(current_index - index)
    return np.nan


def _window_max(values: np.ndarray, start: int, stop: int) -> float:
    window = values[start:stop]
    return float(np.nanmax(window)) if np.isfinite(window).any() else np.nan


def _window_min(values: np.ndarray, start: int, stop: int) -> float:
    window = values[start:stop]
    return float(np.nanmin(window)) if np.isfinite(window).any() else np.nan


def _nearest_level_above(reference: float, values: Iterable[float]) -> float:
    valid = [float(value) for value in values if np.isfinite(value)]
    if not valid:
        return np.nan
    above = [value for value in valid if value >= reference]
    return min(above, default=min(valid, key=lambda value: abs(value - reference)))


def _nearest_level_below(reference: float, values: Iterable[float]) -> float:
    valid = [float(value) for value in values if np.isfinite(value)]
    if not valid:
        return np.nan
    below = [value for value in valid if value <= reference]
    return max(below, default=min(valid, key=lambda value: abs(value - reference)))


def _nearest_level_below_or_equal(reference: float, values: Iterable[float]) -> float:
    valid = [float(value) for value in values if np.isfinite(value) and value <= reference]
    return max(valid) if valid else np.nan


def _nearest_level_above_or_equal(reference: float, values: Iterable[float]) -> float:
    valid = [float(value) for value in values if np.isfinite(value) and value >= reference]
    return min(valid) if valid else np.nan
