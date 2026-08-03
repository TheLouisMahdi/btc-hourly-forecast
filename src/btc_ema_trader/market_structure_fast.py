from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .market_structure import (
    Pivot,
    choose_composite_level,
    confirmed_pivots,
    count_level_touches,
    detect_breakout_events,
    detect_triangle,
    fit_pivot_line,
    level_age,
)


def build_market_structure(
    frame: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    cfg = settings.section("structure")
    output = frame.copy().reset_index(drop=True)
    high = pd.to_numeric(output["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(output["low"], errors="coerce").to_numpy(float)
    close = pd.to_numeric(output["close"], errors="coerce").to_numpy(float)
    atr = pd.to_numeric(output["atr"], errors="coerce").to_numpy(float)

    left = int(cfg.get("pivot_left_bars", 3))
    right = int(cfg.get("pivot_right_bars", 3))
    high_pivots, low_pivots = confirmed_pivots(
        high,
        low,
        left=left,
        right=right,
    )
    high_confirmations = [pivot.confirmed_at for pivot in high_pivots]
    low_confirmations = [pivot.confirmed_at for pivot in low_pivots]

    scales = tuple(
        sorted(
            {
                int(value)
                for value in cfg.get(
                    "lookback_hours",
                    [24, 48, 96, 168, 336, 720],
                )
                if int(value) > left + right + 4
            }
        )
    )
    if not scales:
        scales = (24, 48, 96, 168)

    count = len(output)
    maximum_pivots = int(cfg.get("maximum_pivots_per_line", 10))
    touch_atr = float(cfg.get("level_touch_tolerance_atr", 0.35))
    triangle_scale = int(cfg.get("triangle_lookback_hours", 240))

    static_highs = {
        scale: pd.Series(high).shift(1).rolling(
            scale,
            min_periods=max(2, min(scale, left + right + 1)),
        ).max().to_numpy(float)
        for scale in scales
    }
    static_lows = {
        scale: pd.Series(low).shift(1).rolling(
            scale,
            min_periods=max(2, min(scale, left + right + 1)),
        ).min().to_numpy(float)
        for scale in scales
    }

    result: dict[str, np.ndarray] = {}
    for scale in scales:
        prefix = f"structure_{scale}h"
        result[f"{prefix}_resistance"] = np.full(count, np.nan)
        result[f"{prefix}_support"] = np.full(count, np.nan)
        result[f"{prefix}_resistance_slope_atr"] = np.full(
            count,
            np.nan,
        )
        result[f"{prefix}_support_slope_atr"] = np.full(
            count,
            np.nan,
        )
        result[f"{prefix}_resistance_r2"] = np.full(count, np.nan)
        result[f"{prefix}_support_r2"] = np.full(count, np.nan)
        result[f"{prefix}_resistance_touches"] = np.zeros(count)
        result[f"{prefix}_support_touches"] = np.zeros(count)
        result[f"{prefix}_width_atr"] = np.full(count, np.nan)

    resistance = np.full(count, np.nan)
    support = np.full(count, np.nan)
    resistance_strength = np.zeros(count)
    support_strength = np.zeros(count)
    resistance_age = np.full(count, np.nan)
    support_age = np.full(count, np.nan)

    triangle_type = np.full(count, "NONE", dtype=object)
    triangle_quality = np.zeros(count)
    triangle_upper = np.full(count, np.nan)
    triangle_lower = np.full(count, np.nan)
    triangle_contraction = np.zeros(count)
    triangle_apex_bars = np.full(count, np.nan)

    for index in range(min(scales), count):
        current_atr = atr[index]
        if not np.isfinite(current_atr) or current_atr <= 0:
            continue
        reference_price = close[index - 1]
        resistance_candidates: list[tuple[float, float, float]] = []
        support_candidates: list[tuple[float, float, float]] = []

        for scale in scales:
            prefix = f"structure_{scale}h"
            start = max(0, index - scale)
            available_highs = _pivot_window(
                high_pivots,
                high_confirmations,
                start=start,
                stop=index - 1,
                limit=maximum_pivots,
            )
            available_lows = _pivot_window(
                low_pivots,
                low_confirmations,
                start=start,
                stop=index - 1,
                limit=maximum_pivots,
            )
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
            dynamic_high = high_line.value if high_line else np.nan
            dynamic_low = low_line.value if low_line else np.nan
            scale_resistance = _nearest_above(
                reference_price,
                (dynamic_high, static_highs[scale][index]),
            )
            scale_support = _nearest_below(
                reference_price,
                (dynamic_low, static_lows[scale][index]),
            )
            result[f"{prefix}_resistance"][index] = scale_resistance
            result[f"{prefix}_support"][index] = scale_support

            if high_line is not None:
                result[f"{prefix}_resistance_slope_atr"][index] = (
                    high_line.slope_atr
                )
                result[f"{prefix}_resistance_r2"][index] = high_line.r2
            if low_line is not None:
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
                resistance_candidates.append(
                    (scale_resistance, scale_weight, float(high_touches))
                )
            if np.isfinite(scale_support):
                support_candidates.append(
                    (scale_support, scale_weight, float(low_touches))
                )

        chosen_resistance = choose_composite_level(
            reference_price,
            resistance_candidates,
            direction=1,
        )
        chosen_support = choose_composite_level(
            reference_price,
            support_candidates,
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
            max_pivots=maximum_pivots,
            cfg=cfg,
        )
        if triangle is not None:
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
    output["triangle_code"] = (
        pd.Series(triangle_type)
        .map(
            {
                "NONE": 0,
                "SYMMETRICAL": 1,
                "ASCENDING": 2,
                "DESCENDING": -2,
            }
        )
        .fillna(0)
        .astype(int)
    )
    output["triangle_quality"] = triangle_quality
    output["triangle_upper"] = triangle_upper
    output["triangle_lower"] = triangle_lower
    output["triangle_contraction"] = triangle_contraction
    output["triangle_apex_bars"] = triangle_apex_bars
    output["triangle_width_atr"] = (
        triangle_upper - triangle_lower
    ) / np.where(atr > 0, atr, np.nan)

    event = detect_breakout_events(output, cfg)
    for column, values in event.items():
        output[column] = values
    return output


def _pivot_window(
    pivots: list[Pivot],
    confirmations: list[int],
    start: int,
    stop: int,
    limit: int,
) -> list[Pivot]:
    left = bisect_left(confirmations, start)
    right = bisect_right(confirmations, stop)
    left = max(left, right - limit)
    return pivots[left:right]


def _nearest_above(
    reference: float,
    values: tuple[float, ...],
) -> float:
    valid = [float(value) for value in values if np.isfinite(value)]
    if not valid:
        return np.nan
    above = [value for value in valid if value >= reference]
    return min(
        above,
        default=min(valid, key=lambda value: abs(value - reference)),
    )


def _nearest_below(
    reference: float,
    values: tuple[float, ...],
) -> float:
    valid = [float(value) for value in values if np.isfinite(value)]
    if not valid:
        return np.nan
    below = [value for value in valid if value <= reference]
    return max(
        below,
        default=min(valid, key=lambda value: abs(value - reference)),
    )
