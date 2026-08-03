from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Settings
from .costs import execution_cost_breakdown

LONG = 1
SHORT = -1


@dataclass(frozen=True)
class BreakoutCandidate:
    direction: int
    event_type: str
    source: str
    scale_hours: int
    level: float
    invalidation_level: float
    distance_atr: float
    touches: float
    age_bars: float
    line_slope_atr: float
    line_r2: float
    score: float


def attach_directional_breakout_candidates(
    frame: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    output = frame.copy().sort_values("open_time").reset_index(drop=True)
    mining = settings.section("event_mining")
    long_cfg = settings.section("long_breakout")
    short_cfg = settings.section("short_breakdown")
    scales = [
        int(value)
        for value in mining.get(
            "structure_scales_hours",
            [24, 48, 96, 168, 336, 720],
        )
    ]
    minimum_separation = int(
        mining.get("minimum_event_separation_hours", 2)
    )
    similarity_atr = float(
        mining.get("duplicate_level_similarity_atr", 0.35)
    )

    count = len(output)
    event_direction = np.zeros(count, dtype=int)
    event_type = np.full(count, "NONE", dtype=object)
    event_source = np.full(count, "NONE", dtype=object)
    event_id = np.full(count, None, dtype=object)
    event_score = np.zeros(count, dtype=float)
    event_scale = np.zeros(count, dtype=int)
    breakout_level = np.full(count, np.nan)
    invalidation_level = np.full(count, np.nan)
    breakout_distance = np.full(count, np.nan)
    level_touches = np.zeros(count, dtype=float)
    level_age = np.full(count, np.nan)
    line_slope = np.full(count, np.nan)
    line_r2 = np.full(count, np.nan)
    diversity_key = np.full(count, None, dtype=object)

    close = pd.to_numeric(output["close"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(output["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(output["low"], errors="coerce").to_numpy(float)
    atr = pd.to_numeric(output["atr"], errors="coerce").to_numpy(float)
    body_atr = pd.to_numeric(
        output["body_atr"], errors="coerce"
    ).to_numpy(float)
    close_location = pd.to_numeric(
        output["close_location"], errors="coerce"
    ).to_numpy(float)
    volume_z = pd.to_numeric(
        output.get("volume_z_24", 0.0), errors="coerce"
    ).fillna(0.0).to_numpy(float)

    last_event: dict[int, tuple[int, float, float]] = {}
    sequence = 0
    for index in range(1, count):
        if not _finite_positive(atr[index]):
            continue
        candidates: list[BreakoutCandidate] = []
        for scale in scales:
            candidates.extend(
                _scale_candidates(
                    output=output,
                    index=index,
                    scale=scale,
                    previous_close=close[index - 1],
                    current_close=close[index],
                    current_high=high[index],
                    current_low=low[index],
                    current_atr=atr[index],
                    body_atr=body_atr[index],
                    close_location=close_location[index],
                    volume_z=volume_z[index],
                    long_cfg=long_cfg,
                    short_cfg=short_cfg,
                    high=high,
                    low=low,
                )
            )
        candidates.extend(
            _triangle_candidates(
                output=output,
                index=index,
                previous_close=close[index - 1],
                current_close=close[index],
                current_atr=atr[index],
                body_atr=body_atr[index],
                close_location=close_location[index],
                volume_z=volume_z[index],
                long_cfg=long_cfg,
                short_cfg=short_cfg,
            )
        )
        if not candidates:
            continue

        selected = max(
            candidates,
            key=lambda item: (
                item.score,
                item.touches,
                -abs(item.distance_atr),
                -item.scale_hours,
            ),
        )
        previous = last_event.get(selected.direction)
        if previous is not None:
            previous_index, previous_level, previous_atr = previous
            close_in_time = index - previous_index <= minimum_separation
            level_distance = abs(selected.level - previous_level) / max(
                atr[index],
                previous_atr,
                1e-12,
            )
            if close_in_time and level_distance <= similarity_atr:
                continue

        sequence += 1
        timestamp = pd.Timestamp(output.iloc[index]["open_time"])
        event_direction[index] = selected.direction
        event_type[index] = selected.event_type
        event_source[index] = selected.source
        event_score[index] = selected.score
        event_scale[index] = selected.scale_hours
        breakout_level[index] = selected.level
        invalidation_level[index] = selected.invalidation_level
        breakout_distance[index] = selected.distance_atr
        level_touches[index] = selected.touches
        level_age[index] = selected.age_bars
        line_slope[index] = selected.line_slope_atr
        line_r2[index] = selected.line_r2
        direction_name = "LONG" if selected.direction == LONG else "SHORT"
        event_id[index] = (
            f"V5-{timestamp.strftime('%Y%m%dT%H%MZ')}-"
            f"{direction_name}-{selected.source}"
        )
        diversity_key[index] = _diversity_key(
            timestamp=timestamp,
            candidate=selected,
            row=output.iloc[index],
        )
        last_event[selected.direction] = (
            index,
            selected.level,
            atr[index],
        )

    output["event_direction"] = event_direction
    output["event_type"] = event_type
    output["breakout_source"] = event_source
    output["event_id"] = event_id
    output["event_score"] = event_score
    output["event_scale_hours"] = event_scale
    output["breakout_level"] = breakout_level
    output["breakout_invalidation_level"] = invalidation_level
    output["breakout_distance_atr"] = breakout_distance
    output["breakout_level_touches"] = level_touches
    output["breakout_level_age_bars"] = level_age
    output["breakout_line_slope_atr"] = line_slope
    output["breakout_line_r2"] = line_r2
    output["event_diversity_key"] = diversity_key
    output["breakout_confirmed"] = (event_direction != 0).astype(int)
    output["is_event"] = (event_direction != 0).astype(int)
    output["aligned_body_atr"] = body_atr * event_direction
    output["aligned_close_quality"] = np.where(
        event_direction == LONG,
        close_location,
        np.where(event_direction == SHORT, 1.0 - close_location, 0.5),
    )
    output["aligned_regime"] = (
        pd.to_numeric(output.get("regime_code", 0), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        * event_direction
    )
    output["aligned_rsi"] = (
        pd.to_numeric(output.get("rsi_centered", 0), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        * event_direction
    )
    output["aligned_ema168_slope"] = (
        pd.to_numeric(
            output.get("ema_168_slope_6", 0), errors="coerce"
        )
        .fillna(0.0)
        .to_numpy(float)
        * event_direction
    )
    output["bars_since_event"] = _bars_since_event(event_direction)
    output["active_event_direction"] = _active_direction(event_direction)
    output["event_continuation_bias"] = (
        pd.to_numeric(output.get("regime_code", 0), errors="coerce")
        .fillna(0.0)
        .to_numpy(float)
        * event_direction
    )
    return output


def attach_directional_event_labels(
    frame: pd.DataFrame,
    settings: Settings,
    horizons: Iterable[int],
) -> pd.DataFrame:
    output = frame.copy()
    long_cfg = settings.section("long_breakout")
    short_cfg = settings.section("short_breakdown")
    strategy_cfg = settings.section("strategy")
    costs = execution_cost_breakdown(strategy_cfg)
    base_cost = float(costs["base_cost_bps"]) / 10_000.0
    profit_buffer = float(costs["profit_buffer_bps"]) / 10_000.0

    for raw_horizon in horizons:
        horizon = int(raw_horizon)
        columns = {
            "hold": np.full(len(output), np.nan),
            "success": np.full(len(output), np.nan),
            "false": np.full(len(output), np.nan),
            "neutral": np.full(len(output), np.nan),
            "tradeable": np.full(len(output), np.nan),
            "gross": np.full(len(output), np.nan),
            "net": np.full(len(output), np.nan),
            "mfe": np.full(len(output), np.nan),
            "mae": np.full(len(output), np.nan),
            "hold_ratio": np.full(len(output), np.nan),
            "target_first": np.full(len(output), np.nan),
            "stop_first": np.full(len(output), np.nan),
        }
        for index in range(len(output) - horizon):
            direction = int(output.iloc[index].get("event_direction", 0))
            if direction not in {LONG, SHORT}:
                continue
            cfg = long_cfg if direction == LONG else short_cfg
            label = _label_event(
                output=output,
                index=index,
                horizon=horizon,
                direction=direction,
                cfg=cfg,
                base_cost=base_cost,
                profit_buffer=profit_buffer,
            )
            for key, value in label.items():
                columns[key][index] = value

        output[f"breakout_hold_h{horizon}"] = columns["hold"]
        output[f"breakout_success_h{horizon}"] = columns["success"]
        output[f"event_continuation_h{horizon}"] = columns["success"]
        output[f"false_breakout_h{horizon}"] = columns["false"]
        output[f"neutral_breakout_h{horizon}"] = columns["neutral"]
        output[f"tradeable_h{horizon}"] = columns["tradeable"]
        output[f"event_gross_return_h{horizon}"] = columns["gross"]
        output[f"event_net_return_h{horizon}"] = columns["net"]
        output[f"event_mfe_atr_h{horizon}"] = columns["mfe"]
        output[f"event_mae_atr_h{horizon}"] = columns["mae"]
        output[f"event_hold_ratio_h{horizon}"] = columns["hold_ratio"]
        output[f"event_target_first_h{horizon}"] = columns[
            "target_first"
        ]
        output[f"event_stop_first_h{horizon}"] = columns["stop_first"]
    return output


def event_inventory(
    frame: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    inventory_cfg = settings.section("event_inventory")
    events = (
        frame.loc[frame["event_direction"].isin([LONG, SHORT])]
        .copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    if events.empty:
        raise ValueError("No deterministic breakout events were mined")
    if events["event_id"].isna().any() or events["event_id"].duplicated().any():
        raise ValueError("Breakout event IDs must be unique and non-null")
    if events.duplicated(["open_time", "event_direction"]).any():
        raise ValueError(
            "Only one unique event per timestamp and direction is allowed"
        )

    minimum_per_direction = int(
        inventory_cfg.get("minimum_events_per_direction", 2000)
    )
    direction_report: dict[str, Any] = {}
    blockers: list[str] = []
    for direction, name in ((LONG, "LONG"), (SHORT, "SHORT")):
        subset = events.loc[events["event_direction"] == direction].copy()
        timestamps = pd.to_datetime(subset["open_time"], utc=True)
        years = int(timestamps.dt.year.nunique())
        quarters = int(
            timestamps.dt.tz_localize(None).dt.to_period("Q").nunique()
        )
        scales = int(
            pd.to_numeric(
                subset["event_scale_hours"], errors="coerce"
            ).dropna().nunique()
        )
        diversity_keys = int(subset["event_diversity_key"].nunique())
        volatility_buckets = int(
            subset.apply(_volatility_bucket_row, axis=1).nunique()
        )
        regimes = int(subset["regime"].astype(str).nunique())
        sources = subset["breakout_source"].value_counts().to_dict()
        count = int(len(subset))
        direction_report[name] = {
            "events": count,
            "years": years,
            "quarters": quarters,
            "structure_scales": scales,
            "diversity_keys": diversity_keys,
            "volatility_buckets": volatility_buckets,
            "regimes": regimes,
            "sources": sources,
            "first": timestamps.min().isoformat() if count else None,
            "last": timestamps.max().isoformat() if count else None,
        }
        checks = {
            "events": (
                count,
                minimum_per_direction,
                "unique events",
            ),
            "years": (
                years,
                int(inventory_cfg.get("minimum_years", 6)),
                "calendar years",
            ),
            "quarters": (
                quarters,
                int(inventory_cfg.get("minimum_quarters", 24)),
                "calendar quarters",
            ),
            "structure_scales": (
                scales,
                int(inventory_cfg.get("minimum_structure_scales", 4)),
                "structure scales",
            ),
            "diversity_keys": (
                diversity_keys,
                int(inventory_cfg.get("minimum_diversity_keys", 48)),
                "diversity groups",
            ),
            "volatility_buckets": (
                volatility_buckets,
                int(inventory_cfg.get("minimum_volatility_buckets", 3)),
                "volatility buckets",
            ),
            "regimes": (
                regimes,
                int(inventory_cfg.get("minimum_regimes", 3)),
                "market regimes",
            ),
        }
        for actual, required, label in checks.values():
            if actual < required:
                blockers.append(
                    f"{name} has {actual} {label}; at least {required} are required"
                )

    if blockers:
        raise ValueError("Event inventory gate failed: " + "; ".join(blockers))

    report = {
        "sampling_strategy": "NONE",
        "split_strategy": "CHRONOLOGICAL_EXPANDING_WINDOW",
        "synthetic_events": 0,
        "duplicated_events": 0,
        "retained_events": int(len(events)),
        "minimum_events_per_direction": minimum_per_direction,
        "directions": direction_report,
    }
    return events, report


def event_feature_columns(
    frame: pd.DataFrame,
    general_feature_columns: Iterable[str],
) -> list[str]:
    excluded_exact = {
        "volume",
        "quote_volume",
        "trades",
        "news_available",
        "news_age_hours",
        "is_event",
        "breakout_confirmed",
        "event_direction",
        "active_event_direction",
    }
    excluded_prefixes = (
        "news_",
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
    )
    columns: list[str] = []
    for column in general_feature_columns:
        if column in excluded_exact or column.startswith(excluded_prefixes):
            continue
        if column not in frame or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        columns.append(column)
    for column in (
        "event_score",
        "event_scale_hours",
        "breakout_distance_atr",
        "breakout_level_touches",
        "breakout_level_age_bars",
        "breakout_line_slope_atr",
        "breakout_line_r2",
        "aligned_body_atr",
        "aligned_close_quality",
        "aligned_regime",
        "aligned_rsi",
        "aligned_ema168_slope",
    ):
        if column in frame and column not in columns:
            columns.append(column)
    return columns


def _scale_candidates(
    output: pd.DataFrame,
    index: int,
    scale: int,
    previous_close: float,
    current_close: float,
    current_high: float,
    current_low: float,
    current_atr: float,
    body_atr: float,
    close_location: float,
    volume_z: float,
    long_cfg: dict[str, Any],
    short_cfg: dict[str, Any],
    high: np.ndarray,
    low: np.ndarray,
) -> list[BreakoutCandidate]:
    prefix = f"structure_{scale}h"
    resistance = _number(output.iloc[index].get(f"{prefix}_resistance"))
    support = _number(output.iloc[index].get(f"{prefix}_support"))
    resistance_touches = _number(
        output.iloc[index].get(f"{prefix}_resistance_touches"), 0.0
    )
    support_touches = _number(
        output.iloc[index].get(f"{prefix}_support_touches"), 0.0
    )
    resistance_slope = _number(
        output.iloc[index].get(f"{prefix}_resistance_slope_atr"), 0.0
    )
    support_slope = _number(
        output.iloc[index].get(f"{prefix}_support_slope_atr"), 0.0
    )
    resistance_r2 = _number(
        output.iloc[index].get(f"{prefix}_resistance_r2"), 0.5
    )
    support_r2 = _number(
        output.iloc[index].get(f"{prefix}_support_r2"), 0.5
    )
    candidates: list[BreakoutCandidate] = []

    if resistance is not None:
        age = _last_touch_age(
            high,
            resistance,
            current_atr * float(long_cfg.get("touch_tolerance_atr", 0.35)),
            index,
            scale,
        )
        candidate = _long_candidate(
            previous_close=previous_close,
            current_close=current_close,
            current_high=current_high,
            current_atr=current_atr,
            body_atr=body_atr,
            close_location=close_location,
            volume_z=volume_z,
            level=resistance,
            touches=float(resistance_touches or 0.0),
            age_bars=age,
            line_slope_atr=float(resistance_slope or 0.0),
            line_r2=float(resistance_r2 or 0.0),
            scale=scale,
            source=f"RESISTANCE_{scale}H",
            cfg=long_cfg,
            row=output.iloc[index],
        )
        if candidate is not None:
            candidates.append(candidate)

    if support is not None:
        age = _last_touch_age(
            low,
            support,
            current_atr * float(short_cfg.get("touch_tolerance_atr", 0.35)),
            index,
            scale,
        )
        candidate = _short_candidate(
            previous_close=previous_close,
            current_close=current_close,
            current_low=current_low,
            current_atr=current_atr,
            body_atr=body_atr,
            close_location=close_location,
            volume_z=volume_z,
            level=support,
            touches=float(support_touches or 0.0),
            age_bars=age,
            line_slope_atr=float(support_slope or 0.0),
            line_r2=float(support_r2 or 0.0),
            scale=scale,
            source=f"SUPPORT_{scale}H",
            cfg=short_cfg,
            row=output.iloc[index],
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _triangle_candidates(
    output: pd.DataFrame,
    index: int,
    previous_close: float,
    current_close: float,
    current_atr: float,
    body_atr: float,
    close_location: float,
    volume_z: float,
    long_cfg: dict[str, Any],
    short_cfg: dict[str, Any],
) -> list[BreakoutCandidate]:
    row = output.iloc[index]
    triangle_type = str(row.get("triangle_type", "NONE"))
    quality = _number(row.get("triangle_quality"), 0.0) or 0.0
    minimum_quality = min(
        float(long_cfg.get("minimum_triangle_quality", 0.25)),
        float(short_cfg.get("minimum_triangle_quality", 0.25)),
    )
    if triangle_type == "NONE" or quality < minimum_quality:
        return []
    scale = 240
    candidates: list[BreakoutCandidate] = []
    upper = _number(row.get("triangle_upper"))
    lower = _number(row.get("triangle_lower"))
    if upper is not None and triangle_type in {"SYMMETRICAL", "ASCENDING"}:
        candidate = _long_candidate(
            previous_close=previous_close,
            current_close=current_close,
            current_high=current_close,
            current_atr=current_atr,
            body_atr=body_atr,
            close_location=close_location,
            volume_z=volume_z,
            level=upper,
            touches=2.0 + quality * 4.0,
            age_bars=float(scale),
            line_slope_atr=0.0,
            line_r2=quality,
            scale=scale,
            source="TRIANGLE_UPPER",
            cfg=long_cfg,
            row=row,
            triangle_quality=quality,
        )
        if candidate is not None:
            candidates.append(candidate)
    if lower is not None and triangle_type in {"SYMMETRICAL", "DESCENDING"}:
        candidate = _short_candidate(
            previous_close=previous_close,
            current_close=current_close,
            current_low=current_close,
            current_atr=current_atr,
            body_atr=body_atr,
            close_location=close_location,
            volume_z=volume_z,
            level=lower,
            touches=2.0 + quality * 4.0,
            age_bars=float(scale),
            line_slope_atr=0.0,
            line_r2=quality,
            scale=scale,
            source="TRIANGLE_LOWER",
            cfg=short_cfg,
            row=row,
            triangle_quality=quality,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _long_candidate(
    previous_close: float,
    current_close: float,
    current_high: float,
    current_atr: float,
    body_atr: float,
    close_location: float,
    volume_z: float,
    level: float,
    touches: float,
    age_bars: float,
    line_slope_atr: float,
    line_r2: float,
    scale: int,
    source: str,
    cfg: dict[str, Any],
    row: pd.Series,
    triangle_quality: float = 0.0,
) -> BreakoutCandidate | None:
    del current_high
    tolerance = float(cfg.get("crossing_tolerance_atr", 0.08))
    minimum_cross = float(cfg.get("candidate_minimum_cross_atr", 0.015))
    maximum_extension = float(cfg.get("candidate_maximum_extension_atr", 2.20))
    distance = (current_close - level) / current_atr
    if not (
        previous_close <= level + tolerance * current_atr
        and minimum_cross <= distance <= maximum_extension
        and body_atr >= float(cfg.get("candidate_minimum_body_atr", 0.04))
        and close_location
        >= float(cfg.get("candidate_minimum_close_location", 0.52))
        and volume_z >= float(cfg.get("candidate_minimum_volume_z", -1.50))
        and touches >= float(cfg.get("candidate_minimum_touches", 1.0))
    ):
        return None
    trend = _long_trend_quality(row)
    score = float(
        np.clip(
            0.23 * np.clip(touches / 5.0, 0.0, 1.0)
            + 0.18 * np.clip(body_atr / 1.40, 0.0, 1.0)
            + 0.17
            * np.clip(
                (close_location - 0.50) / 0.50,
                0.0,
                1.0,
            )
            + 0.10 * np.clip((volume_z + 1.5) / 4.0, 0.0, 1.0)
            + 0.10 * np.clip(line_r2, 0.0, 1.0)
            + 0.08
            * np.clip(
                1.0 - distance / maximum_extension,
                0.0,
                1.0,
            )
            + 0.07 * trend
            + 0.07 * np.clip(triangle_quality, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    invalidation = level - current_atr * float(
        cfg.get("invalidation_atr", 0.55)
    )
    return BreakoutCandidate(
        direction=LONG,
        event_type=(
            "TRIANGLE_BREAKOUT_LONG"
            if source == "TRIANGLE_UPPER"
            else "RESISTANCE_BREAKOUT_LONG"
        ),
        source=source,
        scale_hours=scale,
        level=level,
        invalidation_level=invalidation,
        distance_atr=distance,
        touches=touches,
        age_bars=age_bars,
        line_slope_atr=line_slope_atr,
        line_r2=line_r2,
        score=score,
    )


def _short_candidate(
    previous_close: float,
    current_close: float,
    current_low: float,
    current_atr: float,
    body_atr: float,
    close_location: float,
    volume_z: float,
    level: float,
    touches: float,
    age_bars: float,
    line_slope_atr: float,
    line_r2: float,
    scale: int,
    source: str,
    cfg: dict[str, Any],
    row: pd.Series,
    triangle_quality: float = 0.0,
) -> BreakoutCandidate | None:
    del current_low
    tolerance = float(cfg.get("crossing_tolerance_atr", 0.10))
    minimum_cross = float(cfg.get("candidate_minimum_cross_atr", 0.020))
    maximum_extension = float(cfg.get("candidate_maximum_extension_atr", 2.40))
    distance = (level - current_close) / current_atr
    if not (
        previous_close >= level - tolerance * current_atr
        and minimum_cross <= distance <= maximum_extension
        and body_atr <= -float(cfg.get("candidate_minimum_body_atr", 0.05))
        and close_location
        <= float(cfg.get("candidate_maximum_close_location", 0.48))
        and volume_z >= float(cfg.get("candidate_minimum_volume_z", -1.30))
        and touches >= float(cfg.get("candidate_minimum_touches", 1.0))
    ):
        return None
    trend = _short_trend_quality(row)
    score = float(
        np.clip(
            0.20 * np.clip(touches / 5.0, 0.0, 1.0)
            + 0.22 * np.clip(abs(body_atr) / 1.60, 0.0, 1.0)
            + 0.20
            * np.clip(
                (0.50 - close_location) / 0.50,
                0.0,
                1.0,
            )
            + 0.13 * np.clip((volume_z + 1.3) / 4.0, 0.0, 1.0)
            + 0.08 * np.clip(line_r2, 0.0, 1.0)
            + 0.07
            * np.clip(
                1.0 - distance / maximum_extension,
                0.0,
                1.0,
            )
            + 0.06 * trend
            + 0.04 * np.clip(triangle_quality, 0.0, 1.0),
            0.0,
            1.0,
        )
    )
    invalidation = level + current_atr * float(
        cfg.get("invalidation_atr", 0.65)
    )
    return BreakoutCandidate(
        direction=SHORT,
        event_type=(
            "TRIANGLE_BREAKDOWN_SHORT"
            if source == "TRIANGLE_LOWER"
            else "SUPPORT_BREAKDOWN_SHORT"
        ),
        source=source,
        scale_hours=scale,
        level=level,
        invalidation_level=invalidation,
        distance_atr=distance,
        touches=touches,
        age_bars=age_bars,
        line_slope_atr=line_slope_atr,
        line_r2=line_r2,
        score=score,
    )


def _label_event(
    output: pd.DataFrame,
    index: int,
    horizon: int,
    direction: int,
    cfg: dict[str, Any],
    base_cost: float,
    profit_buffer: float,
) -> dict[str, float]:
    entry = _number(output.iloc[index + 1].get("open"))
    atr = _number(output.iloc[index].get("atr"))
    level = _number(output.iloc[index].get("breakout_level"))
    invalidation = _number(
        output.iloc[index].get("breakout_invalidation_level")
    )
    if not all(
        value is not None and np.isfinite(value)
        for value in (entry, atr, level, invalidation)
    ):
        return _empty_label()
    assert entry is not None
    assert atr is not None
    assert level is not None
    assert invalidation is not None
    if entry <= 0 or atr <= 0:
        return _empty_label()

    path = output.iloc[index + 1 : index + horizon + 1]
    highs = pd.to_numeric(path["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(path["low"], errors="coerce").to_numpy(float)
    closes = pd.to_numeric(path["close"], errors="coerce").to_numpy(float)
    if len(path) != horizon or not (
        np.isfinite(highs).all()
        and np.isfinite(lows).all()
        and np.isfinite(closes).all()
    ):
        return _empty_label()

    target_atr = _per_horizon(
        cfg.get("target_atr_by_horizon", {}),
        horizon,
        1.0,
    )
    minimum_hold_ratio = _per_horizon(
        cfg.get("minimum_hold_ratio_by_horizon", {}),
        horizon,
        0.45,
    )
    hold_buffer = atr * float(cfg.get("label_hold_buffer_atr", 0.04))
    if direction == LONG:
        target_price = entry + target_atr * atr
        target_hits = highs >= target_price
        stop_hits = lows <= invalidation
        aligned_closes = closes - level
        favorable = (highs - entry) / atr
        adverse = (entry - lows) / atr
        aligned_return = closes[-1] / entry - 1.0
    else:
        target_price = entry - target_atr * atr
        target_hits = lows <= target_price
        stop_hits = highs >= invalidation
        aligned_closes = level - closes
        favorable = (entry - lows) / atr
        adverse = (highs - entry) / atr
        aligned_return = (entry - closes[-1]) / entry

    target_index = _first_true(target_hits)
    stop_index = _first_true(stop_hits)
    stop_first = stop_index is not None and (
        target_index is None or stop_index <= target_index
    )
    target_first = target_index is not None and not stop_first
    hold_ratio = float(np.mean(aligned_closes > hold_buffer))
    final_holds = bool(aligned_closes[-1] > hold_buffer)
    success = bool(
        target_first
        and final_holds
        and hold_ratio >= minimum_hold_ratio
        and aligned_return > 0
    )
    reentered = bool(aligned_closes[-1] <= 0)
    false_breakout = bool(stop_first or reentered)
    neutral = bool(not success and not false_breakout)
    gross_return = (
        -abs(entry - invalidation) / entry
        if stop_first
        else abs(target_price - entry) / entry
        if target_first
        else aligned_return
    )
    net_return = gross_return - base_cost
    tradeable = bool(success and net_return >= profit_buffer)
    return {
        "hold": float(final_holds),
        "success": float(success),
        "false": float(false_breakout),
        "neutral": float(neutral),
        "tradeable": float(tradeable),
        "gross": float(gross_return),
        "net": float(net_return),
        "mfe": float(np.max(favorable)),
        "mae": float(np.max(adverse)),
        "hold_ratio": hold_ratio,
        "target_first": float(target_first),
        "stop_first": float(stop_first),
    }


def _empty_label() -> dict[str, float]:
    return {
        key: np.nan
        for key in (
            "hold",
            "success",
            "false",
            "neutral",
            "tradeable",
            "gross",
            "net",
            "mfe",
            "mae",
            "hold_ratio",
            "target_first",
            "stop_first",
        )
    }


def _long_trend_quality(row: pd.Series) -> float:
    ema_slope = _number(row.get("ema_168_slope_6"), 0.0) or 0.0
    price_vs_ema = _number(row.get("price_vs_ema_168"), 0.0) or 0.0
    adx = _number(row.get("adx"), 0.0) or 0.0
    return float(
        np.clip(
            0.40 * np.clip((ema_slope + 0.001) / 0.004, 0.0, 1.0)
            + 0.35 * np.clip((price_vs_ema + 0.02) / 0.08, 0.0, 1.0)
            + 0.25 * np.clip(adx / 40.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    )


def _short_trend_quality(row: pd.Series) -> float:
    ema_slope = _number(row.get("ema_168_slope_6"), 0.0) or 0.0
    price_vs_ema = _number(row.get("price_vs_ema_168"), 0.0) or 0.0
    adx = _number(row.get("adx"), 0.0) or 0.0
    return float(
        np.clip(
            0.35 * np.clip((-ema_slope + 0.001) / 0.004, 0.0, 1.0)
            + 0.40 * np.clip((-price_vs_ema + 0.02) / 0.08, 0.0, 1.0)
            + 0.25 * np.clip(adx / 40.0, 0.0, 1.0),
            0.0,
            1.0,
        )
    )


def _diversity_key(
    timestamp: pd.Timestamp,
    candidate: BreakoutCandidate,
    row: pd.Series,
) -> str:
    regime = str(row.get("regime", "UNKNOWN"))
    volatility = _volatility_bucket_row(row)
    slope = (
        "UP"
        if candidate.line_slope_atr > 0.015
        else "DOWN"
        if candidate.line_slope_atr < -0.015
        else "FLAT"
    )
    touches = (
        "1"
        if candidate.touches < 2
        else "2"
        if candidate.touches < 3
        else "3PLUS"
    )
    quarter = f"{timestamp.year}Q{timestamp.quarter}"
    return "|".join(
        (
            quarter,
            "LONG" if candidate.direction == LONG else "SHORT",
            candidate.source,
            str(candidate.scale_hours),
            volatility,
            regime,
            slope,
            touches,
        )
    )


def _volatility_bucket_row(row: pd.Series) -> str:
    value = _number(row.get("atr_pct"), 0.0) or 0.0
    if value < 0.004:
        return "VERY_LOW"
    if value < 0.008:
        return "LOW"
    if value < 0.014:
        return "MEDIUM"
    if value < 0.024:
        return "HIGH"
    return "EXTREME"


def _last_touch_age(
    values: np.ndarray,
    level: float,
    tolerance: float,
    index: int,
    lookback: int,
) -> float:
    start = max(0, index - lookback)
    for position in range(index - 1, start - 1, -1):
        if np.isfinite(values[position]) and abs(values[position] - level) <= tolerance:
            return float(index - position)
    return float(lookback)


def _bars_since_event(direction: np.ndarray) -> np.ndarray:
    result = np.full(len(direction), np.nan)
    last: int | None = None
    for index, value in enumerate(direction):
        if value != 0:
            last = index
        if last is not None:
            result[index] = float(index - last)
    return result


def _active_direction(direction: np.ndarray) -> np.ndarray:
    result = np.zeros(len(direction), dtype=int)
    active = 0
    for index, value in enumerate(direction):
        if value != 0:
            active = int(value)
        result[index] = active
    return result


def _first_true(values: np.ndarray) -> int | None:
    positions = np.flatnonzero(values)
    return int(positions[0]) if len(positions) else None


def _per_horizon(value: Any, horizon: int, default: float) -> float:
    if isinstance(value, dict):
        return float(value.get(horizon, value.get(str(horizon), default)))
    return float(default if value is None else value)


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)
