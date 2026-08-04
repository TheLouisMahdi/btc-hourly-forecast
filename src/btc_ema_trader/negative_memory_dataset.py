from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import Settings
from .costs import execution_cost_breakdown
from .negative_memory_core import (
    SUPPORT, RESISTANCE, fingerprint, _first, _num, _per_horizon,
)

def mine_boundary_encounters(
    frame: pd.DataFrame,
    settings: Settings,
    horizons: Iterable[int],
) -> pd.DataFrame:
    cfg = settings.section("negative_memory")
    maximum = float(cfg.get("maximum_boundary_distance_atr", 0.85))
    minimum = float(cfg.get("minimum_boundary_distance_atr", -0.25))
    cooldown = int(cfg.get("encounter_cooldown_hours", 3))
    buffer = float(cfg.get("break_buffer_atr", 0.08))
    stress_cost = (
        execution_cost_breakdown(settings.section("strategy"))["stress_cost_bps"]
        + float(
            settings.section("strategy").get(
                "economic_execution_uncertainty_bps", 4.0
            )
        )
    ) / 10_000.0
    minimum_profit = float(
        settings.section("strategy").get("minimum_profit_buffer_bps", 8.0)
    ) / 10_000.0
    data = frame.sort_values("open_time").reset_index(drop=True)
    segment = pd.to_numeric(
        data.get("market_segment_id", pd.Series(0, index=data.index)),
        errors="coerce",
    ).fillna(0).to_numpy(dtype=int)
    last = {SUPPORT: -10_000, RESISTANCE: -10_000}
    records: list[dict[str, Any]] = []
    for index, row in data.iterrows():
        atr, close = _num(row.get("atr")), _num(row.get("close"))
        if atr is None or close is None or atr <= 0 or close <= 0:
            continue
        approach = _num(row.get("return_6"), 0.0) or 0.0
        candidates: list[tuple[str, float, float, float, float]] = []
        for side, level_key, distance_key, strength_key, age_key in (
            (
                SUPPORT, "structure_support", "distance_to_support_atr",
                "support_strength", "support_age_bars",
            ),
            (
                RESISTANCE, "structure_resistance",
                "distance_to_resistance_atr", "resistance_strength",
                "resistance_age_bars",
            ),
        ):
            level, distance = _num(row.get(level_key)), _num(row.get(distance_key))
            correct_approach = approach <= 0 if side == SUPPORT else approach >= 0
            if (
                level is not None
                and distance is not None
                and minimum <= distance <= maximum
                and correct_approach
            ):
                candidates.append(
                    (
                        side, level, distance,
                        _num(row.get(strength_key), 0.0) or 0.0,
                        _num(row.get(age_key), 0.0) or 0.0,
                    )
                )
        for side, level, distance, strength, age in candidates:
            if index - last[side] < cooldown:
                continue
            last[side] = index
            for horizon in horizons:
                horizon = int(horizon)
                path = data.iloc[index + 1 : index + horizon + 1]
                if (
                    len(path) != horizon
                    or np.any(
                        segment[index + 1 : index + horizon + 1]
                        != segment[index]
                    )
                ):
                    continue
                label = _label_path(
                    path, side, level, atr, horizon, buffer,
                    settings, stress_cost, minimum_profit,
                )
                if label is None:
                    continue
                item = row.to_dict()
                item.update(
                    {
                        "boundary_side": side,
                        "boundary_side_code": -1 if side == SUPPORT else 1,
                        "boundary_level": level,
                        "boundary_distance_atr": distance,
                        "boundary_strength": strength,
                        "boundary_age_bars": age,
                        "boundary_approach_return_6": approach,
                        "horizon": horizon,
                        **label,
                    }
                )
                item["boundary_fingerprint"] = fingerprint(
                    pd.Series(item), side, horizon
                )
                records.append(item)
    return pd.DataFrame(records)

def _label_path(
    path: pd.DataFrame,
    side: str,
    level: float,
    atr: float,
    horizon: int,
    buffer: float,
    settings: Settings,
    stress_cost: float,
    minimum_profit: float,
) -> dict[str, Any] | None:
    high = pd.to_numeric(path["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(path["low"], errors="coerce").to_numpy(float)
    close = pd.to_numeric(path["close"], errors="coerce").to_numpy(float)
    if not (np.isfinite(high).all() and np.isfinite(low).all() and np.isfinite(close).all()):
        return None
    cfg = settings.section(
        "short_breakdown" if side == SUPPORT else "long_breakout"
    )
    target_atr = _per_horizon(
        cfg.get("target_atr_by_horizon", {}), horizon, 1.0
    )
    invalidation_atr = float(cfg.get("invalidation_atr", 0.60))
    if side == SUPPORT:
        trigger, stop = level - buffer * atr, level + invalidation_atr * atr
        break_hits, opposite_hits = close <= trigger, high >= stop
    else:
        trigger, stop = level + buffer * atr, level - invalidation_atr * atr
        break_hits, opposite_hits = close >= trigger, low <= stop
    break_index, opposite_index = _first(break_hits), _first(opposite_hits)
    broke = break_index is not None and (
        opposite_index is None or break_index < opposite_index
    )
    gross, target_hit, stop_hit = 0.0, False, False
    if broke and break_index is not None:
        post_high, post_low, post_close = (
            high[break_index:], low[break_index:], close[break_index:]
        )
        if side == SUPPORT:
            target = trigger - target_atr * atr
            target_index, stop_index = _first(post_low <= target), _first(post_high >= stop)
            stop_hit = stop_index is not None and (
                target_index is None or stop_index <= target_index
            )
            target_hit = target_index is not None and not stop_hit
            gross = (
                (trigger - target) / trigger if target_hit
                else -(stop - trigger) / trigger if stop_hit
                else (trigger - post_close[-1]) / trigger
            )
        else:
            target = trigger + target_atr * atr
            target_index, stop_index = _first(post_high >= target), _first(post_low <= stop)
            stop_hit = stop_index is not None and (
                target_index is None or stop_index <= target_index
            )
            target_hit = target_index is not None and not stop_hit
            gross = (
                (target - trigger) / trigger if target_hit
                else -(trigger - stop) / trigger if stop_hit
                else (post_close[-1] - trigger) / trigger
            )
    net = gross - stress_cost if broke else 0.0
    profitable = bool(broke and target_hit and net >= minimum_profit)
    return {
        "break_label": int(broke),
        "profitable_label": int(profitable),
        "bad_label": int(not profitable),
        "gross_return": float(gross),
        "stress_net_return": float(net),
    }
