from __future__ import annotations

import math
from typing import Any

import numpy as np

from .trade_lifecycle import (
    TRADE_FEATURES as BASE_TRADE_FEATURES,
    trade_feature_vector as base_trade_feature_vector,
)

CONTEXT_TRADE_SCHEMA_VERSION = 2
CONTEXT_TRADE_FEATURES = (
    "event_body_scaled",
    "event_range_scaled",
    "event_upper_wick_share",
    "event_lower_wick_share",
    "event_close_location",
    "previous_1_body_scaled",
    "previous_1_close_location",
    "previous_2_body_scaled",
    "previous_2_close_location",
    "three_bar_net_return_scaled",
    "event_volume_vs_previous_log",
    "three_bar_wick_pressure",
)
EXTENDED_TRADE_FEATURES = BASE_TRADE_FEATURES + CONTEXT_TRADE_FEATURES


def context_trade_feature_vector(
    record: dict[str, Any],
    plan: dict[str, Any],
    *,
    base_stop_percent: float,
    base_reward_r: float,
    direction_code: float,
) -> np.ndarray:
    base = base_trade_feature_vector(
        record,
        plan,
        base_stop_percent=base_stop_percent,
        base_reward_r=base_reward_r,
        direction_code=direction_code,
    )
    context = record.get("event_candle_context")
    context = context if isinstance(context, dict) else {}
    bars = context.get("bars")
    bars = bars if isinstance(bars, list) else []
    by_role = {
        str(item.get("role")): item
        for item in bars
        if isinstance(item, dict)
    }
    event = by_role.get("EVENT", {})
    previous_1 = by_role.get("PREVIOUS_1", {})
    previous_2 = by_role.get("PREVIOUS_2", {})

    event_range = max(_finite(event.get("range")), 0.0)
    event_upper = max(_finite(event.get("upper_wick")), 0.0)
    event_lower = max(_finite(event.get("lower_wick")), 0.0)
    upper_share = event_upper / event_range if event_range > 0 else 0.0
    lower_share = event_lower / event_range if event_range > 0 else 0.0

    first_open = _finite(previous_2.get("open"))
    event_close = _finite(event.get("close"))
    three_bar_return = (
        event_close / first_open - 1.0
        if first_open > 0 and event_close > 0
        else 0.0
    )
    previous_volumes = [
        _finite(item.get("volume"))
        for item in (previous_2, previous_1)
        if _finite(item.get("volume")) > 0
    ]
    event_volume = _finite(event.get("volume"))
    previous_volume_mean = (
        sum(previous_volumes) / len(previous_volumes)
        if previous_volumes
        else 0.0
    )
    volume_log_ratio = (
        math.log1p(event_volume) - math.log1p(previous_volume_mean)
        if event_volume > 0 and previous_volume_mean > 0
        else 0.0
    )

    upper_total = sum(
        max(_finite(item.get("upper_wick")), 0.0)
        for item in (previous_2, previous_1, event)
    )
    lower_total = sum(
        max(_finite(item.get("lower_wick")), 0.0)
        for item in (previous_2, previous_1, event)
    )
    range_total = sum(
        max(_finite(item.get("range")), 0.0)
        for item in (previous_2, previous_1, event)
    )
    wick_pressure = (
        (lower_total - upper_total) / range_total
        if range_total > 0
        else 0.0
    )

    extra = np.asarray(
        [
            _scaled_percent(event.get("body_percent"), 0.02),
            _scaled_percent(event.get("range_percent"), 0.03),
            _clip(upper_share, 0.0, 1.0),
            _clip(lower_share, 0.0, 1.0),
            _clip(_finite(event.get("close_location"), 0.5), 0.0, 1.0),
            _scaled_percent(previous_1.get("body_percent"), 0.02),
            _clip(
                _finite(previous_1.get("close_location"), 0.5),
                0.0,
                1.0,
            ),
            _scaled_percent(previous_2.get("body_percent"), 0.02),
            _clip(
                _finite(previous_2.get("close_location"), 0.5),
                0.0,
                1.0,
            ),
            _clip(three_bar_return / 0.04, -3.0, 3.0),
            _clip(volume_log_ratio / 2.0, -3.0, 3.0),
            _clip(wick_pressure, -1.0, 1.0),
        ],
        dtype=float,
    )
    return np.concatenate([np.asarray(base, dtype=float), extra])


def install_context_trade_features(module: Any) -> None:
    """Install schema-v2 features into the existing adaptive lifecycle module."""
    module.TRADE_STATE_SCHEMA_VERSION = CONTEXT_TRADE_SCHEMA_VERSION
    module.TRADE_FEATURES = EXTENDED_TRADE_FEATURES
    module.trade_feature_vector = context_trade_feature_vector


def _scaled_percent(value: Any, scale: float) -> float:
    return _clip(_finite(value) / max(scale, 1e-9), -3.0, 3.0)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _clip(value: float, minimum: float, maximum: float) -> float:
    return float(np.clip(value, minimum, maximum))
