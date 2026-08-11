from __future__ import annotations

from typing import Iterable

import pandas as pd

from . import features as base_features
from .candle_context import (
    attach_causal_candle_context,
    candle_context_feature_columns,
)
from .config import Settings
from .directional_events import (
    attach_directional_breakout_candidates,
    attach_directional_event_labels,
)
from .features import FeatureSet
from .forecast_contract import attach_close_based_general_labels
from .market_structure_fast import (
    build_market_structure as build_fast_market_structure,
)
from .sample_policy import (
    apply_model_calendar,
    attach_sample_policy,
    invalidate_ineligible_labels,
    suppress_ineligible_events,
)

base_features.build_market_structure = build_fast_market_structure


def build_feature_set(
    candles: pd.DataFrame,
    news: pd.DataFrame,
    settings: Settings,
    include_labels: bool = True,
) -> FeatureSet:
    model_candles = apply_model_calendar(candles)
    if model_candles.empty:
        raise ValueError("No UTC candles are available")

    base = base_features.build_feature_set(
        model_candles,
        news,
        settings,
        include_labels=False,
    )
    contextual_frame = attach_causal_candle_context(base.frame)
    contextual_frame = attach_sample_policy(contextual_frame, settings)
    frame = attach_directional_breakout_candidates(
        contextual_frame,
        settings,
    )
    frame = suppress_ineligible_events(frame)
    horizons = _configured_horizons(settings, base.horizons)
    if include_labels:
        frame = attach_close_based_general_labels(frame, horizons)
        frame = attach_directional_event_labels(
            frame,
            settings,
            horizons,
        )
        frame = invalidate_ineligible_labels(frame, horizons)

    label_prefixes = (
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
        "model_path_eligible_",
    )
    feature_columns = [
        column
        for column in base.feature_columns
        if column in frame
        and pd.api.types.is_numeric_dtype(frame[column])
        and not column.startswith(label_prefixes)
    ]
    for column in candle_context_feature_columns(frame):
        if column not in feature_columns:
            feature_columns.append(column)
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
        "model_sample_weight_multiplier",
    ):
        if (
            column in frame
            and pd.api.types.is_numeric_dtype(frame[column])
            and column not in feature_columns
        ):
            feature_columns.append(column)
    return FeatureSet(
        frame=frame,
        feature_columns=feature_columns,
        horizons=horizons,
    )


def _configured_horizons(
    settings: Settings,
    fallback: Iterable[int],
) -> list[int]:
    configured = settings.section("model").get(
        "horizons_hours",
        list(fallback),
    )
    horizons = sorted({int(value) for value in configured})
    if 1 not in horizons:
        horizons.insert(0, 1)
    return horizons
