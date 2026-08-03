from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from .config import Settings
from .costs import execution_cost_breakdown
from .directional_events import (
    LONG,
    SHORT,
    event_feature_columns,
    event_inventory,
)
from .features import FeatureSet, sample_weights
from .model import HourlyModelBundle, build_horizon_model

_ABSOLUTE_STRUCTURE_LEVEL = re.compile(
    r"^structure_\d+h_(resistance|support)$"
)


def train_feature_set(
    settings: Settings,
    feature_set: FeatureSet,
    provider: str,
    symbol: str,
) -> dict[str, Any]:
    frame = (
        feature_set.frame.copy()
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    horizons = [int(value) for value in feature_set.horizons]
    trade_horizons = [
        int(value)
        for value in settings.section("model").get(
            "trade_horizons_hours",
            [3, 6, 12],
        )
    ]
    general_feature_columns = [
        column
        for column in feature_set.feature_columns
        if not _is_absolute_price_feature(column)
        and column in frame
        and pd.api.types.is_numeric_dtype(frame[column])
        and frame[column].notna().sum() > max(250, len(frame) * 0.08)
    ]
    if "event_direction" not in general_feature_columns:
        general_feature_columns.append("event_direction")
    directional_feature_columns = event_feature_columns(
        frame,
        general_feature_columns,
    )
    minimum_rows = int(
        settings.section("model").get("min_train_rows", 60_000)
    )
    if len(frame) < minimum_rows:
        raise ValueError(
            f"Only {len(frame)} chronological rows are available; "
            f"at least {minimum_rows} are required"
        )

    events, inventory_report = event_inventory(frame, settings)
    general_weights = sample_weights(frame, settings)
    models: dict[int, Any] = {}
    metrics: dict[str, Any] = {}
    oof_records: list[pd.DataFrame] = []
    model_cfg = settings.section("model")
    general_splits = int(model_cfg.get("general_walk_forward_splits", 4))
    event_splits = int(model_cfg.get("event_walk_forward_splits", 6))
    embargo_hours = int(model_cfg.get("validation_gap_hours", 12))

    for horizon in horizons:
        horizon_metrics: dict[str, Any] = {
            "horizon": horizon,
            "directions": {},
        }
        general_oof = _general_walk_forward(
            frame=frame,
            horizon=horizon,
            settings=settings,
            feature_columns=general_feature_columns,
            event_feature_columns=directional_feature_columns,
            sample_weight=general_weights,
            splits=general_splits,
        )
        horizon_metrics.update(general_oof["metrics"])
        oof_records.append(general_oof["records"])

        if horizon in trade_horizons:
            for direction, direction_name in (
                (LONG, "LONG"),
                (SHORT, "SHORT"),
            ):
                direction_oof = _direction_walk_forward(
                    events=events,
                    horizon=horizon,
                    direction=direction,
                    direction_name=direction_name,
                    settings=settings,
                    general_feature_columns=general_feature_columns,
                    event_feature_columns=directional_feature_columns,
                    splits=event_splits,
                    embargo_hours=embargo_hours,
                )
                horizon_metrics["directions"][direction_name] = (
                    direction_oof["metrics"]
                )
                oof_records.append(direction_oof["records"])
        metrics[str(horizon)] = horizon_metrics

        final_model = build_horizon_model(
            settings,
            horizon,
            general_feature_columns,
            directional_feature_columns,
        )
        general_valid = frame[
            [
                f"target_up_h{horizon}",
                f"future_return_h{horizon}",
            ]
        ].notna().all(axis=1)
        final_model.fit_general(
            frame.loc[general_valid, general_feature_columns],
            frame.loc[
                general_valid,
                f"target_up_h{horizon}",
            ].to_numpy(dtype=int),
            frame.loc[
                general_valid,
                f"future_return_h{horizon}",
            ].to_numpy(dtype=float),
            general_weights[general_valid.to_numpy()],
        )
        for direction in (LONG, SHORT):
            direction_events = events.loc[
                events["event_direction"] == direction
            ].copy()
            valid = direction_events[
                [
                    f"event_continuation_h{horizon}",
                    f"tradeable_h{horizon}",
                    f"event_gross_return_h{horizon}",
                ]
            ].notna().all(axis=1)
            direction_events = direction_events.loc[valid].reset_index(
                drop=True
            )
            event_weights = _event_weights(direction_events)
            final_model.fit_direction(
                direction,
                direction_events,
                direction_events[
                    f"event_continuation_h{horizon}"
                ].to_numpy(dtype=float),
                direction_events[
                    f"tradeable_h{horizon}"
                ].to_numpy(dtype=float),
                direction_events[
                    f"event_gross_return_h{horizon}"
                ].to_numpy(dtype=float),
                event_weights,
            )
        models[horizon] = final_model

    qualification = _qualify_directional_model(
        metrics,
        settings,
        trade_horizons,
    )
    created_at = pd.Timestamp.now(tz="UTC")
    model_id = (
        "directional-breakout-hourly-"
        f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    raw_weights = settings.section("model").get(
        "horizon_weights",
        {1: 1.0},
    )
    horizon_weights = {
        horizon: float(
            raw_weights.get(
                horizon,
                raw_weights.get(str(horizon), 0.0),
            )
        )
        for horizon in horizons
    }
    weight_total = sum(horizon_weights.values())
    if weight_total <= 0:
        horizon_weights = {
            horizon: 1.0 if horizon == 1 else 0.0
            for horizon in horizons
        }
    else:
        horizon_weights = {
            horizon: value / weight_total
            for horizon, value in horizon_weights.items()
        }

    bundle = HourlyModelBundle(
        model_id=model_id,
        created_at=created_at.isoformat(),
        provider=provider,
        symbol=symbol,
        feature_columns=general_feature_columns,
        event_feature_columns=directional_feature_columns,
        horizons=horizons,
        trade_horizons=trade_horizons,
        horizon_weights=horizon_weights,
        models=models,
        metrics=metrics,
        qualification=qualification,
        training_range={
            "start": pd.Timestamp(frame["open_time"].min()).isoformat(),
            "end": pd.Timestamp(frame["open_time"].max()).isoformat(),
        },
        event_inventory=inventory_report,
        config_snapshot={
            "event_mining": settings.section("event_mining"),
            "event_inventory": settings.section("event_inventory"),
            "long_breakout": settings.section("long_breakout"),
            "short_breakdown": settings.section("short_breakdown"),
            "long_model": settings.section("long_model"),
            "short_model": settings.section("short_model"),
            "model": settings.section("model"),
            "strategy": settings.section("strategy"),
            "qualification": settings.section("qualification"),
        },
        schema_version=5,
    )
    model_directory = settings.path("model_dir")
    bundle.save(model_directory / f"{model_id}.joblib")
    bundle.save(model_directory / "latest.joblib")

    report = {
        "model_id": model_id,
        "schema_version": 5,
        "created_at": created_at.isoformat(),
        "provider": provider,
        "training_range": bundle.training_range,
        "feature_count": len(general_feature_columns),
        "event_feature_count": len(directional_feature_columns),
        "features": general_feature_columns,
        "event_features": directional_feature_columns,
        "event_inventory": inventory_report,
        "metrics": metrics,
        "qualification": qualification,
        "execution_costs": execution_cost_breakdown(
            settings.section("strategy")
        ),
        "training_contract": {
            "sampling": "NONE",
            "synthetic_data": False,
            "oversampling": False,
            "undersampling": False,
            "shuffle": False,
            "split": "chronological expanding window",
            "minimum_unique_long_events": 2000,
            "minimum_unique_short_events": 2000,
            "long_and_short_models_are_separate": True,
        },
        "objective": {
            "public_forecast": (
                "next closed one-hour candle direction and close range"
            ),
            "long_setup": (
                "real resistance crossing followed by sustained upside"
            ),
            "short_setup": (
                "real support crossing followed by sustained downside"
            ),
            "entry": "open of the next hourly candle",
            "trade_horizons": trade_horizons,
        },
    }
    report_directory = settings.path("report_dir")
    report_directory.mkdir(parents=True, exist_ok=True)
    _write_json(report_directory / f"{model_id}.json", report)
    _write_json(
        report_directory / "latest_training_report.json",
        report,
    )
    pd.concat(oof_records, ignore_index=True).to_csv(
        report_directory / f"{model_id}_oof.csv",
        index=False,
    )
    _summary_csv(metrics).to_csv(
        report_directory / "latest_metrics.csv",
        index=False,
    )
    return report


def _general_walk_forward(
    frame: pd.DataFrame,
    horizon: int,
    settings: Settings,
    feature_columns: list[str],
    event_feature_columns: list[str],
    sample_weight: np.ndarray,
    splits: int,
) -> dict[str, Any]:
    valid = frame[
        [f"target_up_h{horizon}", f"future_return_h{horizon}"]
    ].notna().all(axis=1)
    subset = frame.loc[valid].reset_index(drop=True)
    weights = sample_weight[valid.to_numpy()]
    X = subset[feature_columns]
    y_direction = subset[f"target_up_h{horizon}"].to_numpy(dtype=int)
    y_return = subset[f"future_return_h{horizon}"].to_numpy(dtype=float)
    oof_probability = np.full(len(subset), np.nan)
    oof_return = np.full(len(subset), np.nan)
    splitter = TimeSeriesSplit(
        n_splits=splits,
        gap=max(1, horizon),
    )
    fold_metrics: list[dict[str, Any]] = []
    for fold, (train_index, test_index) in enumerate(
        splitter.split(X),
        start=1,
    ):
        model = build_horizon_model(
            settings,
            horizon,
            feature_columns,
            event_feature_columns,
        )
        model.fit_general(
            X.iloc[train_index],
            y_direction[train_index],
            y_return[train_index],
            weights[train_index],
        )
        prediction = model.predict(X.iloc[test_index])
        oof_probability[test_index] = prediction["p_up"]
        oof_return[test_index] = prediction["general_return"]
        fold_metrics.append(
            {
                "fold": fold,
                **_general_metrics(
                    y_direction[test_index],
                    prediction["p_up"],
                    y_return[test_index],
                    prediction["general_return"],
                ),
                "test_start": pd.Timestamp(
                    subset.iloc[test_index]["open_time"].min()
                ).isoformat(),
                "test_end": pd.Timestamp(
                    subset.iloc[test_index]["open_time"].max()
                ).isoformat(),
            }
        )
    oof_valid = np.isfinite(oof_probability) & np.isfinite(oof_return)
    metrics = _general_metrics(
        y_direction[oof_valid],
        oof_probability[oof_valid],
        y_return[oof_valid],
        oof_return[oof_valid],
    )
    metrics["general_folds"] = fold_metrics
    records = pd.DataFrame(
        {
            "record_type": "GENERAL",
            "open_time": subset.loc[oof_valid, "open_time"].to_numpy(),
            "horizon": horizon,
            "actual_up": y_direction[oof_valid],
            "actual_return": y_return[oof_valid],
            "p_up": oof_probability[oof_valid],
            "predicted_return": oof_return[oof_valid],
        }
    )
    return {"metrics": metrics, "records": records}


def _direction_walk_forward(
    events: pd.DataFrame,
    horizon: int,
    direction: int,
    direction_name: str,
    settings: Settings,
    general_feature_columns: list[str],
    event_feature_columns: list[str],
    splits: int,
    embargo_hours: int,
) -> dict[str, Any]:
    subset = events.loc[events["event_direction"] == direction].copy()
    required = [
        f"event_continuation_h{horizon}",
        f"tradeable_h{horizon}",
        f"event_gross_return_h{horizon}",
        f"event_net_return_h{horizon}",
    ]
    subset = (
        subset.dropna(subset=required)
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    if len(subset) < 1000:
        raise ValueError(
            f"Only {len(subset)} labeled {direction_name} events are "
            f"available for horizon {horizon}h"
        )
    X = subset[general_feature_columns]
    y_success = subset[
        f"event_continuation_h{horizon}"
    ].to_numpy(dtype=int)
    y_tradeable = subset[f"tradeable_h{horizon}"].to_numpy(dtype=int)
    y_return = subset[
        f"event_gross_return_h{horizon}"
    ].to_numpy(dtype=float)
    actual_net = subset[
        f"event_net_return_h{horizon}"
    ].to_numpy(dtype=float)
    weights = _event_weights(subset)
    oof_success = np.full(len(subset), np.nan)
    oof_tradeable = np.full(len(subset), np.nan)
    oof_return = np.full(len(subset), np.nan)
    fold_metrics: list[dict[str, Any]] = []

    for fold, (train_index, test_index) in enumerate(
        _expanding_event_splits(
            subset["open_time"],
            splits=splits,
            embargo_hours=embargo_hours,
        ),
        start=1,
    ):
        model = build_horizon_model(
            settings,
            horizon,
            general_feature_columns,
            event_feature_columns,
        )
        model.fit_direction(
            direction,
            X.iloc[train_index],
            y_success[train_index],
            y_tradeable[train_index],
            y_return[train_index],
            weights[train_index],
        )
        head = model.long_head if direction == LONG else model.short_head
        prediction = head.predict(X.iloc[test_index])
        oof_success[test_index] = prediction["p_success"]
        oof_tradeable[test_index] = prediction["p_tradeable"]
        oof_return[test_index] = prediction["event_return"]
        fold_metrics.append(
            {
                "fold": fold,
                **_direction_metrics(
                    y_success=y_success[test_index],
                    y_tradeable=y_tradeable[test_index],
                    actual_net=actual_net[test_index],
                    p_success=prediction["p_success"],
                    p_tradeable=prediction["p_tradeable"],
                    predicted_return=prediction["event_return"],
                    horizon=horizon,
                    direction_name=direction_name,
                    settings=settings,
                ),
                "test_start": pd.Timestamp(
                    subset.iloc[test_index]["open_time"].min()
                ).isoformat(),
                "test_end": pd.Timestamp(
                    subset.iloc[test_index]["open_time"].max()
                ).isoformat(),
            }
        )

    valid = (
        np.isfinite(oof_success)
        & np.isfinite(oof_tradeable)
        & np.isfinite(oof_return)
    )
    direction_metrics = _direction_metrics(
        y_success=y_success[valid],
        y_tradeable=y_tradeable[valid],
        actual_net=actual_net[valid],
        p_success=oof_success[valid],
        p_tradeable=oof_tradeable[valid],
        predicted_return=oof_return[valid],
        horizon=horizon,
        direction_name=direction_name,
        settings=settings,
    )
    direction_metrics["folds"] = fold_metrics
    direction_metrics["positive_fold_fraction"] = float(
        np.mean([bool(item["positive"]) for item in fold_metrics])
    )
    direction_metrics["event_types"] = (
        subset.loc[valid, "event_type"].value_counts().to_dict()
    )
    direction_metrics["structure_scales"] = (
        subset.loc[valid, "event_scale_hours"].value_counts().to_dict()
    )
    records = pd.DataFrame(
        {
            "record_type": "EVENT",
            "open_time": subset.loc[valid, "open_time"].to_numpy(),
            "event_id": subset.loc[valid, "event_id"].to_numpy(),
            "event_type": subset.loc[valid, "event_type"].to_numpy(),
            "event_direction": direction,
            "direction_name": direction_name,
            "breakout_source": subset.loc[
                valid,
                "breakout_source",
            ].to_numpy(),
            "event_scale_hours": subset.loc[
                valid,
                "event_scale_hours",
            ].to_numpy(),
            "event_score": subset.loc[valid, "event_score"].to_numpy(),
            "horizon": horizon,
            "actual_continuation": y_success[valid],
            "actual_tradeable": y_tradeable[valid],
            "actual_event_net_return": actual_net[valid],
            "p_continuation": oof_success[valid],
            "p_tradeable": oof_tradeable[valid],
            "predicted_event_gross_return": oof_return[valid],
        }
    )
    return {"metrics": direction_metrics, "records": records}


def _general_metrics(
    actual_direction: np.ndarray,
    probability_up: np.ndarray,
    actual_return: np.ndarray,
    predicted_return: np.ndarray,
) -> dict[str, Any]:
    if len(actual_direction) == 0:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "auc": 0.5,
            "brier": 0.25,
            "log_loss": float(np.log(2)),
            "calibration_error": 1.0,
            "return_mae": 0.0,
        }
    predicted_direction = (probability_up >= 0.5).astype(int)
    return {
        "samples": int(len(actual_direction)),
        "accuracy": float(
            accuracy_score(actual_direction, predicted_direction)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                actual_direction,
                predicted_direction,
            )
        ),
        "precision_up": float(
            precision_score(
                actual_direction,
                predicted_direction,
                zero_division=0,
            )
        ),
        "recall_up": float(
            recall_score(
                actual_direction,
                predicted_direction,
                zero_division=0,
            )
        ),
        "auc": _safe_auc(actual_direction, probability_up),
        "brier": float(brier_score_loss(actual_direction, probability_up)),
        "log_loss": float(
            log_loss(
                actual_direction,
                np.column_stack((1 - probability_up, probability_up)),
                labels=[0, 1],
            )
        ),
        "calibration_error": _calibration_error(
            actual_direction,
            probability_up,
        ),
        "return_mae": float(
            mean_absolute_error(actual_return, predicted_return)
        ),
    }


def _direction_metrics(
    y_success: np.ndarray,
    y_tradeable: np.ndarray,
    actual_net: np.ndarray,
    p_success: np.ndarray,
    p_tradeable: np.ndarray,
    predicted_return: np.ndarray,
    horizon: int,
    direction_name: str,
    settings: Settings,
) -> dict[str, Any]:
    if len(y_success) == 0:
        return {
            "event_samples": 0,
            "success_auc": 0.5,
            "tradeability_auc": 0.5,
            "selected": 0,
            "hit_rate": 0.0,
            "mean_net_return": 0.0,
            "positive": False,
        }
    direction_cfg = settings.section(
        "long_breakout" if direction_name == "LONG" else "short_breakdown"
    )
    success_threshold = _per_horizon(
        direction_cfg.get("minimum_success_probability", {}),
        horizon,
        0.58,
    )
    tradeability_threshold = _per_horizon(
        direction_cfg.get("minimum_tradeability_probability", {}),
        horizon,
        0.56,
    )
    costs = execution_cost_breakdown(settings.section("strategy"))
    stress_cost = float(costs["stress_cost_bps"]) / 10_000.0
    minimum_edge = float(
        settings.section("strategy").get("minimum_net_edge_bps", 8.0)
    ) / 10_000.0
    selected_mask = (
        (p_success >= success_threshold)
        & (p_tradeable >= tradeability_threshold)
        & ((predicted_return - stress_cost) >= minimum_edge)
    )
    selected = int(selected_mask.sum())
    hit_rate = (
        float(np.mean(y_success[selected_mask])) if selected else 0.0
    )
    mean_net = (
        float(np.mean(actual_net[selected_mask])) if selected else 0.0
    )
    cumulative = (
        float(np.prod(1.0 + actual_net[selected_mask]) - 1.0)
        if selected
        else 0.0
    )
    return {
        "event_samples": int(len(y_success)),
        "success_rate": float(np.mean(y_success)),
        "tradeability_rate": float(np.mean(y_tradeable)),
        "success_auc": _safe_auc(y_success, p_success),
        "success_balanced_accuracy": float(
            balanced_accuracy_score(
                y_success,
                (p_success >= 0.5).astype(int),
            )
        ),
        "success_brier": float(brier_score_loss(y_success, p_success)),
        "success_calibration_error": _calibration_error(
            y_success,
            p_success,
        ),
        "tradeability_auc": _safe_auc(y_tradeable, p_tradeable),
        "tradeability_brier": float(
            brier_score_loss(y_tradeable, p_tradeable)
        ),
        "tradeability_calibration_error": _calibration_error(
            y_tradeable,
            p_tradeable,
        ),
        "return_mae": float(
            mean_absolute_error(actual_net, predicted_return)
        ),
        "selected": selected,
        "hit_rate": hit_rate,
        "mean_net_return": mean_net,
        "cumulative_net_return": cumulative,
        "positive": bool(selected > 0 and mean_net > 0),
        "thresholds": {
            "success_probability": success_threshold,
            "tradeability_probability": tradeability_threshold,
            "minimum_net_edge": minimum_edge,
        },
    }


def _qualify_directional_model(
    metrics: dict[str, Any],
    settings: Settings,
    trade_horizons: list[int],
) -> dict[str, Any]:
    cfg = settings.section("qualification")
    qualified_directions: dict[str, list[int]] = {
        "LONG": [],
        "SHORT": [],
    }
    per_direction: dict[str, dict[str, Any]] = {
        "LONG": {},
        "SHORT": {},
    }
    blockers: list[str] = []
    for horizon in trade_horizons:
        horizon_metrics = metrics.get(str(horizon), {})
        for direction_name in ("LONG", "SHORT"):
            item = horizon_metrics.get("directions", {}).get(
                direction_name,
                {},
            )
            reasons: list[str] = []
            minimum_samples = int(
                cfg.get("minimum_oof_events_per_direction", 1000)
            )
            minimum_success_auc = float(
                cfg.get(
                    "long_minimum_success_auc"
                    if direction_name == "LONG"
                    else "short_minimum_success_auc",
                    0.54,
                )
            )
            minimum_tradeability_auc = float(
                cfg.get(
                    "long_minimum_tradeability_auc"
                    if direction_name == "LONG"
                    else "short_minimum_tradeability_auc",
                    0.54,
                )
            )
            if int(item.get("event_samples", 0)) < minimum_samples:
                reasons.append("insufficient chronological OOF events")
            if float(item.get("success_auc", 0.0)) < minimum_success_auc:
                reasons.append("success AUC below direction-specific minimum")
            if (
                float(item.get("tradeability_auc", 0.0))
                < minimum_tradeability_auc
            ):
                reasons.append(
                    "tradeability AUC below direction-specific minimum"
                )
            if (
                float(item.get("success_calibration_error", 1.0))
                > float(cfg.get("maximum_success_calibration_error", 0.10))
            ):
                reasons.append("success calibration error above maximum")
            if (
                float(item.get("tradeability_calibration_error", 1.0))
                > float(
                    cfg.get("maximum_tradeability_calibration_error", 0.10)
                )
            ):
                reasons.append(
                    "tradeability calibration error above maximum"
                )
            if int(item.get("selected", 0)) < int(
                cfg.get("minimum_oof_trades", 30)
            ):
                reasons.append("too few selected OOF trades")
            if float(item.get("hit_rate", 0.0)) < float(
                cfg.get("minimum_oof_hit_rate", 0.53)
            ):
                reasons.append("OOF hit rate below minimum")
            if bool(cfg.get("require_positive_oof_expectancy", True)) and (
                float(item.get("mean_net_return", 0.0)) <= 0
            ):
                reasons.append("OOF net expectancy is not positive")
            if float(item.get("positive_fold_fraction", 0.0)) < float(
                cfg.get("minimum_positive_fold_fraction", 0.50)
            ):
                reasons.append("positive fold fraction below minimum")
            passed = not reasons
            per_direction[direction_name][str(horizon)] = {
                "passed": passed,
                "blockers": reasons,
            }
            if passed:
                qualified_directions[direction_name].append(horizon)
            else:
                blockers.extend(
                    [
                        f"{direction_name} h{horizon}: {reason}"
                        for reason in reasons
                    ]
                )
    qualified_horizons = sorted(
        {
            horizon
            for values in qualified_directions.values()
            for horizon in values
        }
    )
    qualified_direction_count = sum(
        bool(values) for values in qualified_directions.values()
    )
    passed = qualified_direction_count >= int(
        cfg.get("minimum_qualified_directions", 1)
    )
    if not passed:
        blockers.insert(
            0,
            "No direction has a qualified deterministic trade horizon",
        )
    return {
        "passed": passed,
        "qualified_horizons": qualified_horizons,
        "qualified_directions": qualified_directions,
        "per_direction": per_direction,
        "blockers": blockers,
        "validation": {
            "sampling": "NONE",
            "shuffle": False,
            "split": "CHRONOLOGICAL_EXPANDING_WINDOW",
        },
    }


def _expanding_event_splits(
    timestamps: Iterable[Any],
    splits: int,
    embargo_hours: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    times = pd.to_datetime(pd.Series(timestamps), utc=True).reset_index(drop=True)
    count = len(times)
    test_size = count // (splits + 1)
    if test_size < 100:
        raise ValueError(
            f"Only {count} events are available for {splits} expanding folds"
        )
    result: list[tuple[np.ndarray, np.ndarray]] = []
    first_test = count - splits * test_size
    for fold in range(splits):
        start = first_test + fold * test_size
        stop = count if fold == splits - 1 else start + test_size
        test_index = np.arange(start, stop, dtype=int)
        cutoff = times.iloc[start] - pd.Timedelta(hours=embargo_hours)
        train_index = np.flatnonzero((times < cutoff).to_numpy())
        if len(train_index) < 300 or len(test_index) < 100:
            raise ValueError(
                "A chronological event fold has insufficient train or test rows"
            )
        result.append((train_index, test_index))
    return result


def _event_weights(events: pd.DataFrame) -> np.ndarray:
    if events.empty:
        return np.asarray([], dtype=float)
    recency = np.linspace(0.55, 1.0, len(events), dtype=float)
    score = pd.to_numeric(
        events.get("event_score", 0.5),
        errors="coerce",
    ).fillna(0.5).to_numpy(dtype=float)
    touches = pd.to_numeric(
        events.get("breakout_level_touches", 1.0),
        errors="coerce",
    ).fillna(1.0).to_numpy(dtype=float)
    quality = (
        0.65
        + 0.55 * np.clip(score, 0.0, 1.0)
        + 0.20 * np.clip(touches / 5.0, 0.0, 1.0)
    )
    return recency * quality


def _summary_csv(metrics: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon_key, item in metrics.items():
        row: dict[str, Any] = {
            "horizon": f"{horizon_key}h",
            "samples": item.get("samples", 0),
            "accuracy": item.get("accuracy", 0.0),
            "auc": item.get("auc", 0.5),
            "return_mae": item.get("return_mae", 0.0),
        }
        for direction_name, prefix in (("LONG", "long"), ("SHORT", "short")):
            direction = item.get("directions", {}).get(direction_name, {})
            row.update(
                {
                    f"{prefix}_events": direction.get("event_samples", 0),
                    f"{prefix}_success_auc": direction.get(
                        "success_auc",
                        0.5,
                    ),
                    f"{prefix}_tradeability_auc": direction.get(
                        "tradeability_auc",
                        0.5,
                    ),
                    f"{prefix}_selected": direction.get("selected", 0),
                    f"{prefix}_hit_rate": direction.get("hit_rate", 0.0),
                    f"{prefix}_mean_net_return": direction.get(
                        "mean_net_return",
                        0.0,
                    ),
                    f"{prefix}_positive_fold_fraction": direction.get(
                        "positive_fold_fraction",
                        0.0,
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_auc(actual: np.ndarray, probability: np.ndarray) -> float:
    return (
        float(roc_auc_score(actual, probability))
        if len(np.unique(actual)) == 2
        else 0.5
    )


def _calibration_error(
    actual: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    if len(actual) == 0:
        return 1.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(actual)
    error = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (
            (probability >= lower)
            & (probability < upper if index < bins - 1 else probability <= upper)
        )
        if not mask.any():
            continue
        error += (
            float(mask.sum())
            / total
            * abs(float(np.mean(actual[mask])) - float(np.mean(probability[mask])))
        )
    return float(error)


def _is_absolute_price_feature(column: str) -> bool:
    if column in {
        "ema_24",
        "ema_72",
        "ema_168",
        "ema_336",
        "breakout_level",
        "breakout_invalidation_level",
    }:
        return True
    return bool(_ABSOLUTE_STRUCTURE_LEVEL.match(column))


def _per_horizon(value: Any, horizon: int, default: float) -> float:
    if isinstance(value, dict):
        return float(value.get(horizon, value.get(str(horizon), default)))
    return float(default if value is None else value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
