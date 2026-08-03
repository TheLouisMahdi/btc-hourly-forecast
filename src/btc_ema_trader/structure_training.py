from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import Settings
from .costs import execution_cost_breakdown
from .features import FeatureSet, sample_weights
from .model import HourlyModelBundle, build_horizon_model
from .training import _metrics, _summary_csv, qualify_model


def train_feature_set(
    settings: Settings,
    feature_set: FeatureSet,
    provider: str,
    symbol: str,
) -> dict[str, Any]:
    frame = feature_set.frame.copy()
    horizons = feature_set.horizons
    frame = frame.dropna(
        subset=[
            column
            for horizon in horizons
            for column in (
                f"target_up_h{horizon}",
                f"future_return_h{horizon}",
            )
        ]
    ).reset_index(drop=True)
    feature_columns = [
        column
        for column in feature_set.feature_columns
        if frame[column].notna().sum()
        > max(80, len(frame) * 0.12)
    ]
    minimum_rows = int(
        settings.section("model").get("min_train_rows", 5000)
    )
    if len(frame) < minimum_rows:
        raise ValueError(
            f"Only {len(frame)} trainable rows; "
            f"at least {minimum_rows} are required"
        )
    event_count = int(frame["is_event"].sum())
    minimum_events = int(
        settings.section("model").get(
            "minimum_training_events",
            120,
        )
    )
    if event_count < minimum_events:
        raise ValueError(
            f"Only {event_count} confirmed structural breakouts were "
            f"found; at least {minimum_events} are required"
        )

    X = frame[feature_columns]
    weights = sample_weights(frame, settings)
    split_count = int(
        settings.section("model").get("walk_forward_splits", 6)
    )
    gap = max(
        max(horizons),
        int(
            settings.section("model").get(
                "validation_gap_hours",
                6,
            )
        ),
    )
    splitter = TimeSeriesSplit(
        n_splits=split_count,
        gap=gap,
    )

    models: dict[int, Any] = {}
    horizon_metrics: dict[str, Any] = {}
    oof_records: list[pd.DataFrame] = []

    for horizon in horizons:
        y_direction = frame[
            f"target_up_h{horizon}"
        ].astype(int).to_numpy()
        y_return = frame[
            f"future_return_h{horizon}"
        ].to_numpy(dtype=float)
        y_continuation = frame[
            f"event_continuation_h{horizon}"
        ].to_numpy(dtype=float)
        y_tradeable = frame[
            f"tradeable_h{horizon}"
        ].to_numpy(dtype=float)
        y_event_return = frame[
            f"event_gross_return_h{horizon}"
        ].to_numpy(dtype=float)
        y_hold = frame[
            f"breakout_hold_h{horizon}"
        ].to_numpy(dtype=float)
        y_false_breakout = frame[
            f"false_breakout_h{horizon}"
        ].to_numpy(dtype=float)
        event_mask = frame["is_event"].to_numpy(dtype=bool)
        event_direction = frame[
            "event_direction"
        ].to_numpy(dtype=int)

        oof_p_up = np.full(len(frame), np.nan)
        oof_general_return = np.full(len(frame), np.nan)
        oof_continuation = np.full(len(frame), np.nan)
        oof_tradeability = np.full(len(frame), np.nan)
        oof_event_return = np.full(len(frame), np.nan)
        fold_metrics: list[dict[str, Any]] = []

        for fold, (train_index, test_index) in enumerate(
            splitter.split(X),
            start=1,
        ):
            model = build_horizon_model(settings, horizon)
            model.fit(
                X.iloc[train_index],
                y_direction[train_index],
                y_return[train_index],
                weights[train_index],
                event_mask[train_index],
                y_continuation[train_index],
                y_tradeable[train_index],
                y_event_return[train_index],
            )
            prediction = model.predict(X.iloc[test_index])
            oof_p_up[test_index] = prediction["p_up"]
            oof_general_return[test_index] = prediction[
                "general_return"
            ]
            oof_continuation[test_index] = prediction[
                "p_continuation"
            ]
            oof_tradeability[test_index] = prediction[
                "p_tradeable"
            ]
            oof_event_return[test_index] = prediction[
                "event_return"
            ]
            fold_result = _metrics(
                y_direction[test_index],
                prediction["p_up"],
                y_return[test_index],
                prediction["general_return"],
                event_mask[test_index],
                event_direction[test_index],
                y_continuation[test_index],
                prediction["p_continuation"],
                y_tradeable[test_index],
                prediction["p_tradeable"],
                y_event_return[test_index],
                prediction["event_return"],
                frame.iloc[test_index],
                horizon,
                settings,
            )
            fold_result.update(
                {
                    "fold": fold,
                    "test_start": frame.iloc[test_index][
                        "open_time"
                    ].min().isoformat(),
                    "test_end": frame.iloc[test_index][
                        "open_time"
                    ].max().isoformat(),
                    "breakout_hold_rate": _event_rate(
                        y_hold[test_index],
                        event_mask[test_index],
                    ),
                    "false_breakout_rate": _event_rate(
                        y_false_breakout[test_index],
                        event_mask[test_index],
                    ),
                }
            )
            fold_metrics.append(fold_result)

        valid = np.isfinite(oof_p_up)
        metrics = _metrics(
            y_direction[valid],
            oof_p_up[valid],
            y_return[valid],
            oof_general_return[valid],
            event_mask[valid],
            event_direction[valid],
            y_continuation[valid],
            oof_continuation[valid],
            y_tradeable[valid],
            oof_tradeability[valid],
            y_event_return[valid],
            oof_event_return[valid],
            frame.loc[valid],
            horizon,
            settings,
        )
        metrics["folds"] = fold_metrics
        metrics["positive_fold_fraction"] = float(
            np.mean(
                [
                    bool(
                        fold.get("trading", {}).get(
                            "positive",
                            False,
                        )
                    )
                    for fold in fold_metrics
                ]
            )
        )
        metrics["breakout_hold_rate"] = _event_rate(
            y_hold[valid],
            event_mask[valid],
        )
        metrics["false_breakout_rate"] = _event_rate(
            y_false_breakout[valid],
            event_mask[valid],
        )
        horizon_metrics[str(horizon)] = metrics

        oof_records.append(
            pd.DataFrame(
                {
                    "open_time": frame.loc[
                        valid,
                        "open_time",
                    ].to_numpy(),
                    "event_id": frame.loc[
                        valid,
                        "event_id",
                    ].to_numpy(),
                    "event_type": frame.loc[
                        valid,
                        "event_type",
                    ].to_numpy(),
                    "event_direction": event_direction[valid],
                    "breakout_source": frame.loc[
                        valid,
                        "breakout_source",
                    ].to_numpy(),
                    "triangle_type": frame.loc[
                        valid,
                        "triangle_type",
                    ].to_numpy(),
                    "breakout_level": frame.loc[
                        valid,
                        "breakout_level",
                    ].to_numpy(),
                    "event_score": frame.loc[
                        valid,
                        "event_score",
                    ].to_numpy(),
                    "horizon": horizon,
                    "actual_up": y_direction[valid],
                    "actual_return": y_return[valid],
                    "p_up": oof_p_up[valid],
                    "predicted_return": oof_general_return[valid],
                    "actual_continuation": y_continuation[valid],
                    "actual_hold": y_hold[valid],
                    "actual_false_breakout": y_false_breakout[valid],
                    "p_continuation": oof_continuation[valid],
                    "actual_tradeable": y_tradeable[valid],
                    "p_tradeable": oof_tradeability[valid],
                    "actual_event_gross_return": y_event_return[valid],
                    "predicted_event_gross_return": oof_event_return[valid],
                    "is_event": event_mask[valid],
                }
            )
        )

        final_model = build_horizon_model(settings, horizon)
        final_model.fit(
            X,
            y_direction,
            y_return,
            weights,
            event_mask,
            y_continuation,
            y_tradeable,
            y_event_return,
        )
        models[horizon] = final_model

    qualification = qualify_model(horizon_metrics, settings)
    created_at = pd.Timestamp.now(tz="UTC")
    model_id = (
        "structure-breakout-hourly-"
        f"{created_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    raw_weights = settings.section("model").get(
        "horizon_weights",
        {},
    )
    horizon_weights = {
        int(horizon): float(
            raw_weights.get(
                horizon,
                raw_weights.get(str(horizon), 1 / len(horizons)),
            )
        )
        for horizon in horizons
    }
    weight_total = sum(horizon_weights.values())
    horizon_weights = {
        horizon: value / weight_total
        for horizon, value in horizon_weights.items()
    }

    bundle = HourlyModelBundle(
        model_id=model_id,
        created_at=created_at.isoformat(),
        provider=provider,
        symbol=symbol,
        feature_columns=feature_columns,
        horizons=horizons,
        horizon_weights=horizon_weights,
        models=models,
        metrics=horizon_metrics,
        qualification=qualification,
        training_range={
            "start": frame["open_time"].min().isoformat(),
            "end": frame["open_time"].max().isoformat(),
        },
        config_snapshot={
            "features": settings.section("features"),
            "structure": settings.section("structure"),
            "model": settings.section("model"),
            "strategy": settings.section("strategy"),
            "qualification": settings.section("qualification"),
        },
        schema_version=4,
    )
    model_directory = settings.path("model_dir")
    bundle.save(model_directory / f"{model_id}.joblib")
    bundle.save(model_directory / "latest.joblib")

    event_rows = frame.loc[frame["is_event"] == 1]
    event_counts = event_rows["event_type"].value_counts().to_dict()
    triangle_counts = event_rows["triangle_type"].value_counts().to_dict()
    report = {
        "model_id": model_id,
        "schema_version": 4,
        "created_at": created_at.isoformat(),
        "provider": provider,
        "training_range": bundle.training_range,
        "event_counts": event_counts,
        "triangle_event_counts": triangle_counts,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "metrics": horizon_metrics,
        "qualification": qualification,
        "execution_costs": execution_cost_breakdown(
            settings.section("strategy")
        ),
        "objective": {
            "public_forecast": (
                "next closed one-hour candle direction and close range"
            ),
            "trade_setup": (
                "confirmed long resistance breakout or short support breakdown"
            ),
            "structure": (
                "causal multi-scale static and dynamic levels with triangle detection"
            ),
            "trade_label": (
                "level hold, path-aware target or invalidation, and net tradeability"
            ),
            "entry": "open of the next hourly candle",
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
    _summary_csv(horizon_metrics).to_csv(
        report_directory / "latest_metrics.csv",
        index=False,
    )
    return report


def _event_rate(values: np.ndarray, event_mask: np.ndarray) -> float:
    valid = event_mask & np.isfinite(values)
    if not valid.any():
        return 0.0
    return float(np.mean(values[valid]))


def _write_json(path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
