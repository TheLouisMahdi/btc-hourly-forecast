from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .contract_features import build_feature_set
from .features import FeatureSet
from .model import load_bundle
from .storage import Database
from .structure_training import train_feature_set


def train_from_database(
    settings: Settings,
    database: Database,
    provider: str | None = None,
) -> dict[str, object]:
    market_cfg = settings.section("market")
    symbol = str(market_cfg.get("symbol", "BTCUSDT"))
    history_days = float(market_cfg.get("history_days", 3650))
    if provider is None:
        candidates = database.providers(symbol)
        if not candidates:
            raise ValueError(
                "No candle history found. Run: btc-regime fetch "
                f"--days {int(history_days)}"
            )
        provider = str(candidates[0]["provider"])

    candles = database.load_candles(
        provider=provider,
        symbol=symbol,
    )
    cutoff = candles["open_time"].max() - pd.Timedelta(
        days=history_days
    )
    candles = candles[
        candles["open_time"] >= cutoff
    ].reset_index(drop=True)
    news_days = float(
        settings.section("news").get("historical_days", 365)
    )
    news_start = max(
        candles["open_time"].min(),
        candles["open_time"].max() - pd.Timedelta(days=news_days),
    )
    news = database.load_news(
        start=news_start,
        end=candles["open_time"].max() + pd.Timedelta(hours=1),
    )
    feature_set, segmentation = build_segmented_feature_set(
        candles,
        news,
        settings,
        include_labels=True,
    )
    report = train_feature_set(
        settings,
        feature_set,
        provider=provider,
        symbol=symbol,
    )
    report = stamp_model_sample_policy(settings, report)
    report["market_data_segmentation"] = segmentation
    return enrich_interval_metrics(settings, report)


def stamp_model_sample_policy(
    settings: Settings,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Bind the active sample policy to a freshly trained model artifact."""
    model_id = str(report.get("model_id") or "")
    if not model_id:
        raise ValueError("Training report is missing model_id")
    model_dir = settings.path("model_dir")
    latest_path = model_dir / "latest.joblib"
    if not latest_path.exists():
        raise FileNotFoundError(
            "Training finished without latest.joblib for sample-policy stamping"
        )

    bundle = load_bundle(latest_path)
    policy = json.loads(
        json.dumps(settings.section("sample_policy"), ensure_ascii=False)
    )
    bundle.config_snapshot["sample_policy"] = policy
    bundle.save(latest_path)
    model_path = model_dir / f"{model_id}.joblib"
    if model_path.exists():
        bundle.save(model_path)
    report["sample_policy"] = policy
    return report


def build_segmented_feature_set(
    candles: pd.DataFrame,
    news: pd.DataFrame,
    settings: Settings,
    include_labels: bool = True,
) -> tuple[FeatureSet, dict[str, Any]]:
    segments, continuity = _contiguous_segments(candles)
    market_cfg = settings.section("market")
    maximum_gap_hours = float(
        market_cfg.get("training_maximum_gap_hours", 24)
    )
    advisory_maximum_gap_count = int(
        market_cfg.get("training_maximum_gap_count", 12)
    )
    maximum_missing_hours = int(
        market_cfg.get("training_maximum_missing_hours", 72)
    )
    minimum_segment_rows = int(
        market_cfg.get("training_minimum_segment_rows", 1000)
    )

    if continuity["largest_gap_hours"] > maximum_gap_hours:
        raise ValueError(
            "Historical continuity gate failed; largest real gap is "
            f"{continuity['largest_gap_hours']:.1f} hours but the "
            f"training limit is {maximum_gap_hours:.1f} hours"
        )
    if continuity["missing_candle_hours"] > maximum_missing_hours:
        raise ValueError(
            "Historical continuity gate failed; "
            f"{continuity['missing_candle_hours']} missing candle hours "
            f"exceed the configured limit of {maximum_missing_hours}"
        )

    gap_count_advisory_exceeded = bool(
        continuity["gap_count"] > advisory_maximum_gap_count
    )

    built: list[FeatureSet] = []
    segment_audit: list[dict[str, Any]] = []
    skipped_segments: list[dict[str, Any]] = []
    for segment in segments:
        frame = segment["frame"]
        audit = {
            "segment_id": int(segment["segment_id"]),
            "start": pd.Timestamp(segment["start"]).isoformat(),
            "end": pd.Timestamp(segment["end"]).isoformat(),
            "input_rows": int(len(frame)),
            "gap_before_hours": segment["gap_before_hours"],
        }
        if len(frame) < minimum_segment_rows:
            audit["reason"] = "BELOW_MINIMUM_SEGMENT_ROWS"
            skipped_segments.append(audit)
            continue

        feature_set = build_feature_set(
            frame,
            news,
            settings,
            include_labels=include_labels,
        )
        feature_frame = feature_set.frame.copy()
        feature_frame["market_segment_id"] = int(
            segment["segment_id"]
        )
        audit["feature_rows"] = int(len(feature_frame))
        segment_audit.append(audit)
        built.append(
            FeatureSet(
                frame=feature_frame,
                feature_columns=list(feature_set.feature_columns),
                horizons=list(feature_set.horizons),
            )
        )

    if not built:
        raise ValueError(
            "No continuous historical segment is long enough for "
            "feature and breakout training"
        )

    horizons = list(built[0].horizons)
    for item in built[1:]:
        if list(item.horizons) != horizons:
            raise ValueError("Segment horizon configuration mismatch")

    feature_columns = [
        column
        for column in built[0].feature_columns
        if all(column in item.frame.columns for item in built)
    ]
    if not feature_columns:
        raise ValueError("No common feature columns across market segments")

    combined = (
        pd.concat(
            [item.frame for item in built],
            ignore_index=True,
            sort=False,
        )
        .sort_values("open_time")
        .reset_index(drop=True)
    )
    if combined["open_time"].duplicated().any():
        raise ValueError("Duplicate timestamps after segment assembly")

    retained_input_rows = int(
        sum(item["input_rows"] for item in segment_audit)
    )
    retained_input_fraction = float(
        retained_input_rows / max(len(candles), 1)
    )
    audit_report = {
        "policy": "CONTIGUOUS_SEGMENTS_WITHOUT_FILL",
        "random_sampling": False,
        "synthetic_candles": False,
        "interpolation": False,
        "forward_fill": False,
        "input_rows": int(len(candles)),
        "retained_input_rows": retained_input_rows,
        "retained_input_fraction": retained_input_fraction,
        "training_rows": int(len(combined)),
        "used_segment_count": int(len(built)),
        "skipped_segment_count": int(len(skipped_segments)),
        "gap_count": int(continuity["gap_count"]),
        "gap_count_policy": "AUDIT_ONLY",
        "gap_count_advisory_exceeded": gap_count_advisory_exceeded,
        "largest_gap_hours": float(continuity["largest_gap_hours"]),
        "missing_candle_hours": int(
            continuity["missing_candle_hours"]
        ),
        "limits": {
            "maximum_gap_hours": maximum_gap_hours,
            "advisory_maximum_gap_count": advisory_maximum_gap_count,
            "maximum_missing_hours": maximum_missing_hours,
            "minimum_segment_rows": minimum_segment_rows,
        },
        "segments": segment_audit,
        "skipped_segments": skipped_segments,
    }
    return (
        FeatureSet(
            frame=combined,
            feature_columns=feature_columns,
            horizons=horizons,
        ),
        audit_report,
    )


def _contiguous_segments(
    candles: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if candles.empty:
        raise ValueError("No candles available for segmentation")
    prepared = (
        candles.copy()
        .sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
        .reset_index(drop=True)
    )
    prepared["open_time"] = pd.to_datetime(
        prepared["open_time"],
        utc=True,
    )
    differences = (
        prepared["open_time"]
        .diff()
        .dt.total_seconds()
        .div(3600.0)
    )
    continuous = differences.between(
        1.0 - 1e-9,
        1.0 + 1e-9,
        inclusive="both",
    )
    boundaries = differences.isna() | ~continuous
    segment_ids = boundaries.cumsum().astype(int) - 1
    prepared["_market_segment_id"] = segment_ids

    positive_gaps = differences[differences > 1.0 + 1e-9]
    rounded_gaps = [float(value) for value in positive_gaps.tolist()]
    missing_candle_hours = int(
        sum(max(int(round(value)) - 1, 0) for value in rounded_gaps)
    )
    continuity = {
        "gap_count": int(len(rounded_gaps)),
        "largest_gap_hours": (
            float(max(rounded_gaps)) if rounded_gaps else 0.0
        ),
        "missing_candle_hours": missing_candle_hours,
    }

    segments: list[dict[str, Any]] = []
    previous_end: pd.Timestamp | None = None
    for segment_id, segment_frame in prepared.groupby(
        "_market_segment_id",
        sort=True,
    ):
        segment_frame = segment_frame.drop(
            columns=["_market_segment_id"]
        ).reset_index(drop=True)
        start = pd.Timestamp(segment_frame["open_time"].iloc[0])
        end = pd.Timestamp(segment_frame["open_time"].iloc[-1])
        gap_before = None
        if previous_end is not None:
            gap_before = float(
                (start - previous_end).total_seconds() / 3600.0
            )
        segments.append(
            {
                "segment_id": int(segment_id),
                "start": start,
                "end": end,
                "gap_before_hours": gap_before,
                "frame": segment_frame,
            }
        )
        previous_end = end
    return segments, continuity


def enrich_interval_metrics(
    settings: Settings,
    report: dict[str, Any],
) -> dict[str, Any]:
    model_id = str(report.get("model_id") or "")
    if not model_id:
        return report
    report_dir = settings.path("report_dir")
    oof_path = report_dir / f"{model_id}_oof.csv"
    if not oof_path.exists():
        return report

    oof = pd.read_csv(oof_path)
    if "record_type" in oof:
        oof = oof.loc[oof["record_type"] == "GENERAL"].copy()
    interval_probability = float(
        settings.section("forecast").get(
            "interval_probability",
            0.80,
        )
    )
    alpha = (1.0 - interval_probability) / 2.0
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return report

    for horizon_key, item in metrics.items():
        if not isinstance(item, dict):
            continue
        horizon = int(horizon_key)
        rows = oof[oof["horizon"] == horizon].copy()
        actual = pd.to_numeric(
            rows.get("actual_return"),
            errors="coerce",
        )
        predicted = pd.to_numeric(
            rows.get("predicted_return"),
            errors="coerce",
        )
        residual = (actual - predicted).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()
        if residual.empty:
            continue
        lower, upper = residual.quantile(
            [alpha, 1.0 - alpha]
        ).tolist()
        coverage = float(
            ((residual >= lower) & (residual <= upper)).mean()
        )
        item.update(
            {
                "close_interval_probability": interval_probability,
                "close_interval_oof_samples": int(len(residual)),
                "close_interval_residual_low": float(lower),
                "close_interval_residual_high": float(upper),
                "close_interval_oof_coverage": coverage,
            }
        )

    objective = report.setdefault("objective", {})
    if isinstance(objective, dict):
        objective.update(
            {
                "general_forecast": (
                    "probabilistic close range for the next closed "
                    "hourly candle"
                ),
                "general_label": "close-to-close return",
                "interval_calibration": (
                    "chronological walk-forward residual quantiles"
                ),
            }
        )
    _write_report(report_dir / f"{model_id}.json", report)
    _write_report(
        report_dir / "latest_training_report.json",
        report,
    )
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
