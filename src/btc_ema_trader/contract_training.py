from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings
from .features import FeatureSet, build_feature_set
from .forecast_contract import attach_close_based_general_labels
from .storage import Database
from .training import train_feature_set


def train_from_database(
    settings: Settings,
    database: Database,
    provider: str | None = None,
) -> dict[str, object]:
    market_cfg = settings.section("market")
    symbol = str(market_cfg.get("symbol", "BTCUSDT"))
    if provider is None:
        candidates = database.providers(symbol)
        if not candidates:
            raise ValueError(
                "No candle history found. Run: btc-regime fetch --days 180"
            )
        provider = str(candidates[0]["provider"])

    candles = database.load_candles(
        provider=provider,
        symbol=symbol,
    )
    history_days = float(market_cfg.get("history_days", 180))
    cutoff = candles["open_time"].max() - pd.Timedelta(
        days=history_days
    )
    candles = candles[
        candles["open_time"] >= cutoff
    ].reset_index(drop=True)
    news = database.load_news(
        start=candles["open_time"].min(),
        end=candles["open_time"].max()
        + pd.Timedelta(hours=1),
    )
    feature_set = build_feature_set(
        candles,
        news,
        settings,
        include_labels=True,
    )
    prepared = FeatureSet(
        frame=attach_close_based_general_labels(
            feature_set.frame,
            feature_set.horizons,
        ),
        feature_columns=feature_set.feature_columns,
        horizons=feature_set.horizons,
    )
    report = train_feature_set(
        settings,
        prepared,
        provider=provider,
        symbol=symbol,
    )
    return enrich_interval_metrics(settings, report)


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
