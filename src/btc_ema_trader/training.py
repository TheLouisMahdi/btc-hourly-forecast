from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from .config import Settings
from .costs import execution_cost_breakdown
from .features import FeatureSet, build_feature_set, sample_weights
from .model import HourlyModelBundle, build_horizon_model
from .storage import Database

LOGGER = logging.getLogger(__name__)


def train_from_database(settings: Settings, database: Database, provider: str | None = None) -> dict[str, Any]:
    market_cfg = settings.section("market")
    symbol = str(market_cfg.get("symbol", "BTCUSDT"))
    if provider is None:
        candidates = database.providers(symbol)
        if not candidates:
            raise ValueError("No candle history found. Run: btc-ema fetch --days 180")
        provider = str(candidates[0]["provider"])
    candles = database.load_candles(provider=provider, symbol=symbol)
    history_days = float(settings.section("market").get("history_days", 180))
    cutoff = candles["open_time"].max() - pd.Timedelta(days=history_days)
    candles = candles[candles["open_time"] >= cutoff].reset_index(drop=True)
    news = database.load_news(
        start=candles["open_time"].min(),
        end=candles["open_time"].max() + pd.Timedelta(hours=1),
    )
    feature_set = build_feature_set(candles, news, settings, include_labels=True)
    return train_feature_set(settings, feature_set, provider=provider, symbol=symbol)


def train_feature_set(settings: Settings, feature_set: FeatureSet, provider: str, symbol: str) -> dict[str, Any]:
    frame = feature_set.frame.copy()
    horizons = feature_set.horizons
    required_labels: list[str] = []
    for h in horizons:
        required_labels.extend(
            [
                f"target_up_h{h}",
                f"future_return_h{h}",
                f"event_continuation_h{h}",
                f"tradeable_h{h}",
                f"event_gross_return_h{h}",
            ]
        )
    # General labels exist for every row; event labels are NaN on non-events by design.
    frame = frame.dropna(
        subset=[f"target_up_h{h}" for h in horizons] + [f"future_return_h{h}" for h in horizons]
    ).reset_index(drop=True)
    feature_columns = [
        c for c in feature_set.feature_columns
        if frame[c].notna().sum() > max(50, len(frame) * 0.15)
    ]
    min_rows = int(settings.section("model").get("min_train_rows", 1800))
    if len(frame) < min_rows:
        raise ValueError(f"Only {len(frame)} trainable rows; at least {min_rows} are required")
    event_count = int(frame["is_event"].sum())
    min_events = int(settings.section("model").get("minimum_training_events", 80))
    if event_count < min_events:
        raise ValueError(
            f"Only {event_count} independent market events were found; at least {min_events} are required. "
            "Fetch a longer history or loosen event thresholds carefully."
        )

    X = frame[feature_columns]
    weights = sample_weights(frame, settings)
    splits = int(settings.section("model").get("walk_forward_splits", 5))
    gap = max(max(horizons), int(settings.section("model").get("validation_gap_hours", 3)))
    splitter = TimeSeriesSplit(n_splits=splits, gap=gap)

    models: dict[int, Any] = {}
    horizon_metrics: dict[str, Any] = {}
    oof_records: list[pd.DataFrame] = []

    for horizon in horizons:
        y_direction = frame[f"target_up_h{horizon}"].astype(int).to_numpy()
        y_return = frame[f"future_return_h{horizon}"].to_numpy(dtype=float)
        y_continuation = frame[f"event_continuation_h{horizon}"].to_numpy(dtype=float)
        y_tradeable = frame[f"tradeable_h{horizon}"].to_numpy(dtype=float)
        y_event_return = frame[f"event_gross_return_h{horizon}"].to_numpy(dtype=float)
        event_mask = frame["is_event"].to_numpy(dtype=bool)
        event_direction = frame["event_direction"].to_numpy(dtype=int)

        oof_p_up = np.full(len(frame), np.nan)
        oof_general_return = np.full(len(frame), np.nan)
        oof_p_continuation = np.full(len(frame), np.nan)
        oof_p_tradeable = np.full(len(frame), np.nan)
        oof_event_return = np.full(len(frame), np.nan)
        fold_metrics: list[dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            model = build_horizon_model(settings, horizon)
            model.fit(
                X.iloc[train_idx],
                y_direction[train_idx],
                y_return[train_idx],
                weights[train_idx],
                event_mask[train_idx],
                y_continuation[train_idx],
                y_tradeable[train_idx],
                y_event_return[train_idx],
            )
            output = model.predict(X.iloc[test_idx])
            oof_p_up[test_idx] = output["p_up"]
            oof_general_return[test_idx] = output["general_return"]
            oof_p_continuation[test_idx] = output["p_continuation"]
            oof_p_tradeable[test_idx] = output["p_tradeable"]
            oof_event_return[test_idx] = output["event_return"]
            fold_result = _metrics(
                y_direction[test_idx],
                output["p_up"],
                y_return[test_idx],
                output["general_return"],
                event_mask[test_idx],
                event_direction[test_idx],
                y_continuation[test_idx],
                output["p_continuation"],
                y_tradeable[test_idx],
                output["p_tradeable"],
                y_event_return[test_idx],
                output["event_return"],
                frame.iloc[test_idx],
                horizon,
                settings,
            )
            fold_result.update(
                {
                    "fold": fold,
                    "test_start": frame.iloc[test_idx]["open_time"].min().isoformat(),
                    "test_end": frame.iloc[test_idx]["open_time"].max().isoformat(),
                }
            )
            fold_metrics.append(fold_result)

        valid = ~np.isnan(oof_p_up)
        metrics = _metrics(
            y_direction[valid],
            oof_p_up[valid],
            y_return[valid],
            oof_general_return[valid],
            event_mask[valid],
            event_direction[valid],
            y_continuation[valid],
            oof_p_continuation[valid],
            y_tradeable[valid],
            oof_p_tradeable[valid],
            y_event_return[valid],
            oof_event_return[valid],
            frame.loc[valid],
            horizon,
            settings,
        )
        metrics["folds"] = fold_metrics
        metrics["positive_fold_fraction"] = float(
            np.mean([bool(f.get("trading", {}).get("positive", False)) for f in fold_metrics])
        )
        horizon_metrics[str(horizon)] = metrics
        oof_records.append(
            pd.DataFrame(
                {
                    "open_time": frame.loc[valid, "open_time"].to_numpy(),
                    "event_id": frame.loc[valid, "event_id"].to_numpy(),
                    "event_type": frame.loc[valid, "event_type"].to_numpy(),
                    "event_direction": event_direction[valid],
                    "horizon": horizon,
                    "actual_up": y_direction[valid],
                    "actual_return": y_return[valid],
                    "p_up": oof_p_up[valid],
                    "predicted_return": oof_general_return[valid],
                    "actual_continuation": y_continuation[valid],
                    "p_continuation": oof_p_continuation[valid],
                    "actual_tradeable": y_tradeable[valid],
                    "p_tradeable": oof_p_tradeable[valid],
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
    created = pd.Timestamp.now(tz="UTC")
    model_id = f"regime-meta-hourly-{created.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    horizon_weights_raw = settings.section("model").get("horizon_weights", {})
    horizon_weights = {
        int(h): float(horizon_weights_raw.get(h, horizon_weights_raw.get(str(h), 1 / len(horizons))))
        for h in horizons
    }
    total = sum(horizon_weights.values())
    horizon_weights = {h: v / total for h, v in horizon_weights.items()}

    bundle = HourlyModelBundle(
        model_id=model_id,
        created_at=created.isoformat(),
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
            "model": settings.section("model"),
            "strategy": settings.section("strategy"),
            "qualification": settings.section("qualification"),
        },
        schema_version=3,
    )
    model_dir = settings.path("model_dir")
    bundle.save(model_dir / f"{model_id}.joblib")
    bundle.save(model_dir / "latest.joblib")

    event_counts = frame.loc[frame["is_event"] == 1, "event_type"].value_counts().to_dict()
    report = {
        "model_id": model_id,
        "schema_version": 3,
        "created_at": created.isoformat(),
        "provider": provider,
        "training_range": bundle.training_range,
        "event_counts": event_counts,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "metrics": horizon_metrics,
        "qualification": qualification,
        "execution_costs": execution_cost_breakdown(settings.section("strategy")),
        "objective": {
            "general_forecast": "UP/DOWN on all closed hourly candles",
            "trade_gate": "event continuation and event-direction net tradeability",
            "entry": "open of next hourly candle",
        },
    }
    report_dir = settings.path("report_dir")
    report_path = report_dir / f"{model_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "latest_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(oof_records, ignore_index=True).to_csv(report_dir / f"{model_id}_oof.csv", index=False)
    _summary_csv(horizon_metrics).to_csv(report_dir / "latest_metrics.csv", index=False)
    return report


def _metrics(
    y_direction: np.ndarray,
    p_up: np.ndarray,
    actual_return: np.ndarray,
    predicted_return: np.ndarray,
    event_mask: np.ndarray,
    event_direction: np.ndarray,
    y_continuation: np.ndarray,
    p_continuation: np.ndarray,
    y_tradeable: np.ndarray,
    p_tradeable: np.ndarray,
    actual_event_return: np.ndarray,
    predicted_event_return: np.ndarray,
    rows: pd.DataFrame,
    horizon: int,
    settings: Settings,
) -> dict[str, Any]:
    predicted = (p_up >= 0.5).astype(int)
    result: dict[str, Any] = {
        "samples": int(len(y_direction)),
        "accuracy": float(accuracy_score(y_direction, predicted)),
        "balanced_accuracy": _safe_balanced_accuracy(y_direction, predicted),
        "precision_up": float(precision_score(y_direction, predicted, zero_division=0)),
        "recall_up": float(recall_score(y_direction, predicted, zero_division=0)),
        "precision_down": float(precision_score(1 - y_direction, 1 - predicted, zero_division=0)),
        "recall_down": float(recall_score(1 - y_direction, 1 - predicted, zero_division=0)),
        "brier": float(brier_score_loss(y_direction, p_up)),
        "log_loss": float(log_loss(y_direction, np.column_stack([1 - p_up, p_up]), labels=[0, 1])),
        "auc": _safe_auc(y_direction, p_up),
        "calibration_error": _expected_calibration_error(y_direction, p_up),
        "confusion_matrix": confusion_matrix(y_direction, predicted, labels=[0, 1]).tolist(),
        "return_mae": float(np.mean(np.abs(actual_return - predicted_return))),
    }

    valid_event = (
        event_mask
        & np.isfinite(y_continuation)
        & np.isfinite(p_continuation)
        & np.isfinite(y_tradeable)
        & np.isfinite(p_tradeable)
        & np.isfinite(actual_event_return)
        & np.isfinite(predicted_event_return)
    )
    event_count = int(valid_event.sum())
    result["event_samples"] = event_count
    if event_count >= 2:
        yc = y_continuation[valid_event].astype(int)
        pc = p_continuation[valid_event]
        yt = y_tradeable[valid_event].astype(int)
        pt = p_tradeable[valid_event]
        result.update(
            {
                "event_balanced_accuracy": _safe_balanced_accuracy(yc, (pc >= 0.5).astype(int)),
                # event_auc is intentionally continuation AUC for backward-compatible reports.
                "event_auc": _safe_auc(yc, pc),
                "event_calibration_error": _expected_calibration_error(yc, pc),
                "event_brier": float(brier_score_loss(yc, pc)),
                "tradeability_auc": _safe_auc(yt, pt),
                "tradeability_calibration_error": _expected_calibration_error(yt, pt),
                "tradeability_brier": float(brier_score_loss(yt, pt)),
                "tradeability_positive_rate": float(np.mean(yt)),
                "event_return_mae": float(
                    np.mean(np.abs(actual_event_return[valid_event] - predicted_event_return[valid_event]))
                ),
            }
        )
    else:
        result.update(
            {
                "event_balanced_accuracy": 0.0,
                "event_auc": 0.5,
                "event_calibration_error": 1.0,
                "event_brier": 0.25,
                "tradeability_auc": 0.5,
                "tradeability_calibration_error": 1.0,
                "tradeability_brier": 0.25,
                "tradeability_positive_rate": 0.0,
                "event_return_mae": 0.0,
            }
        )

    event_types = rows.get("event_type", pd.Series(index=rows.index, dtype=object)).to_numpy()
    result["event_type_metrics"] = _event_type_metrics(
        y_continuation, p_continuation, y_tradeable, p_tradeable, valid_event, event_types
    )
    result["trading"] = _oof_trading_metrics(
        p_continuation,
        p_tradeable,
        actual_event_return,
        predicted_event_return,
        valid_event,
        horizon,
        settings,
    )
    return result


def _oof_trading_metrics(
    p_continuation: np.ndarray,
    p_tradeable: np.ndarray,
    actual_event_return: np.ndarray,
    predicted_event_return: np.ndarray,
    event_mask: np.ndarray,
    horizon: int,
    settings: Settings,
) -> dict[str, Any]:
    cfg = settings.section("strategy")
    costs = execution_cost_breakdown(cfg)
    base_cost = costs["base_cost_bps"] / 10_000
    stress_cost = costs["stress_cost_bps"] / 10_000
    profit_buffer = costs["profit_buffer_bps"] / 10_000
    min_continuation = _per_horizon(cfg.get("minimum_confidence", {}), horizon, 0.57)
    min_tradeability = _per_horizon(
        cfg.get("minimum_tradeability_probability", {}), horizon, 0.56
    )
    selected = (
        event_mask
        & (p_continuation >= min_continuation)
        & (p_tradeable >= min_tradeability)
        & (predicted_event_return >= stress_cost + profit_buffer)
    )
    net = actual_event_return[selected] - base_cost
    if len(net) == 0:
        return {
            "selected": 0,
            "mean_gross_return": 0.0,
            "mean_net_return": 0.0,
            "hit_rate": 0.0,
            "cumulative_return": 0.0,
            "positive": False,
        }
    return {
        "selected": int(len(net)),
        "mean_gross_return": float(np.mean(actual_event_return[selected])),
        "mean_net_return": float(np.mean(net)),
        "hit_rate": float(np.mean(net > 0)),
        "cumulative_return": float(np.prod(1 + net) - 1),
        "positive": bool(np.mean(net) > 0),
    }


def qualify_model(metrics: dict[str, Any], settings: Settings) -> dict[str, Any]:
    cfg = settings.section("qualification")
    blockers: list[str] = []
    per_horizon: dict[str, Any] = {}
    qualified_horizons: list[int] = []
    for horizon, item in metrics.items():
        reasons: list[str] = []
        warnings: list[str] = []
        if int(item["event_samples"]) < int(cfg.get("minimum_event_samples", 60)):
            reasons.append("insufficient independent events")
        if float(item["event_auc"]) < float(cfg.get("minimum_event_auc", 0.52)):
            reasons.append("event continuation AUC below minimum")
        if float(item["tradeability_auc"]) < float(cfg.get("minimum_tradeability_auc", 0.53)):
            reasons.append("event tradeability AUC below minimum")
        if float(item.get("event_calibration_error", 1.0)) > float(
            cfg.get("maximum_event_calibration_error", 0.10)
        ):
            reasons.append("event continuation calibration error too high")
        if float(item.get("tradeability_calibration_error", 1.0)) > float(
            cfg.get("maximum_tradeability_calibration_error", 0.10)
        ):
            reasons.append("event tradeability calibration error too high")
        positive_fraction = float(item.get("positive_fold_fraction", 0.0))
        if positive_fraction < float(cfg.get("minimum_positive_fold_fraction", 0.50)):
            reasons.append("too few positive walk-forward folds")
        trading = item.get("trading", {})
        if int(trading.get("selected", 0)) < int(cfg.get("minimum_oof_trades", 10)):
            reasons.append("too few selected OOF trades")
        if int(trading.get("selected", 0)) > 0 and float(trading.get("hit_rate", 0.0)) < float(
            cfg.get("minimum_oof_hit_rate", 0.50)
        ):
            reasons.append("OOF hit rate below minimum")
        if cfg.get("require_positive_oof_expectancy", True) and not bool(trading.get("positive", False)):
            reasons.append("non-positive OOF net expectancy")

        # Generic hourly direction remains useful on the dashboard but does not qualify event trades.
        if float(item.get("auc", 0.5)) < 0.50:
            warnings.append("generic UP/DOWN forecast below random ranking")
        per_horizon[horizon] = {"passed": not reasons, "blockers": reasons, "warnings": warnings}
        if not reasons:
            qualified_horizons.append(int(horizon))
        else:
            blockers.extend([f"h{horizon}: {reason}" for reason in reasons])

    minimum_qualified = int(cfg.get("minimum_qualified_horizons", 1))
    passed = len(qualified_horizons) >= minimum_qualified
    if not passed:
        blockers.insert(0, f"Only {len(qualified_horizons)} qualified horizon(s); {minimum_qualified} required")
    return {
        "passed": passed,
        "qualified_horizons": qualified_horizons,
        "per_horizon": per_horizon,
        "blockers": blockers,
        "checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def _event_type_metrics(
    y_continuation: np.ndarray,
    p_continuation: np.ndarray,
    y_tradeable: np.ndarray,
    p_tradeable: np.ndarray,
    event_mask: np.ndarray,
    event_types: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    event_types_text = event_types.astype(str)
    for event_type in sorted({str(x) for x in event_types_text[event_mask] if str(x) != "NONE"}):
        mask = event_mask & (event_types_text == event_type)
        if mask.sum() < 8:
            continue
        yc = y_continuation[mask].astype(int)
        pc = p_continuation[mask]
        yt = y_tradeable[mask].astype(int)
        pt = p_tradeable[mask]
        output[event_type] = {
            "samples": int(mask.sum()),
            "continuation_auc": _safe_auc(yc, pc),
            "continuation_balanced_accuracy": _safe_balanced_accuracy(yc, (pc >= 0.5).astype(int)),
            "tradeability_auc": _safe_auc(yt, pt),
            "tradeability_rate": float(np.mean(yt)),
        }
    return output


def _expected_calibration_error(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    probability = np.asarray(probability)
    valid = np.isfinite(y_true) & np.isfinite(probability)
    y_true = y_true[valid]
    probability = probability[valid]
    if len(y_true) == 0:
        return 1.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (probability >= edges[idx]) & (probability <= edges[idx + 1])
        else:
            mask = (probability >= edges[idx]) & (probability < edges[idx + 1])
        if not mask.any():
            continue
        error += (mask.sum() / total) * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return float(error)


def _safe_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(probability)
    y = np.asarray(y_true)[valid]
    p = np.asarray(probability)[valid]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def _safe_balanced_accuracy(y_true: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(predicted)
    y = np.asarray(y_true)[valid].astype(int)
    p = np.asarray(predicted)[valid].astype(int)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return 0.5
    return float(balanced_accuracy_score(y, p))


def _per_horizon(value: Any, horizon: int, default: float) -> float:
    if isinstance(value, dict):
        return float(value.get(horizon, value.get(str(horizon), default)))
    if value is None:
        return default
    return float(value)


def _summary_csv(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for horizon, item in metrics.items():
        rows.append(
            {
                "horizon": f"{horizon}h",
                "samples": item.get("samples"),
                "accuracy": item.get("accuracy"),
                "balanced_accuracy": item.get("balanced_accuracy"),
                "calibration_error": item.get("calibration_error"),
                "event_samples": item.get("event_samples"),
                "event_auc": item.get("event_auc"),
                "event_calibration_error": item.get("event_calibration_error"),
                "tradeability_auc": item.get("tradeability_auc"),
                "tradeability_calibration_error": item.get("tradeability_calibration_error"),
                "selected_trades": item.get("trading", {}).get("selected"),
                "mean_net_return": item.get("trading", {}).get("mean_net_return"),
                "hit_rate": item.get("trading", {}).get("hit_rate"),
                "positive_fold_fraction": item.get("positive_fold_fraction"),
            }
        )
    return pd.DataFrame(rows)
