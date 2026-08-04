from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from btc_ema_trader.contract_training import (
    build_segmented_feature_set,
    enrich_interval_metrics,
)
from btc_ema_trader.economic_validation import evaluate_and_patch_candidate
from btc_ema_trader.logging_setup import configure_logging
from btc_ema_trader.market import fetch_and_store
from btc_ema_trader.negative_memory import (
    RESISTANCE,
    SUPPORT,
    save_boundary_memory,
    train_sandwiched_boundary_memory,
)
from btc_ema_trader.news import collect_and_store
from btc_ema_trader.storage import Database
from btc_ema_trader.structure_training import train_feature_set

from github_common import build_github_settings, json_safe, write_json

LOGGER = logging.getLogger("github_weekly_retrain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain a challenger and promote it only after locked economic "
            "and sandwiched negative-memory holdout validation"
        )
    )
    parser.add_argument("--output-dir", default="model-state-output")
    parser.add_argument("--runtime-dir", default=".github_runtime/training")
    parser.add_argument("--incumbent-dir", default="incumbent-model-state")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime_dir = (root / args.runtime_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    incumbent_dir = (root / args.incumbent_dir).resolve()
    model_dir = runtime_dir / "models"
    report_dir = runtime_dir / "reports"

    shutil.rmtree(runtime_dir, ignore_errors=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = build_github_settings(
        root,
        runtime_dir,
        model_dir=model_dir,
        report_dir=report_dir,
    )
    configure_logging(settings, verbose=True)
    database = Database(settings)
    database.initialize()
    market_cfg = settings.section("market")
    history_days = float(market_cfg.get("history_days", 3650))
    strict_gap_hours = float(market_cfg.get("maximum_gap_hours", 3))
    training_gap_hours = float(
        market_cfg.get("training_maximum_gap_hours", 24)
    )
    news_days = float(settings.section("news").get("historical_days", 365))

    started = pd.Timestamp.now(tz="UTC")
    market_cfg["maximum_gap_hours"] = training_gap_hours
    try:
        market = fetch_and_store(
            settings,
            database,
            days=history_days,
            provider=None,
        )
    finally:
        market_cfg["maximum_gap_hours"] = strict_gap_hours

    historical_news = optional(
        lambda: collect_and_store(
            settings,
            database,
            historical=True,
            days=news_days,
        )
    )
    recent_news = optional(
        lambda: collect_and_store(
            settings,
            database,
            historical=False,
        )
    )

    provider = str(market["provider"])
    symbol = str(market_cfg.get("symbol", "BTCUSDT"))
    candles = database.load_candles(provider=provider, symbol=symbol)
    cutoff = candles["open_time"].max() - pd.Timedelta(days=history_days)
    candles = candles[candles["open_time"] >= cutoff].reset_index(drop=True)
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
    training = train_feature_set(
        settings,
        feature_set,
        provider=provider,
        symbol=symbol,
    )
    training["market_data_segmentation"] = segmentation
    training = enrich_interval_metrics(settings, training)

    latest_model = model_dir / "latest.joblib"
    latest_report = report_dir / "latest_training_report.json"
    if not latest_model.exists() or not latest_report.exists():
        raise FileNotFoundError(
            "Training finished without the candidate model/report artifacts"
        )
    model_id = str(training.get("model_id") or "")
    oof_path = report_dir / f"{model_id}_oof.csv"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"Training finished without chronological OOF records: {oof_path}"
        )

    memory, memory_report, memory_oof = train_sandwiched_boundary_memory(
        feature_set.frame,
        settings,
        horizons=settings.section("model").get(
            "trade_horizons_hours", [3, 6, 12]
        ),
        feature_columns=feature_set.feature_columns,
        model_id=model_id,
    )
    candidate_memory = runtime_dir / "negative_memory.joblib"
    save_boundary_memory(memory, candidate_memory)
    write_json(report_dir / "negative_memory_report.json", memory_report)
    if not memory_oof.empty:
        memory_oof.to_csv(
            report_dir / f"{model_id}_negative_memory_oof.csv",
            index=False,
        )

    economic_path = output_dir / "economic_validation.json"
    incumbent_economic = incumbent_dir / "economic_validation.json"
    economic = evaluate_and_patch_candidate(
        report_path=latest_report,
        oof_path=oof_path,
        model_path=latest_model,
        output_path=economic_path,
        incumbent_economic_report=incumbent_economic,
    )
    promotion = dict(economic["promotion"])
    joint_pairs = _joint_economic_memory_pairs(economic, memory_report)
    memory_report["joint_economic_pairs"] = joint_pairs
    memory_gate_passed = bool(joint_pairs)
    memory_report["joint_economic_gate_passed"] = memory_gate_passed
    write_json(report_dir / "negative_memory_report.json", memory_report)

    decision = str(promotion.get("decision") or "KEEP_INCUMBENT")
    if decision == "PROMOTE" and not memory_gate_passed:
        decision = "KEEP_INCUMBENT"
        promotion["decision"] = decision
        promotion["reason"] = (
            "Candidate passed the economic gate but no direction/horizon "
            "also passed the sandwiched negative-memory holdout gate"
        )
    economic["promotion"] = promotion
    economic["joint_economic_memory_pairs"] = joint_pairs
    economic["joint_economic_memory_gate_passed"] = memory_gate_passed
    write_json(economic_path, economic)

    # Candidate diagnostics are always retained. Production-facing names are
    # emitted only for a jointly promoted economic + negative-memory challenger.
    shutil.copy2(latest_model, output_dir / "candidate_latest.joblib")
    shutil.copy2(candidate_memory, output_dir / "candidate_negative_memory.joblib")
    shutil.copy2(
        latest_report,
        output_dir / "candidate_training_report.json",
    )
    shutil.copy2(
        report_dir / "negative_memory_report.json",
        output_dir / "candidate_negative_memory_report.json",
    )
    memory_oof_path = report_dir / f"{model_id}_negative_memory_oof.csv"
    if memory_oof_path.exists():
        shutil.copy2(
            memory_oof_path,
            output_dir / "candidate_negative_memory_oof.csv",
        )
    metrics_path = report_dir / "latest_metrics.csv"
    if metrics_path.exists():
        shutil.copy2(metrics_path, output_dir / "candidate_metrics.csv")
    if decision == "PROMOTE":
        shutil.copy2(latest_model, output_dir / "latest.joblib")
        shutil.copy2(candidate_memory, output_dir / "negative_memory.joblib")
        shutil.copy2(latest_report, output_dir / "latest_training_report.json")
        shutil.copy2(
            report_dir / "negative_memory_report.json",
            output_dir / "negative_memory_report.json",
        )
        if metrics_path.exists():
            shutil.copy2(metrics_path, output_dir / "latest_metrics.csv")

    finished = pd.Timestamp.now(tz="UTC")
    metadata = json_safe(
        {
            "started_at": started,
            "finished_at": finished,
            "duration_seconds": (finished - started).total_seconds(),
            "market": market,
            "market_validation": {
                "runtime_maximum_gap_hours": strict_gap_hours,
                "weekly_fetch_maximum_gap_hours": training_gap_hours,
                "training_gap_policy": "CONTIGUOUS_SEGMENTS_WITHOUT_FILL",
            },
            "historical_news": historical_news,
            "recent_news": recent_news,
            "training": training,
            "negative_memory": memory_report,
            "economic_validation": economic,
            "promotion": promotion,
            "training_history_days": history_days,
            "news_history_days": news_days,
            "forecast_target": "NEXT_CLOSED_1H_CANDLE",
            "trade_setup": "SANDWICHED_ECONOMIC_BOUNDARY_BREAKOUT",
            "general_label": "CLOSE_TO_CLOSE_RETURN",
            "sampling_strategy": "NONE",
            "synthetic_events": 0,
            "synthetic_candles": 0,
        }
    )
    write_json(output_dir / "model_metadata.json", metadata)
    write_json(
        output_dir / "promotion.json",
        {
            "decision": decision,
            "reason": promotion.get("reason"),
            "candidate_model_id": model_id,
            "candidate_score_bps": promotion.get("candidate_score_bps"),
            "incumbent_score_bps": promotion.get("incumbent_score_bps"),
            "joint_economic_memory_pairs": joint_pairs,
            "checked_at": finished.isoformat(),
        },
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def _joint_economic_memory_pairs(
    economic: dict,
    memory_report: dict,
) -> list[dict[str, object]]:
    qualified = (
        economic.get("qualification", {})
        .get("qualified_directions", {})
    )
    mapping = {"LONG": RESISTANCE, "SHORT": SUPPORT}
    pairs: list[dict[str, object]] = []
    for direction, side in mapping.items():
        for raw_horizon in qualified.get(direction, []):
            horizon = int(raw_horizon)
            item = (
                memory_report.get("sides", {})
                .get(side, {})
                .get(str(horizon), {})
            )
            if bool(item.get("qualified", False)):
                pairs.append(
                    {
                        "direction": direction,
                        "boundary_side": side,
                        "horizon": horizon,
                    }
                )
    return pairs


def optional(callback):
    try:
        return callback()
    except Exception as exc:
        LOGGER.warning("Optional step failed: %s", exc)
        return {
            "status": "warning",
            "error": f"{type(exc).__name__}: {exc}",
        }


if __name__ == "__main__":
    raise SystemExit(main())
