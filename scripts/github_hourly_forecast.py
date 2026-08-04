from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from btc_ema_trader.contract_features import build_feature_set
from btc_ema_trader.forecast_contract import (
    attach_close_based_general_labels,
    build_next_candle_forecast,
)
from btc_ema_trader.logging_setup import configure_logging
from btc_ema_trader.market import fetch_and_store
from btc_ema_trader.model import latest_bundle
from btc_ema_trader.negative_memory import (
    install_runtime_guard,
    load_boundary_memory,
)
from btc_ema_trader.price_adaptive import PriceAdaptiveEngine
from btc_ema_trader.runtime import RuntimeEngine
from btc_ema_trader.storage import Database

from github_common import (
    build_github_settings,
    copy_latest_model_from_state,
    json_safe,
    write_json,
)

LOGGER = logging.getLogger("github_hourly_forecast")
MAX_HISTORY = 24 * 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one adaptive hourly forecast with sandwiched negative-memory "
            "protection and persist its state"
        )
    )
    parser.add_argument("--state-dir", default=".github_state")
    parser.add_argument("--model-state-dir", default=".model_state")
    parser.add_argument("--adaptive-state-dir", default=".adaptive_state")
    parser.add_argument("--site-dir", default="site")
    parser.add_argument("--runtime-dir", default=".github_runtime/hourly")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    state_dir = (root / args.state_dir).resolve()
    model_state_dir = (root / args.model_state_dir).resolve()
    adaptive_state_dir = (root / args.adaptive_state_dir).resolve()
    site_dir = (root / args.site_dir).resolve()
    runtime_dir = (root / args.runtime_dir).resolve()

    shutil.rmtree(runtime_dir, ignore_errors=True)
    shutil.rmtree(site_dir, ignore_errors=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    adaptive_state_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)

    history_path = state_dir / "history.json"
    previous_history = load_history(history_path)
    used_weekly_model = copy_latest_model_from_state(
        root,
        model_state_dir,
    )
    settings = build_github_settings(
        root,
        runtime_dir,
        adaptive_state_dir=adaptive_state_dir,
    )
    configure_logging(settings, verbose=True)
    database = Database(settings)
    database.initialize()

    memory = None
    memory_error = None
    memory_path = model_state_dir / "negative_memory.joblib"
    if memory_path.exists():
        try:
            memory = load_boundary_memory(memory_path)
        except Exception as exc:
            memory_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Negative-memory artifact could not be loaded")
    require_memory = bool(
        settings.section("negative_memory").get("require_for_trade", True)
    )
    install_runtime_guard(memory, require_for_trade=require_memory)

    started_at = pd.Timestamp.now(tz="UTC")
    price_summary: dict[str, Any] = {"status": "UNAVAILABLE"}
    try:
        bundle = latest_bundle(settings)
        market = fetch_and_store(
            settings,
            database,
            days=180,
            provider=bundle.provider,
        )
        result = RuntimeEngine(
            settings,
            database,
        ).run_once(force=True)
        boundary = (
            result.get("trade_plan", {}).get("boundary_memory")
            if isinstance(result.get("trade_plan"), dict)
            else None
        )
        if isinstance(boundary, dict):
            result["boundary_memory"] = boundary
        status = (
            "OK"
            if result.get("status") != "FAIL_SAFE"
            else "FAIL_SAFE"
        )
        if status == "OK" and result.get("candle_time"):
            price_prediction = build_price_prediction(
                settings,
                database,
                bundle,
            )
            price_summary = price_prediction
            result["price_forecast_model"] = price_prediction
            result["general_probabilities"] = {
                "1": price_prediction["fused_probability_up"]
            }
            result["general_return_estimates"] = {
                "1": price_prediction["fused_return"]
            }
            interval_probability = float(
                settings.section("forecast").get(
                    "interval_probability",
                    0.80,
                )
            )
            recent_candles = database.load_candles(
                provider=bundle.provider,
                symbol=bundle.symbol,
                limit=168,
            )
            contract = build_next_candle_forecast(
                result,
                load_contract_metrics(settings, bundle.metrics),
                recent_candles,
                previous_history,
                interval_probability=interval_probability,
            )
            result = attach_forecast_contract(result, contract)
    except Exception as exc:
        LOGGER.exception("Hourly forecast failed")
        market = None
        result = {
            "status": "FAIL_SAFE",
            "error": f"{type(exc).__name__}: {exc}",
        }
        status = "FAIL_SAFE"

    finished_at = pd.Timestamp.now(tz="UTC")
    fresh_record = json_safe(
        {
            **result,
            "run_status": status,
            "run_started_at": started_at,
            "run_finished_at": finished_at,
            "run_duration_seconds": (
                finished_at - started_at
            ).total_seconds(),
            "market_refresh": market,
            "weekly_model_loaded": used_weekly_model,
            "negative_memory_loaded": memory is not None,
            "negative_memory_model_id": (
                None if memory is None else memory.model_id
            ),
            "negative_memory_error": memory_error,
        }
    )
    fresh_adaptive_summary = fresh_record.get(
        "adaptive",
        {"status": "UNAVAILABLE"},
    )
    record = preserve_existing_forecast(
        previous_history,
        fresh_record,
    )

    history = append_unique(
        previous_history,
        record,
    )[-MAX_HISTORY:]
    write_json(state_dir / "latest.json", record)
    write_json(history_path, history)
    write_json(site_dir / "latest.json", record)
    write_json(site_dir / "history.json", history)
    write_json(
        adaptive_state_dir / "summary.json",
        fresh_adaptive_summary,
    )
    write_json(
        adaptive_state_dir / "price_summary.json",
        price_summary,
    )
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(json.dumps(record, ensure_ascii=False, indent=2))
    if status != "OK":
        print(
            "::warning::Forecast completed in FAIL_SAFE mode; "
            "diagnostics were persisted."
        )
    return 0


def build_price_prediction(
    settings,
    database: Database,
    bundle,
) -> dict[str, Any]:
    candles = database.load_candles(
        provider=bundle.provider,
        symbol=bundle.symbol,
    )
    news = database.load_news(
        start=candles["open_time"].min(),
        end=pd.Timestamp.now(tz="UTC"),
    )
    feature_set = build_feature_set(
        candles,
        news,
        settings,
        include_labels=True,
    )
    prepared = attach_close_based_general_labels(
        feature_set.frame,
        feature_set.horizons,
    )
    usable = prepared.dropna(
        subset=["kama", "donchian_mid", "atr", "adx"]
    )
    if usable.empty:
        raise RuntimeError(
            "No closed candle is available for the price forecast"
        )
    latest_row = usable.iloc[-1]
    base_prediction = bundle.predict_frame(usable.tail(1))
    base_probability_up = float(base_prediction["probabilities"][1])
    base_return = float(base_prediction["returns"][1])
    engine = PriceAdaptiveEngine(settings, bundle)
    engine.synchronize(prepared)
    return engine.predict(
        latest_row,
        base_probability_up,
        base_return,
    )


def load_contract_metrics(
    settings,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    path = settings.path("report_dir") / "latest_training_report.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report.get("metrics")
        return metrics if isinstance(metrics, dict) else fallback
    except Exception:
        return fallback


def attach_forecast_contract(
    result: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    output = dict(result)
    output["trade_forecast_direction"] = output.get(
        "forecast_direction"
    )
    output["trade_selected_horizon"] = output.get(
        "selected_horizon"
    )
    output["trade_confidence"] = output.get("confidence")
    output["next_candle_forecast"] = contract
    output["forecast_contract_version"] = contract[
        "contract_version"
    ]
    output["forecast_direction"] = contract["direction"]
    output["selected_horizon"] = 1
    output["confidence"] = contract["direction_confidence"]
    output["expected_return"] = contract["median_return"]
    output["target_candle_time"] = contract["target_open_time"]
    output["target_candle_open_time"] = contract[
        "target_open_time"
    ]
    output["target_candle_close_time"] = contract[
        "target_close_time"
    ]
    output["predicted_close_median"] = contract["median_close"]
    output["predicted_close_low"] = contract[
        "likely_close_low"
    ]
    output["predicted_close_high"] = contract[
        "likely_close_high"
    ]
    output["prediction_result"] = "PENDING"
    output["direction_result"] = "PENDING"
    output["interval_result"] = "PENDING"
    output["resolved_at"] = None
    output["forecast_frozen"] = True
    return output


def preserve_existing_forecast(
    history: list[dict[str, Any]],
    record: dict[str, Any],
) -> dict[str, Any]:
    key = record.get("candle_time")
    if not key:
        return record
    existing = next(
        (
            item
            for item in reversed(history)
            if item.get("candle_time") == key
            and isinstance(
                item.get("next_candle_forecast"),
                dict,
            )
            and item.get("run_status") == "OK"
        ),
        None,
    )
    if existing is None:
        return record
    return dict(existing)


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def append_unique(
    history: list[dict[str, Any]],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    key = record.get("candle_time") or record.get(
        "run_finished_at"
    )
    filtered = [
        item
        for item in history
        if (
            item.get("candle_time")
            or item.get("run_finished_at")
        )
        != key
    ]
    filtered.append(record)
    return sorted(
        filtered,
        key=lambda item: str(
            item.get("candle_time")
            or item.get("run_finished_at")
            or ""
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
