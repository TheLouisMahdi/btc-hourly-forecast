from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from btc_ema_trader.contract_features import build_feature_set
from btc_ema_trader.execution_path import resolve_open_trades_after_entry
from btc_ema_trader.github_runtime import (
    CanonicalAdaptiveTradeEngine,
    CanonicalRuntimeEngine,
    attach_optional_forecast,
    build_optional_secondary_forecast,
    open_trade_with_context,
    optional_price_prediction,
    preserve_canonical_forecast,
    retain_directional_history,
)
from btc_ema_trader.logging_setup import configure_logging
from btc_ema_trader.market import MarketDataClient
from btc_ema_trader.model import latest_bundle
from btc_ema_trader.negative_memory import (
    install_runtime_guard,
    load_boundary_memory,
)
from btc_ema_trader.price_adaptive import PriceAdaptiveEngine
from btc_ema_trader.runtime_history import fetch_latest_contiguous_and_store
from btc_ema_trader.sample_policy import filter_history_for_model_policy
from btc_ema_trader.storage import Database
from btc_ema_trader.trade_lifecycle import active_trade

from github_common import (
    build_github_settings,
    copy_latest_model_from_state,
    json_safe,
    write_json,
)

LOGGER = logging.getLogger("github_hourly_forecast")
MAX_HISTORY = 24 * 30
MAX_TRADES = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one canonical aggressive paper-trade cycle with causal "
            "context, live-quote entry, adaptive exits and immutable outcomes"
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
    latest_path = state_dir / "latest.json"
    trades_path = state_dir / "trades.json"
    raw_history = load_history(history_path)
    previous_history = filter_history_for_model_policy(
        retain_directional_history(raw_history)
    )
    if previous_history != raw_history:
        write_json(history_path, previous_history)
    previous_latest = load_record(latest_path)
    if previous_latest and not filter_history_for_model_policy(
        retain_directional_history([previous_latest])
    ):
        write_json(latest_path, {})
    trades = load_history(trades_path)

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

    display_quote: dict[str, Any] | None = None
    try:
        quote = MarketDataClient(settings).live_quote()
        display_quote = {
            "provider": quote.provider,
            "symbol": quote.symbol,
            "price": float(quote.price),
            "timestamp": quote.timestamp,
        }
    except Exception as exc:
        LOGGER.warning("Independent display quote unavailable: %s", exc)

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
        settings.section("negative_memory").get("require_for_trade", False)
    )
    install_runtime_guard(memory, require_for_trade=require_memory)

    started_at = pd.Timestamp.now(tz="UTC")
    price_summary: dict[str, Any] = {"status": "UNAVAILABLE"}
    trade_summary: dict[str, Any] = {"status": "UNAVAILABLE"}
    resolved_trades_now = 0
    opened_trade_id: str | None = None
    try:
        bundle = latest_bundle(settings)
        market = fetch_latest_contiguous_and_store(
            settings,
            database,
            days=180,
            provider=bundle.provider,
        )
        candles = database.load_candles(
            provider=bundle.provider,
            symbol=bundle.symbol,
        )

        # Rebuild path-dependent stop state from its immutable entry values
        # before deterministic post-entry replay.
        for trade in trades:
            if trade.get("status") == "OPEN":
                trade["current_stop_price"] = trade.get(
                    "initial_stop_price",
                    trade.get("current_stop_price"),
                )
                trade["max_favorable_r"] = 0.0
                trade["max_adverse_r"] = 0.0
                trade["breakeven_armed"] = False
                trade["trailing_armed"] = False
        resolved_trades_now = resolve_open_trades_after_entry(
            trades,
            candles,
            settings,
        )

        trade_engine = CanonicalAdaptiveTradeEngine(
            settings,
            bundle.model_id,
        )
        trade_summary = trade_engine.synchronize(trades)

        result = CanonicalRuntimeEngine(
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
            result["run_finished_at"] = pd.Timestamp.now(tz="UTC").isoformat()
            raw_plan = result.get("trade_plan")
            raw_plan = raw_plan if isinstance(raw_plan, dict) else {}
            result["trade_plan"] = trade_engine.enrich_trade_plan(
                result,
                raw_plan,
            )

            current_trade = active_trade(trades)
            if current_trade is not None:
                if result.get("action") in {"LONG", "SHORT"}:
                    result["blockers"] = list(
                        dict.fromkeys(
                            list(result.get("blockers", []))
                            + ["ACTIVE_TRADE_IN_PROGRESS"]
                        )
                    )
                result["action"] = (
                    "HOLD_LONG"
                    if current_trade.get("direction") == "LONG"
                    else "HOLD_SHORT"
                )
                result["trade_plan"]["status"] = "MANAGING_OPEN_TRADE"
            elif result.get("action") in {"LONG", "SHORT"}:
                candidate = open_trade_with_context(result)
                if candidate is not None and not any(
                    item.get("trade_id") == candidate.get("trade_id")
                    for item in trades
                ):
                    trades.append(candidate)
                    opened_trade_id = str(candidate["trade_id"])
                current_trade = active_trade(trades)

            trade_summary = trade_engine.summary(trades)
            result["active_trade"] = current_trade
            result["trade_lifecycle_summary"] = trade_summary
            result["trade_resolved_now"] = resolved_trades_now
            result["trade_opened_now"] = opened_trade_id
            result["trade_contract"] = "TARGET_STOP_TIME_EXIT"

            price_prediction = optional_price_prediction(
                build_price_prediction,
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
            contract = build_optional_secondary_forecast(
                result,
                load_contract_metrics(settings, bundle.metrics),
                candles.tail(168),
                previous_history,
                interval_probability=interval_probability,
            )
            result = attach_optional_forecast(result, contract)
    except Exception as exc:
        LOGGER.exception("Hourly trade lifecycle failed")
        market = None
        result = {
            "status": "FAIL_SAFE",
            "error": f"{type(exc).__name__}: {exc}",
            "active_trade": active_trade(trades),
            "trade_lifecycle_summary": trade_summary,
            "trade_resolved_now": resolved_trades_now,
        }
        status = "FAIL_SAFE"

    if display_quote is not None and result.get("price") is None:
        result["price"] = display_quote["price"]
        result["display_price_provider"] = display_quote["provider"]
        result["display_price_symbol"] = display_quote["symbol"]
        result["display_price_time"] = display_quote["timestamp"]

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
            "paper_trade_mode": "AGGRESSIVE_STRUCTURAL_RISK_SCALED",
            "runtime_contract": "CANONICAL_GITHUB_RUNTIME_V2",
        }
    )
    fresh_adaptive_summary = fresh_record.get(
        "adaptive",
        {"status": "UNAVAILABLE"},
    )
    excluded_cycle = result.get("status") == "MODEL_SAMPLE_EXCLUDED"
    if excluded_cycle and previous_history:
        record = dict(previous_history[-1])
        history = previous_history[-MAX_HISTORY:]
    else:
        record = preserve_canonical_forecast(
            previous_history,
            fresh_record,
        )
        history = append_unique(previous_history, record)[-MAX_HISTORY:]

    if display_quote is not None and record.get("price") is None:
        record["price"] = float(display_quote["price"])
        record["display_price_provider"] = display_quote["provider"]
        record["display_price_symbol"] = display_quote["symbol"]
        record["display_price_time"] = json_safe(display_quote["timestamp"])

    trades = trades[-MAX_TRADES:]
    write_json(latest_path, record)
    write_json(history_path, history)
    write_json(trades_path, trades)
    write_json(site_dir / "latest.json", record)
    write_json(site_dir / "history.json", history)
    write_json(site_dir / "trades.json", trades)
    write_json(
        adaptive_state_dir / "summary.json",
        fresh_adaptive_summary,
    )
    write_json(
        adaptive_state_dir / "price_summary.json",
        price_summary,
    )
    write_json(
        adaptive_state_dir / "trade_summary.json",
        trade_summary,
    )
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(json.dumps(fresh_record if excluded_cycle else record, ensure_ascii=False, indent=2))
    if status != "OK":
        print(
            "::warning::Trade cycle completed in FAIL_SAFE mode; "
            "the persistent trade ledger was preserved."
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
    prepared = feature_set.frame.copy()
    usable = prepared.dropna(
        subset=["kama", "donchian_mid", "atr", "adx"]
    )
    if "model_sample_eligible" in usable:
        usable = usable.loc[usable["model_sample_eligible"].astype(bool)]
    if usable.empty:
        raise RuntimeError(
            "No eligible closed candle is available for the price forecast"
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


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def load_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def append_unique(
    history: list[dict[str, Any]],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    key = record.get("candle_time") or record.get("run_finished_at")
    filtered = [
        item
        for item in history
        if (item.get("candle_time") or item.get("run_finished_at")) != key
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
