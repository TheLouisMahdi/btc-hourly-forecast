from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from .config import load_settings
from .meta_filter import (
    apply_precision_gate,
    load_precision_meta_filter,
)
from .pattern_memory import (
    LiveCandlePatternMemory,
    adjust_forecast_with_pattern_memory,
    load_static_pattern_bundle,
)

LOGGER = logging.getLogger(__name__)
_INSTALLED = False
_ORIGINAL_ENRICH: Callable[..., Any] | None = None
_ORIGINAL_STRICT_FORECAST: Callable[..., Any] | None = None


def install_trade_assistant_runtime() -> None:
    """Install a runtime overlay without changing the incumbent model.

    If new artifacts have not been trained yet, the one-hour forecast stays
    available while new position entries are treated as experimental and
    blocked. Existing open positions are never rewritten.
    """
    global _INSTALLED, _ORIGINAL_ENRICH, _ORIGINAL_STRICT_FORECAST
    if _INSTALLED:
        return

    from . import github_runtime as runtime_module

    original_enrich = runtime_module.CanonicalAdaptiveTradeEngine.enrich_trade_plan
    original_strict = runtime_module.build_strict_next_candle_forecast
    _ORIGINAL_ENRICH = original_enrich
    _ORIGINAL_STRICT_FORECAST = original_strict

    def assistant_enrich(self, record, plan):
        candidate = original_enrich(self, record, plan)

        # Active-position contracts are immutable. The new architecture only
        # controls future entries and must not move an already-open target/stop.
        if (
            record.get("active_position_contract")
            or candidate.get("contract_type") == "ACTIVE_TARGET_STOP_POSITION"
            or candidate.get("status") == "MANAGING_OPEN_TRADE"
        ):
            record["trade_assistant"] = {
                "status": "ACTIVE_POSITION_PRESERVED",
                "selected": False,
                "reason": "EXISTING_POSITION_CONTRACT_IS_IMMUTABLE",
            }
            return candidate

        action = str(record.get("action") or "").upper()
        if action not in {"LONG", "SHORT"}:
            record["trade_assistant"] = {
                "status": "NO_POSITION_CANDIDATE",
                "selected": False,
                "reason": "NO_ACTIONABLE_DIRECTIONAL_EVENT",
            }
            return candidate

        meta, patterns, artifact_status = _artifacts_for_record(record)
        if meta is None:
            assessment = {
                "status": "UNAVAILABLE",
                "qualified": False,
                "selected": False,
                "reason": artifact_status,
            }
        else:
            assessment = meta.assess(record, patterns)

        gated, blockers = apply_precision_gate(
            record,
            candidate,
            assessment,
            self.settings,
        )
        record["trade_assistant"] = assessment
        if blockers:
            record["candidate_action_before_meta_gate"] = action
            record["candidate_trade_plan_before_meta_gate"] = candidate
            record["action"] = "WAIT"
            record["blockers"] = list(
                dict.fromkeys(list(record.get("blockers", [])) + blockers)
            )
        return gated

    def assistant_strict_forecast(
        record,
        model_metrics,
        recent_candles,
        history,
        interval_probability=0.80,
    ):
        probability_source = record.get(
            "general_probabilities",
            record.get("probabilities"),
        )
        raw_probability = _mapping_value(probability_source, 1, 0.5)
        raw_direction = "UP" if raw_probability >= 0.5 else "DOWN"

        _, patterns, _ = _artifacts_for_record(record)
        static = (
            patterns.assess_record(record, horizon=1)
            if patterns is not None
            else None
        )
        live = _live_memory()
        if live is not None:
            try:
                live.synchronize(history)
                live_assessment = live.assess(
                    record,
                    direction=raw_direction,
                )
            except Exception:
                LOGGER.exception("Live candle-pattern memory failed")
                live_assessment = None
        else:
            live_assessment = None

        prediction = record.get("price_forecast_model")
        prediction = prediction if isinstance(prediction, dict) else {}
        settings = _runtime_settings()
        if prediction and settings is not None:
            adjusted = adjust_forecast_with_pattern_memory(
                prediction,
                record=record,
                static=static,
                live=live_assessment,
                settings=settings,
            )
            record["price_forecast_model"] = adjusted
            record.setdefault("general_probabilities", {})["1"] = float(
                adjusted.get("fused_probability_up", raw_probability)
            )
            record["candle_pattern_memory"] = {
                "static": static,
                "live": live_assessment,
                "summary": live.summary() if live is not None else None,
            }

        contract = original_strict(
            record,
            model_metrics,
            recent_candles,
            history,
            interval_probability=interval_probability,
        )
        adjustment = record.get("price_forecast_model", {}).get(
            "pattern_memory_adjustment"
        )
        if isinstance(adjustment, dict):
            contract["pattern_memory_adjustment"] = adjustment
        return contract

    assistant_enrich._trade_assistant_overlay = True
    assistant_strict_forecast._trade_assistant_overlay = True
    runtime_module.CanonicalAdaptiveTradeEngine.enrich_trade_plan = assistant_enrich
    runtime_module.build_strict_next_candle_forecast = assistant_strict_forecast
    _INSTALLED = True


def _artifacts_for_record(record: dict[str, Any]):
    root = Path.cwd()
    model_id = str(record.get("model_id") or "")
    meta_path = root / ".model_state" / "trade_assistant_meta.joblib"
    pattern_path = root / ".model_state" / "trade_assistant_patterns.joblib"

    try:
        meta = load_precision_meta_filter(meta_path) if meta_path.exists() else None
    except Exception:
        LOGGER.exception("Trade-assistant meta artifact could not be loaded")
        meta = None
    try:
        patterns = (
            load_static_pattern_bundle(pattern_path)
            if pattern_path.exists()
            else None
        )
    except Exception:
        LOGGER.exception("Trade-assistant pattern artifact could not be loaded")
        patterns = None

    if meta is not None and model_id and meta.model_id != model_id:
        meta = None
        status = "META_MODEL_ID_MISMATCH"
    elif meta is None:
        status = "META_FILTER_NOT_TRAINED"
    else:
        status = "READY"

    if patterns is not None and model_id and patterns.model_id != model_id:
        patterns = None
    return meta, patterns, status


_SETTINGS = None
_LIVE_MEMORY = None


def _runtime_settings():
    global _SETTINGS
    if _SETTINGS is not None:
        return _SETTINGS
    path = Path.cwd() / "config" / "default.yaml"
    try:
        _SETTINGS = load_settings(path)
    except Exception:
        LOGGER.exception("Trade-assistant settings could not be loaded")
        _SETTINGS = None
    return _SETTINGS


def _live_memory():
    global _LIVE_MEMORY
    if _LIVE_MEMORY is not None:
        return _LIVE_MEMORY
    settings = _runtime_settings()
    if settings is None:
        return None
    path = Path.cwd() / ".adaptive_state" / "candle_pattern_memory.joblib"
    _LIVE_MEMORY = LiveCandlePatternMemory(settings, path)
    return _LIVE_MEMORY


def _mapping_value(value: Any, key: int, default: float) -> float:
    if not isinstance(value, dict):
        return float(default)
    raw = value.get(key, value.get(str(key), default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)
