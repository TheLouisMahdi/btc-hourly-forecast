from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import pandas as pd

from .active_position_contract import (
    ACTIVE_POSITION_CONTRACT,
    build_active_position_plan,
)
from .candle_context import CONTEXT_CONTRACT, extract_candle_context
from .context_trade_features import (
    EXTENDED_TRADE_FEATURES,
    append_context_features,
    migrate_trade_feature_vectors,
)
from .execution_entry import apply_execution_quote
from .execution_path import install_execution_path_contract
from .risk_economics import apply_risk_scaled_economics
from .runtime import RuntimeEngine
from .strict_forecast_contract import build_strict_next_candle_forecast
from .trade_lifecycle import (
    AdaptiveTradeEngine,
    TradeAdaptiveState,
    active_trade,
    open_trade_from_record,
)

MODEL_PREFIX = "directional-breakout-hourly-"
TRADE_STATE_SCHEMA_VERSION = 2
RUNTIME_CONTRACT = "CANONICAL_GITHUB_RUNTIME_V2"

FORECAST_IMMUTABLE_FIELDS = {
    "next_candle_forecast",
    "forecast_contract_version",
    "next_candle_direction",
    "next_candle_confidence",
    "next_candle_expected_return",
    "target_candle_time",
    "target_candle_open_time",
    "target_candle_close_time",
    "predicted_close_median",
    "predicted_close_low",
    "predicted_close_high",
    "prediction_result",
    "direction_result",
    "interval_result",
    "actual_close",
    "actual_price",
    "actual_close_return",
    "actual_return",
    "actual_direction",
    "actual_candle_open",
    "actual_candle_high",
    "actual_candle_low",
    "resolved_at",
    "evaluation_available_at",
    "seconds_until_evaluation",
    "forecast_frozen",
    "secondary_forecast_status",
    "secondary_forecast_error",
    "secondary_forecast_timing_status",
}

POSITION_PLAN_FIELDS = (
    "entry_definition",
    "entry_reference_kind",
    "entry_quote_provider",
    "entry_quote_time",
    "entry_quote_observed_at",
    "source_candle_close",
    "policy_name",
    "policy_version",
    "risk_contract_version",
    "entry_contract",
    "risk_score",
    "risk_fraction",
    "risk_assessment",
    "soft_risk_flags",
    "qualification_passed",
    "direction_qualified",
    "modeled_risk_fraction",
    "modeled_total_risk_usd",
    "risk_budget_utilization",
    "gap_risk_buffer_bps",
    "label_execution_aligned",
    "label_entry_definition",
    "runtime_entry_definition",
    "execution_alignment_status",
)


class CanonicalRuntimeEngine(RuntimeEngine):
    """Bind core structural decisions to causal context and a fresh quote."""

    def run_once(self, force: bool = False) -> dict[str, Any]:
        result = super().run_once(force=force)
        if not isinstance(result, dict) or not result.get("candle_time"):
            return result
        if result.get("status") == "FAIL_SAFE":
            return result

        provider = str(result.get("provider") or "") or None
        symbol = str(
            self.settings.section("market").get("symbol", "BTCUSDT")
        )
        candles = self.database.load_candles(
            provider=provider,
            symbol=symbol,
        )
        context = extract_candle_context(
            candles,
            result["candle_time"],
            previous_bars=2,
        )
        result["event_candle_context"] = context
        result["candle_context_contract"] = CONTEXT_CONTRACT
        result["candle_context_complete"] = bool(
            context.get("complete", False)
        )

        observed_at = pd.Timestamp.now(tz="UTC")
        try:
            quote = self.market.live_quote(provider_hint=provider)
            return apply_execution_quote(
                result,
                provider=quote.provider,
                price=quote.price,
                quote_time=quote.timestamp,
                observed_at=observed_at,
                maximum_age_seconds=float(
                    self.settings.section("market").get(
                        "quote_stale_seconds",
                        90,
                    )
                ),
            )
        except Exception as exc:
            result["execution_quote"] = {
                "contract": "LIVE_QUOTE_AT_SIGNAL_RUN",
                "fresh": False,
                "error": f"{type(exc).__name__}: {exc}",
                "observed_at": observed_at.isoformat(),
            }
            result["blockers"] = list(
                dict.fromkeys(
                    list(result.get("blockers", []))
                    + ["EXECUTION_QUOTE_UNAVAILABLE"]
                )
            )
            if result.get("action") in {"LONG", "SHORT"}:
                result["candidate_action_before_quote_block"] = result.get(
                    "action"
                )
                result["action"] = "WAIT"
                plan = result.get("trade_plan")
                if isinstance(plan, dict):
                    blocked = dict(plan)
                    blocked["status"] = "BLOCKED"
                    blocked["entry_reference_kind"] = (
                        "EXECUTION_QUOTE_UNAVAILABLE"
                    )
                    result["trade_plan"] = blocked
            return result


class CanonicalAdaptiveTradeEngine(AdaptiveTradeEngine):
    """Context-aware adaptive exits with an explicit active-position contract."""

    def __init__(self, *args, **kwargs) -> None:
        self._feature_record: dict[str, Any] | None = None
        self._last_extended_vector: np.ndarray | None = None
        self._active_position: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    def _load_or_create(self) -> TradeAdaptiveState:
        if self.enabled and self.path.exists():
            try:
                state = joblib.load(self.path)
                if (
                    isinstance(state, TradeAdaptiveState)
                    and state.schema_version == TRADE_STATE_SCHEMA_VERSION
                ):
                    return state
            except Exception:
                pass
        now = pd.Timestamp.now(tz="UTC").isoformat()
        return TradeAdaptiveState(
            schema_version=TRADE_STATE_SCHEMA_VERSION,
            created_at=now,
            updated_at=now,
        )

    def synchronize(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Migrate old vectors and learn exactly once from resolved positions."""
        migrate_trade_feature_vectors(trades)
        self._active_position = active_trade(trades)
        learned = 0
        if not self.enabled:
            return self.summary(trades, learned_now=0)

        for trade in trades:
            trade_id = str(trade.get("trade_id") or "")
            if (
                not trade_id
                or trade_id in self.state.learned_trade_ids
                or trade.get("status") != "CLOSED"
            ):
                continue
            vector = np.asarray(
                trade.get("entry_feature_vector", []),
                dtype=float,
            )
            if (
                vector.shape != (len(EXTENDED_TRADE_FEATURES),)
                or not np.isfinite(vector).all()
            ):
                continue

            outcome = str(trade.get("outcome") or "")
            target = int(outcome == "TARGET")
            stop = int(outcome == "STOP")
            realized_r = _finite(trade.get("realized_r"), 0.0)
            weight = 1.0 + min(3.0, abs(realized_r))
            if stop:
                weight *= float(
                    self.cfg.get("stop_learning_weight", 1.75)
                )

            matrix = vector.reshape(1, -1)
            self.state.scaler.partial_fit(matrix)
            transformed = self.state.scaler.transform(matrix)
            classifier_kwargs: dict[str, Any] = {
                "sample_weight": np.asarray([weight], dtype=float)
            }
            if not self.state.initialized:
                classifier_kwargs["classes"] = np.asarray(
                    [0, 1],
                    dtype=int,
                )
            self.state.target_model.partial_fit(
                transformed,
                np.asarray([target], dtype=int),
                **classifier_kwargs,
            )
            self.state.stop_model.partial_fit(
                transformed,
                np.asarray([stop], dtype=int),
                **dict(classifier_kwargs),
            )
            self.state.r_model.partial_fit(
                transformed,
                np.asarray([realized_r], dtype=float),
                sample_weight=np.asarray([weight], dtype=float),
            )
            self.state.initialized = True
            self.state.samples_seen += 1
            self.state.learned_trade_ids.add(trade_id)
            trade["adaptive_learned"] = True
            trade["adaptive_learned_at"] = pd.Timestamp.now(
                tz="UTC"
            ).isoformat()
            self.state.recent_outcomes.append(
                {
                    "trade_id": trade_id,
                    "closed_at": trade.get("closed_at"),
                    "direction": trade.get("direction"),
                    "outcome": outcome,
                    "realized_r": realized_r,
                    "realized_net_pnl_usd": _finite(
                        trade.get("realized_net_pnl_usd"),
                        0.0,
                    ),
                }
            )
            learned += 1

        self.state.recent_outcomes = self.state.recent_outcomes[-500:]
        self.state.updated_at = pd.Timestamp.now(tz="UTC").isoformat()
        self.save()
        return self.summary(trades, learned_now=learned)

    def enrich_trade_plan(
        self,
        record: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        self._feature_record = record
        self._last_extended_vector = None
        try:
            candidate = super().enrich_trade_plan(record, plan)
        finally:
            self._feature_record = None

        if self._last_extended_vector is not None:
            candidate["entry_feature_names"] = list(
                EXTENDED_TRADE_FEATURES
            )
            candidate["entry_feature_vector"] = (
                self._last_extended_vector.tolist()
            )
        candidate = apply_risk_scaled_economics(
            candidate,
            self.settings,
        )

        active = self._active_position
        if active is None:
            return candidate

        record["candidate_action"] = record.get("action")
        record["candidate_trade_plan"] = candidate
        record["candidate_blockers"] = list(record.get("blockers", []))
        record["managed_trade_id"] = active.get("trade_id")
        record["managed_position_direction"] = active.get("direction")
        record["active_position_contract"] = ACTIVE_POSITION_CONTRACT
        record["blockers"] = ["ACTIVE_TRADE_IN_PROGRESS"]
        return build_active_position_plan(active)

    def _predict(
        self,
        vector: np.ndarray,
        *,
        fallback_target: float,
        fallback_stop: float,
        fallback_r: float,
    ) -> tuple[float, float, float, float]:
        extended = append_context_features(
            vector,
            self._feature_record or {},
        )
        self._last_extended_vector = extended
        return super()._predict(
            extended,
            fallback_target=fallback_target,
            fallback_stop=fallback_stop,
            fallback_r=fallback_r,
        )

    def summary(
        self,
        trades: Iterable[dict[str, Any]],
        *,
        learned_now: int = 0,
    ) -> dict[str, Any]:
        summary = super().summary(trades, learned_now=learned_now)
        summary["schema_version"] = TRADE_STATE_SCHEMA_VERSION
        summary["feature_count"] = len(EXTENDED_TRADE_FEATURES)
        summary["runtime_contract"] = RUNTIME_CONTRACT
        return summary


def open_trade_with_context(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    trade = open_trade_from_record(record)
    if trade is None:
        return None

    context = record.get("event_candle_context")
    if isinstance(context, dict):
        trade["event_candle_context"] = context
        trade["candle_context_contract"] = str(
            record.get("candle_context_contract") or CONTEXT_CONTRACT
        )
        trade["candle_context_complete"] = bool(
            record.get(
                "candle_context_complete",
                context.get("complete", False),
            )
        )

    plan = record.get("trade_plan")
    plan = plan if isinstance(plan, dict) else {}
    _copy_plan_fields(trade, plan, POSITION_PLAN_FIELDS)
    trade.setdefault("entry_definition", "PAPER_MARKET_ORDER_AT_SIGNAL_RUN")
    trade.setdefault("policy_name", "LEGACY_AGGRESSIVE_PAPER")
    trade.setdefault("policy_version", 1)
    trade.setdefault("risk_contract_version", 1)
    trade.setdefault("entry_contract", "LEGACY_FIXED_RISK")
    trade["soft_risk_flags"] = list(
        trade.get("soft_risk_flags") or []
    )
    trade["qualification_passed"] = bool(
        trade.get("qualification_passed", False)
    )
    trade["direction_qualified"] = bool(
        trade.get("direction_qualified", False)
    )

    trade["selected_horizon"] = record.get(
        "trade_selected_horizon",
        record.get("selected_horizon"),
    )
    trade["breakout_source"] = record.get("breakout_source")
    trade["breakout_level"] = record.get("breakout_level")
    trade["invalidation_level"] = record.get(
        "breakout_invalidation_level"
    )
    trade["regime"] = record.get("regime")
    trade["event_score"] = record.get("trigger_score")

    execution_quote = record.get("execution_quote")
    if isinstance(execution_quote, dict):
        trade["execution_quote"] = execution_quote

    observed_at = plan.get("entry_quote_observed_at")
    if observed_at:
        opened_at = _utc(observed_at)
        holding_hours = int(trade.get("maximum_holding_hours", 72))
        trade["opened_at"] = opened_at.isoformat()
        trade["expires_at"] = (
            opened_at + pd.Timedelta(hours=holding_hours)
        ).isoformat()
    return install_execution_path_contract(trade)


def optional_price_prediction(
    builder: Callable[..., dict[str, Any]],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Keep secondary price-model failure outside the primary trade cycle."""
    try:
        return builder(*args, **kwargs)
    except Exception as exc:
        return {
            "status": "UNAVAILABLE",
            "source": "SECONDARY_PRICE_MODEL_UNAVAILABLE",
            "error": f"{type(exc).__name__}: {exc}",
            "batch_probability_up": 0.5,
            "online_probability_up": 0.5,
            "fused_probability_up": 0.5,
            "direction_blend_weight": 0.0,
            "batch_return": 0.0,
            "online_return": 0.0,
            "fused_return": 0.0,
            "return_blend_weight": 0.0,
            "metrics": {},
            "samples_seen": 0,
        }


def build_optional_secondary_forecast(
    record: dict[str, Any],
    model_metrics: dict[str, Any] | None,
    recent_candles: pd.DataFrame,
    history: list[dict[str, Any]],
    interval_probability: float = 0.80,
) -> dict[str, Any]:
    price_model = record.get("price_forecast_model")
    price_model = price_model if isinstance(price_model, dict) else {}
    if price_model.get("status") == "UNAVAILABLE":
        return _not_created_contract(
            str(
                price_model.get("error")
                or "Secondary price model unavailable"
            ),
            "PRICE_MODEL_UNAVAILABLE",
        )
    try:
        return build_strict_next_candle_forecast(
            record,
            model_metrics,
            recent_candles,
            history,
            interval_probability=interval_probability,
        )
    except Exception as exc:
        return _not_created_contract(
            f"{type(exc).__name__}: {exc}",
            "MISSED_OR_INVALID_TARGET_WINDOW",
        )


def attach_optional_forecast(
    result: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    if contract.get("status") == "NOT_CREATED":
        output = dict(result)
        output["secondary_forecast_status"] = "NOT_CREATED"
        output["secondary_forecast_error"] = contract.get("error")
        output["secondary_forecast_timing_status"] = contract.get(
            "timing_status"
        )
        output["prediction_result"] = "NOT_SCORED"
        output["direction_result"] = "NOT_SCORED"
        output["interval_result"] = "NOT_SCORED"
        output["resolved_at"] = None
        output["primary_objective"] = (
            "OPEN_TRADE_TO_TARGET_STOP_OR_TIME_EXIT"
        )
        return output

    output = attach_forecast_contract(result, contract)
    output["secondary_forecast_status"] = "CREATED"
    output["secondary_forecast_error"] = None
    output["secondary_forecast_timing_status"] = contract.get(
        "timing_status"
    )
    return output


def attach_forecast_contract(
    result: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Attach secondary diagnostics without replacing the primary position."""
    output = dict(result)
    output["trade_forecast_direction"] = output.get("forecast_direction")
    output["trade_selected_horizon"] = output.get("selected_horizon")
    output["trade_confidence"] = output.get("confidence")
    output["next_candle_forecast"] = contract
    output["forecast_contract_version"] = contract["contract_version"]
    output["next_candle_direction"] = contract["direction"]
    output["next_candle_confidence"] = contract[
        "direction_confidence"
    ]
    output["next_candle_expected_return"] = contract["median_return"]
    output["target_candle_time"] = contract["target_open_time"]
    output["target_candle_open_time"] = contract["target_open_time"]
    output["target_candle_close_time"] = contract["target_close_time"]
    output["predicted_close_median"] = contract["median_close"]
    output["predicted_close_low"] = contract["likely_close_low"]
    output["predicted_close_high"] = contract["likely_close_high"]
    output["prediction_result"] = "PENDING"
    output["direction_result"] = "PENDING"
    output["interval_result"] = "PENDING"
    output["resolved_at"] = None
    output["forecast_frozen"] = True
    output["primary_objective"] = (
        "OPEN_TRADE_TO_TARGET_STOP_OR_TIME_EXIT"
    )
    return output


def preserve_canonical_forecast(
    history: list[dict[str, Any]],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Keep frozen forecast outcomes while refreshing runtime metadata."""
    key = record.get("candle_time")
    if not key:
        return record
    existing = next(
        (
            item
            for item in reversed(history)
            if item.get("candle_time") == key
            and isinstance(item.get("next_candle_forecast"), dict)
            and item.get("run_status") == "OK"
        ),
        None,
    )
    if existing is None:
        return record

    merged = dict(existing)
    for field, value in record.items():
        if field not in FORECAST_IMMUTABLE_FIELDS:
            merged[field] = value
    return merged


def retain_directional_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        item
        for item in history
        if str(item.get("model_id") or "").startswith(MODEL_PREFIX)
    ]


def _copy_plan_fields(
    target: dict[str, Any],
    source: dict[str, Any],
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        if field in source:
            target[field] = source[field]


def _not_created_contract(
    error: str,
    timing_status: str,
) -> dict[str, Any]:
    return {
        "contract_version": 0,
        "status": "NOT_CREATED",
        "target": "NEXT_CLOSED_1H_CANDLE",
        "timing_status": timing_status,
        "retroactive_forecast": False,
        "error": error,
    }


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
