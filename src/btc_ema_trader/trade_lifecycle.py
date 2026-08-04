from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .costs import execution_cost_breakdown

TRADE_STATE_SCHEMA_VERSION = 1
TRADE_FEATURES = (
    "direction_code",
    "confidence",
    "tradeability",
    "event_score",
    "expected_net_edge_bps",
    "atr_pct",
    "adx",
    "rsi_centered",
    "volume_z_24",
    "regime_code",
    "boundary_break_probability",
    "boundary_unprofitable_probability",
    "front_bloom_hit",
    "backup_bloom_hit",
    "base_stop_percent",
    "base_reward_r",
)


def _classifier(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=2e-4,
        l1_ratio=0.05,
        learning_rate="optimal",
        average=True,
        random_state=seed,
    )


def _regressor(seed: int) -> SGDRegressor:
    return SGDRegressor(
        loss="huber",
        penalty="elasticnet",
        alpha=2e-4,
        l1_ratio=0.05,
        learning_rate="invscaling",
        eta0=0.01,
        power_t=0.25,
        average=True,
        random_state=seed,
    )


@dataclass
class TradeAdaptiveState:
    schema_version: int
    created_at: str
    updated_at: str
    scaler: StandardScaler = field(default_factory=StandardScaler)
    target_model: SGDClassifier = field(default_factory=lambda: _classifier(9101))
    stop_model: SGDClassifier = field(default_factory=lambda: _classifier(9102))
    r_model: SGDRegressor = field(default_factory=lambda: _regressor(9103))
    initialized: bool = False
    samples_seen: int = 0
    learned_trade_ids: set[str] = field(default_factory=set)
    source_model_ids: list[str] = field(default_factory=list)
    recent_outcomes: list[dict[str, Any]] = field(default_factory=list)


class AdaptiveTradeEngine:
    """Online target/stop learner and adaptive 5R trade-plan generator."""

    def __init__(
        self,
        settings: Settings,
        model_id: str,
        state_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.cfg = settings.section("trade_lifecycle")
        self.strategy_cfg = settings.section("strategy")
        self.enabled = bool(self.cfg.get("enabled", True))
        self.model_id = str(model_id)
        self.path = state_path or settings.path("trade_adaptive_state")
        self.state = self._load_or_create()
        if self.model_id and self.model_id not in self.state.source_model_ids:
            self.state.source_model_ids.append(self.model_id)
            self.state.source_model_ids = self.state.source_model_ids[-20:]

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

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        joblib.dump(self.state, temporary)
        temporary.replace(self.path)

    def synchronize(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Learn once from every newly resolved paper trade."""
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
                trade.get("entry_feature_vector", []), dtype=float
            )
            if vector.shape != (len(TRADE_FEATURES),) or not np.isfinite(vector).all():
                continue
            outcome = str(trade.get("outcome") or "")
            target = int(outcome == "TARGET")
            stop = int(outcome == "STOP")
            realized_r = _finite(trade.get("realized_r"), 0.0)
            weight = 1.0 + min(3.0, abs(realized_r))
            if stop:
                weight *= float(self.cfg.get("stop_learning_weight", 1.75))

            matrix = vector.reshape(1, -1)
            self.state.scaler.partial_fit(matrix)
            transformed = self.state.scaler.transform(matrix)
            kwargs: dict[str, Any] = {
                "sample_weight": np.asarray([weight], dtype=float)
            }
            if not self.state.initialized:
                kwargs["classes"] = np.asarray([0, 1], dtype=int)
            self.state.target_model.partial_fit(
                transformed,
                np.asarray([target], dtype=int),
                **kwargs,
            )
            stop_kwargs = dict(kwargs)
            self.state.stop_model.partial_fit(
                transformed,
                np.asarray([stop], dtype=int),
                **stop_kwargs,
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
                        trade.get("realized_net_pnl_usd"), 0.0
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
        """Replace symmetric price range with adaptive target/stop barriers."""
        output = dict(plan)
        entry = _finite(
            output.get("entry_reference", record.get("price")),
            0.0,
        )
        if entry <= 0:
            return output
        direction = str(record.get("action") or "WAIT").upper()
        if direction not in {"LONG", "SHORT"}:
            direction = (
                "LONG"
                if str(record.get("trade_forecast_direction") or record.get("forecast_direction"))
                == "UP"
                else "SHORT"
            )
        direction_code = 1.0 if direction == "LONG" else -1.0
        atr = _finite(output.get("entry_atr"), 0.0)
        atr_pct = _finite(output.get("atr_pct"), 0.0)
        if atr <= 0 and atr_pct > 0:
            atr = entry * atr_pct
        if atr <= 0:
            atr = entry * float(self.cfg.get("fallback_atr_percent", 0.008))
            atr_pct = atr / entry

        structural_stop = _finite(output.get("stop_price"), math.nan)
        structural_stop_pct = (
            abs(entry - structural_stop) / entry
            if math.isfinite(structural_stop)
            else 0.0
        )
        base_stop_pct = max(
            structural_stop_pct,
            float(self.cfg.get("base_stop_atr_multiplier", 0.75))
            * atr
            / entry,
        )
        base_stop_pct = float(
            np.clip(
                base_stop_pct,
                float(self.cfg.get("minimum_stop_percent", 0.0025)),
                float(self.cfg.get("maximum_stop_percent", 0.025)),
            )
        )
        base_reward_r = float(self.cfg.get("base_reward_r", 5.0))
        vector = trade_feature_vector(
            record,
            output,
            base_stop_percent=base_stop_pct,
            base_reward_r=base_reward_r,
            direction_code=direction_code,
        )
        fallback_target, fallback_stop = _fallback_probabilities(record, output)
        p_target, p_stop, predicted_r, online_weight = self._predict(
            vector,
            fallback_target=fallback_target,
            fallback_stop=fallback_stop,
            fallback_r=_fallback_expected_r(record, base_stop_pct),
        )
        p_expiry = max(0.0, 1.0 - p_target - p_stop)

        minimum_samples = int(self.cfg.get("minimum_online_samples", 20))
        if self.state.samples_seen < minimum_samples:
            reward_r = base_reward_r
            stop_scale = 1.0
            holding_scale = 1.0
        else:
            reward_scale = float(
                np.clip(
                    1.0
                    + float(self.cfg.get("reward_probability_gain", 0.80))
                    * (p_target - p_stop)
                    + float(self.cfg.get("reward_r_gain", 0.08))
                    * predicted_r,
                    0.60,
                    1.60,
                )
            )
            reward_r = base_reward_r * reward_scale
            stop_scale = float(
                np.clip(
                    0.85 + 0.55 * p_stop - 0.20 * p_target,
                    0.65,
                    1.35,
                )
            )
            holding_scale = float(
                np.clip(0.75 + 1.10 * p_target - 0.45 * p_stop, 0.50, 2.0)
            )
        reward_r = float(
            np.clip(
                reward_r,
                float(self.cfg.get("minimum_reward_r", 3.0)),
                float(self.cfg.get("maximum_reward_r", 8.0)),
            )
        )
        stop_pct = float(
            np.clip(
                base_stop_pct * stop_scale,
                float(self.cfg.get("minimum_stop_percent", 0.0025)),
                float(self.cfg.get("maximum_stop_percent", 0.025)),
            )
        )
        stop_distance = entry * stop_pct
        target_distance = stop_distance * reward_r
        stop_price = (
            entry - stop_distance if direction == "LONG" else entry + stop_distance
        )
        target_price = (
            entry + target_distance if direction == "LONG" else entry - target_distance
        )
        base_holding = int(self.cfg.get("base_maximum_holding_hours", 72))
        holding_hours = int(
            np.clip(
                round(base_holding * holding_scale),
                int(self.cfg.get("minimum_holding_hours", 12)),
                int(self.cfg.get("maximum_holding_hours", 168)),
            )
        )

        economics = _margin_economics(
            settings=self.settings,
            entry=entry,
            stop_pct=stop_pct,
            target_pct=target_distance / entry,
            p_target=p_target,
            p_stop=p_stop,
            p_expiry=p_expiry,
            predicted_r=predicted_r,
        )
        output.update(
            {
                "status": (
                    "ACTIONABLE"
                    if record.get("action") in {"LONG", "SHORT"}
                    else output.get("status", "BLOCKED")
                ),
                "contract_type": "ADAPTIVE_TARGET_STOP_LIFECYCLE",
                "entry_reference": entry,
                "entry_definition": "PAPER_MARKET_ORDER_AT_SIGNAL_RUN",
                "stop_price": float(stop_price),
                "initial_stop_price": float(stop_price),
                "target_price": float(target_price),
                "stop_percent": stop_pct,
                "target_percent": target_distance / entry,
                "risk_reward": reward_r,
                "base_reward_r": base_reward_r,
                "adaptive_reward_r": reward_r,
                "maximum_holding_hours": holding_hours,
                "expiry_policy": "TARGET_OR_STOP_OR_TIME_EXIT",
                "same_bar_resolution": str(
                    self.cfg.get("same_bar_policy", "NEAREST_TO_OPEN")
                ),
                "breakeven_trigger_r": float(
                    self.cfg.get("breakeven_trigger_r", 2.0)
                ),
                "trailing_trigger_r": float(
                    self.cfg.get("trailing_trigger_r", 3.0)
                ),
                "trailing_atr_multiplier": float(
                    self.cfg.get("trailing_atr_multiplier", 1.0)
                ),
                "adaptive_target_probability": p_target,
                "adaptive_stop_probability": p_stop,
                "adaptive_expiry_probability": p_expiry,
                "adaptive_predicted_r": predicted_r,
                "adaptive_online_weight": online_weight,
                "adaptive_samples_seen": self.state.samples_seen,
                "entry_feature_names": list(TRADE_FEATURES),
                "entry_feature_vector": vector.tolist(),
                "entry_atr": atr,
                "atr_pct": atr_pct,
                **economics,
            }
        )
        return output

    def _predict(
        self,
        vector: np.ndarray,
        *,
        fallback_target: float,
        fallback_stop: float,
        fallback_r: float,
    ) -> tuple[float, float, float, float]:
        if not self.enabled or not self.state.initialized:
            return fallback_target, fallback_stop, fallback_r, 0.0
        transformed = self.state.scaler.transform(vector.reshape(1, -1))
        online_target = float(
            np.clip(self.state.target_model.predict_proba(transformed)[0, 1], 0.01, 0.98)
        )
        online_stop = float(
            np.clip(self.state.stop_model.predict_proba(transformed)[0, 1], 0.01, 0.98)
        )
        online_r = float(np.clip(self.state.r_model.predict(transformed)[0], -3.0, 8.0))
        min_samples = max(1, int(self.cfg.get("minimum_online_samples", 20)))
        max_weight = float(self.cfg.get("maximum_online_weight", 0.65))
        weight = min(max_weight, max_weight * self.state.samples_seen / min_samples)
        p_target = (1.0 - weight) * fallback_target + weight * online_target
        p_stop = (1.0 - weight) * fallback_stop + weight * online_stop
        total = p_target + p_stop
        if total > 0.95:
            scale = 0.95 / total
            p_target *= scale
            p_stop *= scale
        predicted_r = (1.0 - weight) * fallback_r + weight * online_r
        return float(p_target), float(p_stop), float(predicted_r), float(weight)

    def summary(
        self,
        trades: Iterable[dict[str, Any]],
        *,
        learned_now: int = 0,
    ) -> dict[str, Any]:
        items = list(trades)
        resolved = [item for item in items if item.get("status") == "CLOSED"]
        active = [item for item in items if item.get("status") == "OPEN"]
        targets = sum(item.get("outcome") == "TARGET" for item in resolved)
        stops = sum(item.get("outcome") == "STOP" for item in resolved)
        time_exits = len(resolved) - targets - stops
        pnl = sum(_finite(item.get("realized_net_pnl_usd"), 0.0) for item in resolved)
        r_values = [_finite(item.get("realized_r"), 0.0) for item in resolved]
        return {
            "status": "ACTIVE" if self.enabled else "DISABLED",
            "schema_version": TRADE_STATE_SCHEMA_VERSION,
            "samples_seen": self.state.samples_seen,
            "learned_now": learned_now,
            "active_trades": len(active),
            "resolved_trades": len(resolved),
            "target_hits": targets,
            "stop_hits": stops,
            "time_exits": time_exits,
            "target_hit_rate": targets / len(resolved) if resolved else None,
            "net_pnl_usd": pnl,
            "average_r": float(np.mean(r_values)) if r_values else None,
            "updated_at": self.state.updated_at,
            "source_model_ids": self.state.source_model_ids[-5:],
        }


def trade_feature_vector(
    record: dict[str, Any],
    plan: dict[str, Any],
    *,
    base_stop_percent: float,
    base_reward_r: float,
    direction_code: float,
) -> np.ndarray:
    boundary = plan.get("boundary_memory")
    boundary = boundary if isinstance(boundary, dict) else {}
    horizons = boundary.get("horizons")
    horizons = horizons if isinstance(horizons, dict) else {}
    selected = boundary.get("selected_horizon")
    item = horizons.get(str(selected), {}) if selected is not None else {}
    item = item if isinstance(item, dict) else {}
    values = (
        direction_code,
        _finite(record.get("trade_confidence", record.get("confidence")), 0.5),
        _finite(record.get("tradeability_probability"), 0.5),
        _finite(record.get("trigger_score", plan.get("event_score")), 0.0),
        _finite(record.get("expected_net_edge_bps"), 0.0) / 100.0,
        _finite(plan.get("atr_pct"), 0.0),
        _finite(plan.get("adx"), 0.0) / 100.0,
        _finite(plan.get("rsi_centered"), 0.0),
        _finite(plan.get("volume_z_24"), 0.0) / 5.0,
        _finite(plan.get("regime_code"), 0.0) / 5.0,
        _finite(item.get("p_break", boundary.get("p_break")), 0.5),
        _finite(
            item.get("p_unprofitable", boundary.get("p_unprofitable")),
            0.5,
        ),
        float(bool(item.get("front_memory_hit", False))),
        float(bool(item.get("backup_memory_hit", False))),
        base_stop_percent,
        base_reward_r / 10.0,
    )
    return np.asarray(values, dtype=float)


def open_trade_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    action = str(record.get("action") or "").upper()
    plan = record.get("trade_plan")
    if action not in {"LONG", "SHORT"} or not isinstance(plan, dict):
        return None
    if plan.get("status") != "ACTIONABLE":
        return None
    opened_at = _utc(
        record.get("run_finished_at")
        or record.get("created_at")
        or record.get("candle_time")
        or pd.Timestamp.now(tz="UTC")
    )
    signal_time = _utc(record.get("candle_time") or opened_at)
    entry = _finite(plan.get("entry_reference"), 0.0)
    stop = _finite(plan.get("stop_price"), 0.0)
    target = _finite(plan.get("target_price"), 0.0)
    if entry <= 0 or stop <= 0 or target <= 0:
        return None
    raw_id = "|".join(
        [
            str(record.get("event_id") or "NONE"),
            signal_time.isoformat(),
            action,
            f"{entry:.8f}",
        ]
    )
    trade_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20]
    maximum_holding = int(plan.get("maximum_holding_hours", 72))
    initial_risk_price = abs(entry - stop)
    return {
        "schema_version": 1,
        "trade_id": trade_id,
        "status": "OPEN",
        "direction": action,
        "model_id": record.get("model_id"),
        "event_id": record.get("event_id"),
        "event_type": record.get("event_type"),
        "signal_candle_time": signal_time.isoformat(),
        "opened_at": opened_at.isoformat(),
        "entry_price": entry,
        "target_price": target,
        "initial_stop_price": stop,
        "current_stop_price": stop,
        "initial_risk_price": initial_risk_price,
        "initial_risk_percent": initial_risk_price / entry,
        "risk_reward": _finite(plan.get("risk_reward"), 5.0),
        "maximum_holding_hours": maximum_holding,
        "expires_at": (opened_at + pd.Timedelta(hours=maximum_holding)).isoformat(),
        "breakeven_trigger_r": _finite(plan.get("breakeven_trigger_r"), 2.0),
        "trailing_trigger_r": _finite(plan.get("trailing_trigger_r"), 3.0),
        "trailing_atr_multiplier": _finite(
            plan.get("trailing_atr_multiplier"), 1.0
        ),
        "entry_atr": _finite(plan.get("entry_atr"), initial_risk_price),
        "quantity_btc": _finite(plan.get("quantity_btc"), 0.0),
        "notional_usd": _finite(plan.get("notional_usd"), 0.0),
        "margin_required_usd": _finite(plan.get("margin_required_usd"), 0.0),
        "risk_budget_usd": _finite(plan.get("risk_budget_usd"), 0.0),
        "stress_execution_cost_bps": _finite(
            plan.get("stress_execution_cost_bps"), 0.0
        ),
        "target_net_profit_usd": _finite(plan.get("target_net_profit_usd"), 0.0),
        "stop_net_loss_usd": _finite(plan.get("stop_net_loss_usd"), 0.0),
        "expected_value_usd": _finite(plan.get("expected_value_usd"), 0.0),
        "target_margin_roi": _finite(plan.get("target_margin_roi"), 0.0),
        "adaptive_target_probability": _finite(
            plan.get("adaptive_target_probability"), 0.5
        ),
        "adaptive_stop_probability": _finite(
            plan.get("adaptive_stop_probability"), 0.4
        ),
        "entry_feature_names": list(plan.get("entry_feature_names", TRADE_FEATURES)),
        "entry_feature_vector": list(plan.get("entry_feature_vector", [])),
        "max_favorable_r": 0.0,
        "max_adverse_r": 0.0,
        "breakeven_armed": False,
        "trailing_armed": False,
        "adaptive_learned": False,
    }


def resolve_open_trades(
    trades: list[dict[str, Any]],
    candles: pd.DataFrame,
    settings: Settings,
) -> int:
    if candles.empty:
        return 0
    frame = candles.copy().sort_values("open_time").reset_index(drop=True)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    resolved = 0
    for trade in trades:
        if trade.get("status") != "OPEN":
            continue
        signal_time = _utc(trade.get("signal_candle_time"))
        expiry = _utc(trade.get("expires_at"))
        relevant = frame.loc[frame["open_time"] > signal_time]
        if relevant.empty:
            continue
        for _, candle in relevant.iterrows():
            candle_time = _utc(candle["open_time"])
            event = _evaluate_candle(trade, candle, settings)
            if event is not None:
                _close_trade(
                    trade,
                    exit_price=event["exit_price"],
                    outcome=event["outcome"],
                    closed_at=(candle_time + pd.Timedelta(hours=1)),
                )
                resolved += 1
                break
            _update_dynamic_stop(trade, candle)
            if candle_time + pd.Timedelta(hours=1) >= expiry:
                _close_trade(
                    trade,
                    exit_price=float(candle["close"]),
                    outcome="TIME_EXIT",
                    closed_at=(candle_time + pd.Timedelta(hours=1)),
                )
                resolved += 1
                break
    return resolved


def _evaluate_candle(
    trade: dict[str, Any],
    candle: pd.Series,
    settings: Settings,
) -> dict[str, Any] | None:
    direction = str(trade["direction"])
    entry = float(trade["entry_price"])
    target = float(trade["target_price"])
    stop = float(trade["current_stop_price"])
    high = float(candle["high"])
    low = float(candle["low"])
    open_price = float(candle["open"])
    risk = max(float(trade["initial_risk_price"]), 1e-9)
    if direction == "LONG":
        favorable = (high - entry) / risk
        adverse = (entry - low) / risk
        target_hit = high >= target
        stop_hit = low <= stop
    else:
        favorable = (entry - low) / risk
        adverse = (high - entry) / risk
        target_hit = low <= target
        stop_hit = high >= stop
    trade["max_favorable_r"] = max(
        float(trade.get("max_favorable_r", 0.0)), favorable
    )
    trade["max_adverse_r"] = max(
        float(trade.get("max_adverse_r", 0.0)), adverse
    )
    if not target_hit and not stop_hit:
        return None
    if target_hit and stop_hit:
        policy = str(
            settings.section("trade_lifecycle").get(
                "same_bar_policy", "NEAREST_TO_OPEN"
            )
        ).upper()
        if policy == "TARGET_FIRST":
            return {"outcome": "TARGET", "exit_price": target}
        if policy == "STOP_FIRST":
            return {"outcome": "STOP", "exit_price": stop}
        target_distance = abs(open_price - target)
        stop_distance = abs(open_price - stop)
        if target_distance < stop_distance:
            return {"outcome": "TARGET", "exit_price": target}
        return {"outcome": "STOP", "exit_price": stop}
    if target_hit:
        return {"outcome": "TARGET", "exit_price": target}
    return {"outcome": "STOP", "exit_price": stop}


def _update_dynamic_stop(trade: dict[str, Any], candle: pd.Series) -> None:
    entry = float(trade["entry_price"])
    risk = max(float(trade["initial_risk_price"]), 1e-9)
    direction = str(trade["direction"])
    current = float(trade["current_stop_price"])
    mfe = float(trade.get("max_favorable_r", 0.0))
    cost_fraction = float(trade.get("stress_execution_cost_bps", 0.0)) / 10_000.0
    if mfe >= float(trade.get("breakeven_trigger_r", 2.0)):
        breakeven = entry * (
            1.0 + cost_fraction if direction == "LONG" else 1.0 - cost_fraction
        )
        current = max(current, breakeven) if direction == "LONG" else min(current, breakeven)
        trade["breakeven_armed"] = True
    if mfe >= float(trade.get("trailing_trigger_r", 3.0)):
        trail = max(
            float(trade.get("entry_atr", risk))
            * float(trade.get("trailing_atr_multiplier", 1.0)),
            risk * 0.25,
        )
        if direction == "LONG":
            candidate = float(candle["high"]) - trail
            current = max(current, candidate)
        else:
            candidate = float(candle["low"]) + trail
            current = min(current, candidate)
        trade["trailing_armed"] = True
    trade["current_stop_price"] = float(current)


def _close_trade(
    trade: dict[str, Any],
    *,
    exit_price: float,
    outcome: str,
    closed_at: pd.Timestamp,
) -> None:
    entry = float(trade["entry_price"])
    direction = str(trade["direction"])
    gross_return = exit_price / entry - 1.0
    aligned_return = gross_return if direction == "LONG" else -gross_return
    cost_fraction = float(trade.get("stress_execution_cost_bps", 0.0)) / 10_000.0
    net_return = aligned_return - cost_fraction
    notional = float(trade.get("notional_usd", 0.0))
    net_pnl = notional * net_return
    risk_budget = max(float(trade.get("risk_budget_usd", 0.0)), 1e-9)
    realized_r = net_pnl / risk_budget
    final_outcome = outcome
    if outcome == "TIME_EXIT":
        final_outcome = "TIME_EXIT_WIN" if net_pnl > 0 else "TIME_EXIT_LOSS"
    trade.update(
        {
            "status": "CLOSED",
            "outcome": final_outcome,
            "closed_at": _utc(closed_at).isoformat(),
            "exit_price": float(exit_price),
            "gross_aligned_return": float(aligned_return),
            "realized_net_return": float(net_return),
            "realized_net_pnl_usd": float(net_pnl),
            "realized_r": float(realized_r),
        }
    )


def active_trade(trades: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for trade in reversed(list(trades)):
        if trade.get("status") == "OPEN":
            return trade
    return None


def _fallback_probabilities(
    record: dict[str, Any],
    plan: dict[str, Any],
) -> tuple[float, float]:
    confidence = _finite(record.get("trade_confidence", record.get("confidence")), 0.5)
    tradeability = _finite(record.get("tradeability_probability"), 0.5)
    boundary = plan.get("boundary_memory")
    boundary = boundary if isinstance(boundary, dict) else {}
    horizons = boundary.get("horizons")
    horizons = horizons if isinstance(horizons, dict) else {}
    selected = boundary.get("selected_horizon")
    item = horizons.get(str(selected), {}) if selected is not None else {}
    item = item if isinstance(item, dict) else {}
    p_break = _finite(item.get("p_break", boundary.get("p_break")), 0.5)
    p_bad = _finite(
        item.get("p_unprofitable", boundary.get("p_unprofitable")), 0.5
    )
    bloom_penalty = (
        0.10 * float(bool(item.get("front_memory_hit", False)))
        + 0.15 * float(bool(item.get("backup_memory_hit", False)))
    )
    p_target = float(
        np.clip(
            0.20
            + 0.40 * confidence
            + 0.25 * tradeability
            + 0.25 * p_break
            - 0.30 * p_bad
            - bloom_penalty,
            0.05,
            0.90,
        )
    )
    p_stop = float(
        np.clip(
            0.85 - p_target + 0.15 * p_bad + bloom_penalty,
            0.05,
            0.90,
        )
    )
    if p_target + p_stop > 0.95:
        scale = 0.95 / (p_target + p_stop)
        p_target *= scale
        p_stop *= scale
    return p_target, p_stop


def _fallback_expected_r(record: dict[str, Any], stop_pct: float) -> float:
    move = abs(_finite(record.get("expected_return"), 0.0))
    return float(np.clip(move / max(stop_pct, 1e-6), -1.0, 5.0))


def _margin_economics(
    *,
    settings: Settings,
    entry: float,
    stop_pct: float,
    target_pct: float,
    p_target: float,
    p_stop: float,
    p_expiry: float,
    predicted_r: float,
) -> dict[str, Any]:
    strategy = settings.section("strategy")
    account = float(strategy.get("account_equity_usd", 1000.0))
    risk_fraction = float(strategy.get("risk_per_trade_fraction", 0.01))
    risk_budget = account * risk_fraction
    maximum_leverage = float(strategy.get("maximum_leverage", 5.0))
    costs = execution_cost_breakdown(strategy)
    stress_bps = float(costs["stress_cost_bps"])
    cost_fraction = stress_bps / 10_000.0
    unit_risk = entry * (stop_pct + cost_fraction)
    quantity = risk_budget / max(unit_risk, 1e-9)
    notional = min(quantity * entry, account * maximum_leverage)
    quantity = notional / entry
    leverage = float(
        np.clip(notional / max(account, 1e-9), 1.0, maximum_leverage)
    )
    margin = notional / leverage
    execution_cost = notional * cost_fraction
    target_gross = notional * target_pct
    stop_gross = notional * stop_pct
    target_net = target_gross - execution_cost
    stop_net = -(stop_gross + execution_cost)
    expiry_net = notional * predicted_r * stop_pct - execution_cost
    expected_value = (
        p_target * target_net + p_stop * stop_net + p_expiry * expiry_net
    )
    return {
        "risk_budget_usd": float(risk_budget),
        "quantity_btc": float(quantity),
        "notional_usd": float(notional),
        "suggested_leverage": leverage,
        "margin_required_usd": float(margin),
        "round_trip_stress_cost_usd": float(execution_cost),
        "target_gross_profit_usd": float(target_gross),
        "target_net_profit_usd": float(target_net),
        "stop_gross_loss_usd": float(-stop_gross),
        "stop_net_loss_usd": float(stop_net),
        "profit_margin_usd": float(target_net),
        "target_margin_roi": float(target_net / max(margin, 1e-9)),
        "stop_margin_roi": float(stop_net / max(margin, 1e-9)),
        "expected_value_usd": float(expected_value),
        "stress_execution_cost_bps": stress_bps,
        "paper_only": bool(strategy.get("paper_only", True)),
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
