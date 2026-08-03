from __future__ import annotations

import json
import logging
import threading
from typing import Any

import pandas as pd

from .adaptive_runtime import AdaptiveEngine
from .config import Settings
from .contract_training import train_from_database
from .costs import execution_cost_breakdown
from .features import build_feature_set
from .market import MarketDataClient
from .model import latest_bundle
from .news import collect_and_store
from .storage import Database
from .strategy import make_decision

LOGGER = logging.getLogger(__name__)
_THREAD: threading.Thread | None = None
_STOP = threading.Event()


class RuntimeEngine:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self.market = MarketDataClient(settings)
        self.state_path = settings.path("runtime_state")
        self.state = self._load_or_create_state()

    def _load_or_create_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                state = json.loads(
                    self.state_path.read_text(encoding="utf-8")
                )
                state.setdefault("schema_version", 4)
                return state
            except Exception:
                pass
        now = pd.Timestamp.now(tz="UTC")
        state = {
            "schema_version": 4,
            "session_start": now.isoformat(),
            "first_eligible_close": next_hour_boundary(now).isoformat(),
            "last_processed_open_time": None,
            "last_cycle": None,
            "status": "WAITING_FOR_NEXT_CLOSED_CANDLE",
            "last_error": None,
        }
        self._save_state(state)
        return state

    def reset_session_clock(self) -> dict[str, Any]:
        now = pd.Timestamp.now(tz="UTC")
        self.state.update(
            {
                "schema_version": 4,
                "session_start": now.isoformat(),
                "first_eligible_close": next_hour_boundary(now).isoformat(),
                "last_processed_open_time": None,
                "status": "WAITING_FOR_NEXT_CLOSED_CANDLE",
                "last_error": None,
            }
        )
        self._save_state(self.state)
        return self.state

    def run_once(self, force: bool = False) -> dict[str, Any]:
        try:
            bundle = latest_bundle(self.settings)
            provider = bundle.provider
            recent = self.market.refresh_recent(provider, days=10)
            self.database.upsert_candles(recent)
            if self.settings.section("live").get(
                "collect_recent_news_each_cycle",
                True,
            ):
                try:
                    collect_and_store(
                        self.settings,
                        self.database,
                        historical=False,
                    )
                except Exception as exc:
                    self.database.log_event(
                        "WARNING",
                        "NEWS_REFRESH_FAILED",
                        str(exc),
                    )

            candles = self.database.load_candles(
                provider=provider,
                symbol=bundle.symbol,
            )
            news = self.database.load_news(
                start=candles["open_time"].min(),
                end=pd.Timestamp.now(tz="UTC"),
            )
            feature_set = build_feature_set(
                candles,
                news,
                self.settings,
                include_labels=True,
            )
            usable = feature_set.frame.dropna(
                subset=["kama", "donchian_mid", "atr", "adx"]
            ).copy()
            if usable.empty:
                raise RuntimeError("No feature row is ready")
            latest_row = usable.iloc[-1]
            candle_open = _utc(latest_row["open_time"])
            candle_close = candle_open + pd.Timedelta(hours=1)
            first_eligible = _utc(self.state["first_eligible_close"])
            last_processed = self.state.get("last_processed_open_time")
            if not force and candle_close < first_eligible:
                return self._waiting_result(candle_close, first_eligible)
            if not force and last_processed and candle_open <= _utc(
                last_processed
            ):
                return {
                    "status": "NO_NEW_CLOSED_CANDLE",
                    "latest_open_time": candle_open.isoformat(),
                }

            model_age_days = (
                pd.Timestamp.now(tz="UTC") - _utc(bundle.created_at)
            ).total_seconds() / 86400
            if model_age_days > float(
                self.settings.section("model").get(
                    "auto_retrain_days",
                    7,
                )
            ):
                if self.settings.section("live").get(
                    "auto_retrain",
                    True,
                ):
                    train_from_database(
                        self.settings,
                        self.database,
                        provider=provider,
                    )
                    bundle = latest_bundle(self.settings)
                    model_age_days = 0.0

            base_prediction = bundle.predict_frame(usable.tail(1))
            adaptive_engine = AdaptiveEngine(self.settings, bundle)
            adaptive_engine.synchronize(feature_set.frame)
            prediction, adaptive_summary = adaptive_engine.apply(
                latest_row,
                base_prediction,
            )

            quote_ok = True
            quote_provider: str | None = None
            try:
                quote = self.market.live_quote(provider_hint=provider)
                quote_provider = quote.provider
                quote_age = (
                    pd.Timestamp.now(tz="UTC") - quote.timestamp
                ).total_seconds()
                quote_ok = quote_age <= float(
                    self.settings.section("market").get(
                        "quote_stale_seconds",
                        90,
                    )
                )
            except Exception:
                quote_ok = False

            gaps = (
                pd.DatetimeIndex(candles["open_time"])
                .to_series()
                .diff()
                .dropna()
                .dt.total_seconds()
                .div(3600)
            )
            candles_ok = bool(
                gaps.empty
                or gaps.max()
                <= float(
                    self.settings.section("market").get(
                        "maximum_gap_hours",
                        2,
                    )
                )
            )
            news_age = float(latest_row.get("news_age_hours", 999))
            data_health = {
                "candles_ok": candles_ok,
                "quote_ok": quote_ok,
                "provider_mismatch": provider != bundle.provider
                or (
                    quote_provider is not None
                    and quote_provider != bundle.provider
                ),
                "model_stale": model_age_days
                > float(
                    self.settings.section("model").get(
                        "auto_retrain_days",
                        7,
                    )
                ),
                "news_stale": news_age
                > float(
                    self.settings.section("news").get(
                        "stale_hours",
                        6,
                    )
                ),
            }
            recent_count, hours_since_last = (
                self.database.signal_stats_today()
            )
            event_id = latest_row.get("event_id")
            if pd.isna(event_id):
                event_id = None
            event_already_traded = self.database.has_action_for_event(
                None if event_id is None else str(event_id)
            )
            decision = make_decision(
                latest_row,
                prediction,
                bundle,
                self.settings,
                data_health=data_health,
                recent_signal_count=recent_count,
                hours_since_last_signal=hours_since_last,
                event_already_traded=event_already_traded,
            )
            costs = execution_cost_breakdown(
                self.settings.section("strategy")
            )
            payload = {
                "candle_time": candle_open,
                "created_at": pd.Timestamp.now(tz="UTC"),
                "provider": provider,
                "price": float(latest_row["close"]),
                "trend_kama": float(latest_row["kama"]),
                "donchian_mid": float(latest_row["donchian_mid"]),
                "event_id": event_id,
                "event_type": str(
                    latest_row.get("event_type", "NONE")
                ),
                "event_direction": int(
                    latest_row.get("event_direction", 0)
                ),
                "bars_since_event": (
                    None
                    if pd.isna(latest_row.get("bars_since_event"))
                    else float(latest_row.get("bars_since_event"))
                ),
                "regime": str(latest_row.get("regime", "UNKNOWN")),
                "trigger_score": float(
                    latest_row.get("event_score", 0.0)
                ),
                "forecast_direction": decision.forecast_direction,
                "action": decision.action,
                "confidence": decision.confidence,
                "tradeability_probability": (
                    decision.tradeability_probability
                ),
                "expected_return": decision.expected_return,
                "expected_net_edge_bps": decision.expected_net_edge_bps,
                "selected_horizon": decision.selected_horizon,
                "probabilities": decision.probabilities,
                "tradeability": decision.tradeability,
                "returns": decision.returns,
                "trade_plan": decision.trade_plan,
                "blockers": decision.blockers,
                "model_id": bundle.model_id,
                "qualification_passed": bool(
                    bundle.qualification.get("passed", False)
                ),
                "actual_cost_bps": costs["base_cost_bps"],
                "adaptive": adaptive_summary,
                "base_model": {
                    "direction": base_prediction["direction"],
                    "selected_horizon": base_prediction[
                        "selected_horizon"
                    ],
                    "probabilities": base_prediction["probabilities"],
                    "continuation": base_prediction["continuation"],
                    "tradeability": base_prediction["tradeability"],
                    "event_returns": base_prediction["event_returns"],
                },
            }
            self.database.save_signal(payload)
            self.database.resolve_due_signals(candles)
            self.state.update(
                {
                    "last_processed_open_time": candle_open.isoformat(),
                    "last_cycle": pd.Timestamp.now(
                        tz="UTC"
                    ).isoformat(),
                    "status": decision.action,
                    "last_error": None,
                    "last_event_id": event_id,
                    "last_event_type": str(
                        latest_row.get("event_type", "NONE")
                    ),
                    "adaptive_status": adaptive_summary.get("status"),
                }
            )
            self._save_state(self.state)
            self.database.log_event(
                "INFO",
                "SIGNAL_CREATED",
                f"{decision.forecast_direction}/{decision.action}",
                payload,
            )
            return payload | {"data_health": data_health}
        except Exception as exc:
            LOGGER.exception("Live cycle failed")
            self.state.update(
                {
                    "last_cycle": pd.Timestamp.now(
                        tz="UTC"
                    ).isoformat(),
                    "status": "FAIL_SAFE",
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
            )
            self._save_state(self.state)
            self.database.log_event(
                "ERROR",
                "LIVE_CYCLE_FAILED",
                str(exc),
            )
            return {
                "status": "FAIL_SAFE",
                "error": self.state["last_error"],
            }

    def run_forever(self) -> None:
        poll = max(
            5,
            int(self.settings.section("live").get("poll_seconds", 15)),
        )
        while not _STOP.is_set():
            self.run_once(force=False)
            _STOP.wait(poll)

    def status(self) -> dict[str, Any]:
        state = dict(self.state)
        now = pd.Timestamp.now(tz="UTC")
        first = _utc(state["first_eligible_close"])
        state["seconds_to_first_eligible"] = max(
            0,
            int((first - now).total_seconds()),
        )
        return state

    def _waiting_result(
        self,
        latest_close: pd.Timestamp,
        first_eligible: pd.Timestamp,
    ) -> dict[str, Any]:
        self.state.update(
            {
                "last_cycle": pd.Timestamp.now(
                    tz="UTC"
                ).isoformat(),
                "status": "WAITING_FOR_NEXT_CLOSED_CANDLE",
            }
        )
        self._save_state(self.state)
        return {
            "status": "WAITING_FOR_NEXT_CLOSED_CANDLE",
            "latest_closed_candle": latest_close.isoformat(),
            "first_eligible_close": first_eligible.isoformat(),
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)


def next_hour_boundary(timestamp: pd.Timestamp) -> pd.Timestamp:
    value = _utc(timestamp)
    return value.floor("h") + pd.Timedelta(hours=1)


def start_background_engine(
    settings: Settings,
    database: Database,
) -> RuntimeEngine:
    global _THREAD
    engine = RuntimeEngine(settings, database)
    if _THREAD is None or not _THREAD.is_alive():
        engine.reset_session_clock()
        _STOP.clear()
        _THREAD = threading.Thread(
            target=engine.run_forever,
            name="btc-regime-live-engine",
            daemon=True,
        )
        _THREAD.start()
    return engine


def stop_background_engine() -> None:
    _STOP.set()


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
