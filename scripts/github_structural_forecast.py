from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import btc_ema_trader.trade_lifecycle as trade_lifecycle_module
import github_hourly_forecast
from btc_ema_trader.active_position_contract import (
    ACTIVE_POSITION_CONTRACT,
    build_active_position_plan,
)
from btc_ema_trader.candle_context import (
    CONTEXT_CONTRACT,
    extract_candle_context,
)
from btc_ema_trader.context_trade_features import (
    install_context_trade_features,
)
from btc_ema_trader.execution_entry import apply_execution_quote
from btc_ema_trader.runtime_history import (
    fetch_latest_contiguous_and_store,
)
from btc_ema_trader.strict_forecast_contract import (
    build_strict_next_candle_forecast,
)

MODEL_PREFIX = "directional-breakout-hourly-"
_BASE_RUNTIME_ENGINE = github_hourly_forecast.RuntimeEngine
_BASE_OPEN_TRADE = github_hourly_forecast.open_trade_from_record
_BASE_ADAPTIVE_TRADE_ENGINE = trade_lifecycle_module.AdaptiveTradeEngine


class ContextRuntimeEngine:
    """Install causal candle context and a fresh paper-entry quote."""

    def __init__(self, settings, database) -> None:
        self.settings = settings
        self.database = database
        self.delegate = _BASE_RUNTIME_ENGINE(settings, database)

    def run_once(self, force: bool = False) -> dict[str, Any]:
        result = self.delegate.run_once(force=force)
        if not isinstance(result, dict) or not result.get("candle_time"):
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
        result["candle_context_complete"] = bool(context.get("complete", False))

        observed_at = pd.Timestamp.now(tz="UTC")
        try:
            quote = self.delegate.market.live_quote(provider_hint=provider)
            result = apply_execution_quote(
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
                    plan = dict(plan)
                    plan["status"] = "BLOCKED"
                    plan["entry_reference_kind"] = (
                        "EXECUTION_QUOTE_UNAVAILABLE"
                    )
                    result["trade_plan"] = plan
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class CanonicalAdaptiveTradeEngine(_BASE_ADAPTIVE_TRADE_ENGINE):
    """Keep the active position contract separate from a new candidate plan."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_position: dict[str, Any] | None = None

    def synchronize(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        summary = super().synchronize(trades)
        self._active_position = trade_lifecycle_module.active_trade(trades)
        return summary

    def enrich_trade_plan(
        self,
        record: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        candidate = super().enrich_trade_plan(record, plan)
        candidate = _apply_risk_scaled_economics(candidate, self.settings)
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


def _apply_risk_scaled_economics(
    plan: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    """Recalculate final target-stop economics with the decision risk fraction."""
    output = dict(plan)
    try:
        entry = float(output["entry_reference"])
        stop_pct = abs(float(output["stop_percent"]))
        target_pct = abs(float(output["target_percent"]))
    except (KeyError, TypeError, ValueError):
        return output
    if entry <= 0 or stop_pct <= 0 or target_pct <= 0:
        return output

    strategy = settings.section("strategy")
    account = float(strategy.get("account_equity_usd", 1000.0))
    risk_fraction = float(
        output.get(
            "risk_fraction",
            strategy.get("risk_per_trade_fraction", 0.0125),
        )
    )
    minimum_fraction = float(
        strategy.get("minimum_risk_per_trade_fraction", 0.005)
    )
    maximum_fraction = float(
        strategy.get("maximum_risk_per_trade_fraction", 0.03)
    )
    risk_fraction = float(
        np.clip(risk_fraction, minimum_fraction, maximum_fraction)
    )
    risk_budget = account * risk_fraction
    maximum_leverage = float(strategy.get("maximum_leverage", 5.0))
    stress_bps = float(output.get("stress_execution_cost_bps", 0.0))
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
    predicted_r = float(output.get("adaptive_predicted_r", 0.0))
    p_target = float(output.get("adaptive_target_probability", 0.5))
    p_stop = float(output.get("adaptive_stop_probability", 0.4))
    p_expiry = max(0.0, 1.0 - p_target - p_stop)
    expiry_net = notional * predicted_r * stop_pct - execution_cost
    expected_value = (
        p_target * target_net + p_stop * stop_net + p_expiry * expiry_net
    )
    output.update(
        {
            "risk_fraction": risk_fraction,
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
        }
    )
    return output


def open_trade_with_context(
    record: dict[str, Any],
) -> dict[str, Any] | None:
    trade = _BASE_OPEN_TRADE(record)
    if trade is None:
        return None
    context = record.get("event_candle_context")
    if isinstance(context, dict):
        trade["event_candle_context"] = context
        trade["candle_context_contract"] = str(
            record.get("candle_context_contract") or CONTEXT_CONTRACT
        )
        trade["candle_context_complete"] = bool(
            record.get("candle_context_complete", context.get("complete", False))
        )
    plan = record.get("trade_plan")
    plan = plan if isinstance(plan, dict) else {}
    trade["entry_definition"] = str(
        plan.get("entry_definition", "PAPER_MARKET_ORDER_AT_SIGNAL_RUN")
    )
    trade["entry_reference_kind"] = plan.get("entry_reference_kind")
    trade["entry_quote_provider"] = plan.get("entry_quote_provider")
    trade["entry_quote_time"] = plan.get("entry_quote_time")
    trade["source_candle_close"] = plan.get(
        "source_candle_close",
        record.get("price"),
    )
    trade["policy_name"] = str(
        plan.get("policy_name") or "LEGACY_AGGRESSIVE_PAPER"
    )
    trade["policy_version"] = int(plan.get("policy_version", 1))
    trade["risk_contract_version"] = int(
        plan.get("risk_contract_version", 1)
    )
    trade["entry_contract"] = str(
        plan.get("entry_contract") or "LEGACY_FIXED_RISK"
    )
    trade["risk_score"] = float(plan.get("risk_score", 0.0))
    trade["risk_fraction"] = float(
        plan.get("risk_fraction", 0.0)
    )
    trade["soft_risk_flags"] = list(plan.get("soft_risk_flags", []))
    trade["qualification_passed"] = bool(
        plan.get("qualification_passed", False)
    )
    trade["direction_qualified"] = bool(
        plan.get("direction_qualified", False)
    )
    execution_quote = record.get("execution_quote")
    if isinstance(execution_quote, dict):
        trade["execution_quote"] = execution_quote
    return trade


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    state_dir = root / ".github_state"
    history_path = state_dir / "history.json"
    latest_path = state_dir / "latest.json"
    history = _load_list(history_path)
    directional_history = [
        item
        for item in history
        if _is_directional_record(item)
    ]
    if directional_history != history:
        state_dir.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                directional_history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        latest = _load_dict(latest_path)
        if latest and not _is_directional_record(latest):
            latest_path.write_text("{}\n", encoding="utf-8")

    install_context_trade_features(trade_lifecycle_module)
    github_hourly_forecast.fetch_and_store = (
        fetch_latest_contiguous_and_store
    )
    github_hourly_forecast.RuntimeEngine = ContextRuntimeEngine
    github_hourly_forecast.AdaptiveTradeEngine = CanonicalAdaptiveTradeEngine
    github_hourly_forecast.open_trade_from_record = open_trade_with_context
    github_hourly_forecast.build_next_candle_forecast = (
        build_strict_next_candle_forecast
    )
    return github_hourly_forecast.main()


def _is_directional_record(item: dict[str, Any]) -> bool:
    return str(item.get("model_id") or "").startswith(MODEL_PREFIX)


def _load_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _load_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
