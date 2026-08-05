from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import github_hourly_forecast
from btc_ema_trader.candle_context import (
    CONTEXT_CONTRACT,
    extract_candle_context,
)
from btc_ema_trader.runtime_history import (
    fetch_latest_contiguous_and_store,
)
from btc_ema_trader.strict_forecast_contract import (
    build_strict_next_candle_forecast,
)

MODEL_PREFIX = "directional-breakout-hourly-"
_BASE_RUNTIME_ENGINE = github_hourly_forecast.RuntimeEngine
_BASE_OPEN_TRADE = github_hourly_forecast.open_trade_from_record


class ContextRuntimeEngine:
    """Decorate the runtime result with an auditable causal candle window."""

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
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


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

    github_hourly_forecast.fetch_and_store = (
        fetch_latest_contiguous_and_store
    )
    github_hourly_forecast.RuntimeEngine = ContextRuntimeEngine
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
