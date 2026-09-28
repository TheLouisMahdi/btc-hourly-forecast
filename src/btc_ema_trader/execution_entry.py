from __future__ import annotations

from typing import Any

import pandas as pd

EXECUTION_ENTRY_CONTRACT = "LIVE_QUOTE_AT_SIGNAL_RUN"
BATCH_LABEL_ENTRY_CONTRACT = "NEXT_HOURLY_OPEN"


def apply_execution_quote(
    record: dict[str, Any],
    *,
    provider: str,
    price: float,
    quote_time: Any,
    observed_at: Any,
    maximum_age_seconds: float,
) -> dict[str, Any]:
    """Bind a candidate paper entry to a fresh execution-time quote.

    The source candle close remains unchanged because it belongs to the exact
    next-close forecast contract. Position barriers are rebased to the actual
    paper entry so stop distance, target distance and R:R remain internally
    consistent after the live quote replaces the signal-time close.
    """
    quote_price = float(price)
    if quote_price <= 0:
        raise ValueError("Execution quote price must be positive")
    quote_timestamp = _utc(quote_time)
    observation_timestamp = _utc(observed_at)
    age_seconds = float(
        (observation_timestamp - quote_timestamp).total_seconds()
    )
    if age_seconds < -5.0:
        raise ValueError("Execution quote timestamp is unexpectedly in the future")
    if age_seconds > float(maximum_age_seconds):
        raise ValueError(
            "Execution quote is stale: "
            f"{age_seconds:.1f}s exceeds {float(maximum_age_seconds):.1f}s"
        )

    output = dict(record)
    output["execution_quote"] = {
        "contract": EXECUTION_ENTRY_CONTRACT,
        "provider": str(provider),
        "price": quote_price,
        "timestamp": quote_timestamp.isoformat(),
        "observed_at": observation_timestamp.isoformat(),
        "age_seconds": max(0.0, age_seconds),
        "fresh": True,
    }
    plan = output.get("trade_plan")
    if isinstance(plan, dict):
        plan = dict(plan)
        previous_entry = _positive_or_none(plan.get("entry_reference"))
        plan["source_candle_close"] = output.get("price")
        if previous_entry is not None:
            plan["signal_entry_reference"] = previous_entry
        plan["entry_reference"] = quote_price
        plan["entry_reference_kind"] = EXECUTION_ENTRY_CONTRACT
        plan["entry_definition"] = "PAPER_ENTRY_AT_OBSERVED_LIVE_QUOTE"
        plan["entry_quote_provider"] = str(provider)
        plan["entry_quote_time"] = quote_timestamp.isoformat()
        plan["entry_quote_observed_at"] = observation_timestamp.isoformat()
        plan["entry_quote_age_seconds"] = max(0.0, age_seconds)
        plan["label_execution_aligned"] = False
        plan["label_entry_definition"] = BATCH_LABEL_ENTRY_CONTRACT
        plan["runtime_entry_definition"] = EXECUTION_ENTRY_CONTRACT
        plan["execution_alignment_status"] = (
            "APPROXIMATE_UNTIL_MINUTE_LEVEL_RETRAIN"
        )
        _rebase_barriers(
            plan,
            action=str(output.get("action") or ""),
            previous_entry=previous_entry,
            execution_entry=quote_price,
        )
        output["trade_plan"] = plan
    return output


def _rebase_barriers(
    plan: dict[str, Any],
    *,
    action: str,
    previous_entry: float | None,
    execution_entry: float,
) -> None:
    if previous_entry is None or previous_entry <= 0:
        return

    direction = action.upper()
    previous_target = _positive_or_none(plan.get("target_price"))
    previous_stop = _positive_or_none(plan.get("stop_price"))
    if direction not in {"LONG", "SHORT"} and previous_target is not None:
        direction = "LONG" if previous_target > previous_entry else "SHORT"
    if direction not in {"LONG", "SHORT"}:
        return

    stop_pct = _positive_or_none(plan.get("stop_percent"))
    target_pct = _positive_or_none(plan.get("target_percent"))
    if stop_pct is None and previous_stop is not None:
        stop_pct = abs(previous_entry - previous_stop) / previous_entry
    if target_pct is None and previous_target is not None:
        target_pct = abs(previous_target - previous_entry) / previous_entry
    if stop_pct is None or target_pct is None:
        return

    if previous_stop is not None:
        plan["signal_stop_price"] = previous_stop
    if previous_target is not None:
        plan["signal_target_price"] = previous_target

    if direction == "LONG":
        stop = execution_entry * (1.0 - stop_pct)
        target = execution_entry * (1.0 + target_pct)
    else:
        stop = execution_entry * (1.0 + stop_pct)
        target = execution_entry * (1.0 - target_pct)
    if stop <= 0 or target <= 0:
        raise ValueError("Execution-time stop/target rebasing produced invalid prices")

    plan["stop_price"] = float(stop)
    plan["target_price"] = float(target)
    plan["execution_price_drift_bps"] = float(
        (execution_entry / previous_entry - 1.0) * 10_000.0
    )
    plan["execution_barriers_rebased"] = True


def _positive_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
