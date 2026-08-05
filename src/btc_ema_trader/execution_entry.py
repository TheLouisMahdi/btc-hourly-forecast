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
    next-close forecast contract. Only the candidate trade entry is rebased.
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
        plan["source_candle_close"] = output.get("price")
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
        output["trade_plan"] = plan
    return output


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
