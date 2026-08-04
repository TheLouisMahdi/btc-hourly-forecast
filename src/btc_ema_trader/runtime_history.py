from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import Settings
from .market import MarketDataClient
from .storage import Database

LOGGER = logging.getLogger(__name__)


def latest_contiguous_tail(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the newest uninterrupted hourly segment without filling gaps."""
    if frame.empty:
        raise ValueError("Provider returned no closed hourly candles")

    prepared = (
        frame.copy()
        .sort_values("open_time")
        .drop_duplicates("open_time", keep="last")
        .reset_index(drop=True)
    )
    prepared["open_time"] = pd.to_datetime(
        prepared["open_time"],
        utc=True,
    )
    differences = (
        prepared["open_time"]
        .diff()
        .dt.total_seconds()
        .div(3600.0)
    )
    continuous = differences.between(
        1.0 - 1e-9,
        1.0 + 1e-9,
        inclusive="both",
    )
    boundaries = differences.notna() & ~continuous
    start_index = int(boundaries[boundaries].index[-1]) if boundaries.any() else 0
    tail = prepared.iloc[start_index:].reset_index(drop=True)

    positive_gaps = differences[differences > 1.0 + 1e-9]
    largest_gap = (
        float(positive_gaps.max()) if not positive_gaps.empty else 0.0
    )
    missing_hours = int(
        sum(
            max(int(round(float(value))) - 1, 0)
            for value in positive_gaps.tolist()
        )
    )
    audit = {
        "policy": "LATEST_CONTIGUOUS_SEGMENT_WITHOUT_FILL",
        "raw_rows": int(len(prepared)),
        "selected_rows": int(len(tail)),
        "discarded_older_rows": int(len(prepared) - len(tail)),
        "gap_count": int(len(positive_gaps)),
        "largest_gap_hours": largest_gap,
        "missing_candle_hours": missing_hours,
        "selected_start": pd.Timestamp(tail["open_time"].iloc[0]).isoformat(),
        "selected_end": pd.Timestamp(tail["open_time"].iloc[-1]).isoformat(),
        "synthetic_candles": False,
        "interpolation": False,
        "forward_fill": False,
    }
    return tail, audit


def fetch_latest_contiguous_and_store(
    settings: Settings,
    database: Database,
    days: float | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Fetch live history and persist only its newest continuous segment."""
    client = MarketDataClient(settings)
    market_cfg = settings.section("market")
    requested_days = float(days or 180.0)
    requested = provider or str(market_cfg.get("provider", "auto"))
    providers = (
        [requested]
        if requested != "auto"
        else list(market_cfg.get("provider_order", []))
    )
    minimum_rows = int(
        market_cfg.get("runtime_minimum_contiguous_rows", 1000)
    )
    errors: dict[str, str] = {}

    for name in providers:
        try:
            raw = client._fetch_provider(name, requested_days)
            selected, audit = latest_contiguous_tail(raw)
            if len(selected) < minimum_rows:
                raise ValueError(
                    "Latest continuous market segment has only "
                    f"{len(selected)} candles; at least {minimum_rows} "
                    "are required for runtime features"
                )
            client._validate(
                selected,
                days=max(float(len(selected)) / 24.0, 1.0),
                enforce_minimum_rows=False,
            )
            database.upsert_candles(selected)
            return {
                "provider": name,
                "rows": int(len(selected)),
                "first": selected["open_time"].min().isoformat(),
                "last": selected["open_time"].max().isoformat(),
                "requested_days": requested_days,
                "minimum_contiguous_rows": minimum_rows,
                "continuity": audit,
            }
        except Exception as exc:
            errors[str(name)] = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "Runtime market provider %s failed: %s",
                name,
                exc,
            )

    raise RuntimeError(
        "All runtime market providers failed: "
        f"{errors}"
    )
