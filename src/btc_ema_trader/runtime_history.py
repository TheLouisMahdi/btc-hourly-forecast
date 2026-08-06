from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import Settings
from .market import MarketDataClient
from .storage import Database

LOGGER = logging.getLogger(__name__)

DEFAULT_RUNTIME_FETCH_WINDOWS_DAYS = (180.0, 120.0, 90.0, 60.0, 45.0)


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


def expected_latest_closed_open(
    now: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return the expected open time of the newest fully closed 1h candle."""
    current = _utc(now or pd.Timestamp.now(tz="UTC"))
    return current.floor("h") - pd.Timedelta(hours=1)


def latest_candle_freshness(
    frame: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    maximum_lag_hours: float = 1.0,
) -> dict[str, Any]:
    """Audit whether a provider's newest closed candle is recent enough.

    Missing GitHub schedule executions are harmless because later executions
    refetch the complete range. A stale provider response is different: it must
    never be accepted as the current market state or used for online learning.
    """
    if frame.empty:
        raise ValueError("Cannot audit freshness of an empty candle frame")
    latest = _utc(pd.to_datetime(frame["open_time"], utc=True).max())
    expected = expected_latest_closed_open(now)
    lag_hours = max(
        0.0,
        float((expected - latest).total_seconds() / 3600.0),
    )
    maximum_lag = max(0.0, float(maximum_lag_hours))
    return {
        "fresh": bool(lag_hours <= maximum_lag + 1e-9),
        "latest_closed_open": latest.isoformat(),
        "expected_latest_closed_open": expected.isoformat(),
        "lag_hours": lag_hours,
        "maximum_lag_hours": maximum_lag,
    }


def fetch_latest_contiguous_and_store(
    settings: Settings,
    database: Database,
    days: float | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Fetch a fresh continuous tail with shrinking-window recovery.

    Every attempt uses the same requested provider, so a champion trained on
    Coinbase is never silently fed candles from another exchange. Shorter
    lookbacks recover from partial historical API failures while preserving at
    least the configured feature warm-up. No gap is filled or interpolated.
    """
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
    maximum_lag_hours = float(
        market_cfg.get("runtime_maximum_latest_lag_hours", 1.0)
    )
    configured_windows = market_cfg.get(
        "runtime_fetch_windows_days",
        DEFAULT_RUNTIME_FETCH_WINDOWS_DAYS,
    )
    windows = _runtime_windows(
        requested_days,
        configured_windows,
        minimum_rows,
    )
    attempts: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    for name in providers:
        for lookback_days in windows:
            attempt_key = f"{name}:{lookback_days:g}d"
            try:
                raw = client._fetch_provider(name, lookback_days)
                selected, continuity = latest_contiguous_tail(raw)
                freshness = latest_candle_freshness(
                    selected,
                    maximum_lag_hours=maximum_lag_hours,
                )
                if not freshness["fresh"]:
                    raise ValueError(
                        "Latest provider candle is stale by "
                        f"{freshness['lag_hours']:.1f} hours"
                    )
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
                attempts.append(
                    {
                        "provider": name,
                        "lookback_days": lookback_days,
                        "status": "SELECTED",
                        "rows": int(len(selected)),
                    }
                )
                return {
                    "provider": name,
                    "rows": int(len(selected)),
                    "first": selected["open_time"].min().isoformat(),
                    "last": selected["open_time"].max().isoformat(),
                    "requested_days": requested_days,
                    "selected_lookback_days": lookback_days,
                    "minimum_contiguous_rows": minimum_rows,
                    "recovery_policy": "SAME_PROVIDER_SHRINKING_WINDOW",
                    "continuity": continuity,
                    "freshness": freshness,
                    "attempts": attempts,
                }
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                errors[attempt_key] = message
                attempts.append(
                    {
                        "provider": name,
                        "lookback_days": lookback_days,
                        "status": "REJECTED",
                        "error": message,
                    }
                )
                LOGGER.warning(
                    "Runtime market provider %s (%sd) failed: %s",
                    name,
                    lookback_days,
                    exc,
                )

    raise RuntimeError(
        "All fresh runtime market fetch attempts failed: "
        f"{errors}"
    )


def _runtime_windows(
    requested_days: float,
    configured: Any,
    minimum_rows: int,
) -> list[float]:
    minimum_days = max(float(minimum_rows) / 24.0 + 1.0, 2.0)
    raw_values = [requested_days]
    if isinstance(configured, (list, tuple)):
        raw_values.extend(configured)
    else:
        raw_values.extend(DEFAULT_RUNTIME_FETCH_WINDOWS_DAYS)
    windows: list[float] = []
    for value in raw_values:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            continue
        candidate = min(candidate, requested_days)
        if candidate + 1e-9 < minimum_days:
            continue
        if not any(abs(candidate - existing) < 1e-9 for existing in windows):
            windows.append(candidate)
    windows.sort(reverse=True)
    if not windows:
        windows.append(max(requested_days, minimum_days))
    return windows


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
