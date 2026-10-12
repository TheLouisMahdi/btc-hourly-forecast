"""Read-only readiness audit for chronological, executable paper-policy replays.

This script NEVER calculates replacement PnL from settled prediction labels.
Settled next-candle returns can include prices prior to the actual run time.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STRUCTURAL_EVENTS = frozenset({
    "RESISTANCE_BREAKOUT_LONG", "TRIANGLE_BREAKOUT_LONG",
    "SUPPORT_BREAKDOWN_SHORT", "TRIANGLE_BREAKDOWN_SHORT",
})
HOUR = timedelta(hours=1)
MIN_CONTINUOUS_DAYS = 28
MIN_STRUCTURAL_OBSERVATIONS = 30
MIN_REPLAYABLE_TRADES = 30
MIN_CANDLE_COVERAGE = 0.95


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value.astimezone(timezone.utc) if value.tzinfo else None


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _valid_ohlc(row: dict[str, Any], *, historical: bool) -> bool:
    names = (
        ("actual_candle_open", "actual_candle_high", "actual_candle_low", "actual_close")
        if historical else ("open", "high", "low", "close")
    )
    if not all(_positive_finite(row.get(k)) for k in names):
        return False
    o, hi, lo, c = (float(row[k]) for k in names)
    return lo <= min(o, c) and max(o, c) <= hi


def _target_open(record: dict[str, Any]) -> datetime | None:
    contract = record.get("next_candle_forecast")
    if not isinstance(contract, dict):
        contract = {}
    return _time(record.get("target_candle_open_time") or contract.get("target_open_time"))


def _target_close(record: dict[str, Any]) -> datetime | None:
    contract = record.get("next_candle_forecast")
    if not isinstance(contract, dict):
        contract = {}
    return _time(record.get("target_candle_close_time") or contract.get("target_close_time"))


def _coverage(timestamps: set[datetime]) -> dict[str, Any]:
    times = sorted(timestamps)
    if not times:
        return {
            "observed_hours": 0, "span_hours": 0, "missing_hours": 0,
            "gap_count": 0, "longest_contiguous_hours": 0, "coverage_fraction": 0.0,
            "first_open": None, "last_open": None,
        }
    span = int((times[-1] - times[0]) / HOUR) + 1
    missing = 0
    gaps = 0
    longest = current = 1
    for prev, next_time in zip(times, times[1:]):
        diff = int((next_time - prev) / HOUR)
        if diff > 1:
            gaps += 1
            missing += diff - 1
            current = 1
        else:
            current += 1
            longest = max(longest, current)
    return {
        "observed_hours": len(times),
        "span_hours": span,
        "missing_hours": missing,
        "gap_count": gaps,
        "longest_contiguous_hours": longest,
        "coverage_fraction": round(len(times) / span, 6),
        "first_open": times[0].isoformat(),
        "last_open": times[-1].isoformat(),
    }


def _first_complete_open(opened: datetime) -> datetime:
    rounded = opened.replace(minute=0, second=0, microsecond=0)
    return rounded if opened == rounded else rounded + HOUR


def _fully_covered_trade(trade: dict[str, Any], times: set[datetime]) -> bool:
    opened, closed = _time(trade.get("opened_at")), _time(trade.get("closed_at"))
    if opened is None or closed is None or closed <= opened:
        return False
    current = _first_complete_open(opened)
    last = closed.replace(minute=0, second=0, microsecond=0)
    if closed == last:
        last -= HOUR
    # The OHLC candle containing an intrabar exit must be present too.
    # Presence is not proof of intrabar fill ordering or the exit timestamp.
    if current > last:
        return False
    while current <= last:
        if current not in times:
            return False
        current += HOUR
    return True


def audit(
    history: list[dict[str, Any]],
    candles: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    if not all(isinstance(series, list) for series in (history, candles, trades)):
        raise ValueError("History, candles and trades must be JSON arrays")
    if not all(isinstance(row, dict) for series in (history, candles, trades) for row in series):
        raise ValueError("Every input record must be an object")

    candle_times: set[datetime] = set()
    invalid_chart = 0
    for row in candles:
        instant = _time(row.get("open_time"))
        if instant is None or instant.minute or instant.second or instant.microsecond or not _valid_ohlc(row, historical=False):
            invalid_chart += 1
        else:
            candle_times.add(instant)
    derived_times: set[datetime] = set()
    invalid_settlement = 0
    timing = {
        "before_target_open": 0,
        "during_target_candle": 0,
        "at_or_after_target_close": 0,
        "missing_or_invalid_timestamps": 0,
    }
    structural_events: list[dict[str, Any]] = []
    recorded_entries = 0
    economically_verified = 0
    for row in history:
        opened, closed, finished = _target_open(row), _target_close(row), _time(row.get("run_finished_at"))
        if opened is None or closed is None or finished is None or opened >= closed:
            timing["missing_or_invalid_timestamps"] += 1
        elif finished <= opened:
            timing["before_target_open"] += 1
        elif finished < closed:
            timing["during_target_candle"] += 1
        else:
            timing["at_or_after_target_close"] += 1
        if opened is not None and not (opened.minute or opened.second or opened.microsecond):
            if _valid_ohlc(row, historical=True):
                derived_times.add(opened)
            elif row.get("actual_close") is not None:
                invalid_settlement += 1
        if row.get("event_type") in STRUCTURAL_EVENTS:
            structural_events.append(row)
            plan = row.get("trade_plan")
            if not isinstance(plan, dict):
                plan = {}
            try:
                edge = float(row.get("expected_net_edge_bps"))
            except (TypeError, ValueError):
                edge = float("nan")
            if (
                row.get("qualification_passed") is True
                and plan.get("direction_qualified") is True
                and plan.get("economic_policy_available") is True
                and math.isfinite(edge) and edge > 0
            ):
                economically_verified += 1
        if row.get("action") in {"LONG", "SHORT"}:
            recorded_entries += 1

    chart = _coverage(candle_times)
    historical = _coverage(derived_times)
    first_chart = min(candle_times) if candle_times else None
    last_chart = max(candle_times) + HOUR if candle_times else None
    structurals_in_chart = sum(
        first_chart is not None
        and _time(record.get("candle_time")) is not None
        and first_chart <= _time(record["candle_time"]) < last_chart
        for record in structural_events
    )
    closed_trades = [row for row in trades if row.get("status") == "CLOSED"]
    covered_trades = sum(_fully_covered_trade(row, candle_times) for row in closed_trades)
    blockers: list[str] = []
    if historical["coverage_fraction"] < MIN_CANDLE_COVERAGE:
        blockers.append("SETTLED_HISTORICAL_OHLC_IS_DISCONTINUOUS")
    if chart["longest_contiguous_hours"] < MIN_CONTINUOUS_DAYS * 24:
        blockers.append("INDEPENDENT_CONTIGUOUS_OHLC_WINDOW_TOO_SHORT")
    if structurals_in_chart < MIN_STRUCTURAL_OBSERVATIONS:
        blockers.append("INSUFFICIENT_STRUCTURAL_SIGNALS_IN_REPLAY_WINDOW")
    if covered_trades < MIN_REPLAYABLE_TRADES:
        blockers.append("INSUFFICIENT_RECORDED_TRADES_WITH_FULL_EXIT_PATH")
    if invalid_chart or invalid_settlement:
        blockers.append("INVALID_PRICE_ROWS_PRESENT")
    return {
        "schema_version": 1,
        "analysis_type": "CHRONOLOGICAL_BACKTEST_READINESS_NOT_BACKTEST",
        "status": "READY_FOR_RESEARCH_REPLAY_ONLY" if not blockers else "INSUFFICIENT_CAUSAL_BACKTEST_DATA",
        "hypothetical_profit_or_pnl_computed": False,
        "history_records": len(history),
        "target_candle_timing": timing,
        "settled_historical_ohlc": historical,
        "recent_contiguous_chart": chart,
        "invalid_chart_rows": invalid_chart,
        "invalid_settled_ohlc_rows": invalid_settlement,
        "recorded_long_short_decisions": recorded_entries,
        "observed_structural_event_records": len(structural_events),
        "structural_event_records_in_chart": structurals_in_chart,
        "strict_economic_verified_structural_records": economically_verified,
        "recorded_closed_trades": len(closed_trades),
        "closed_trades_with_complete_chart_candles": covered_trades,
        "research_minimums": {
            "historical_ohlc_coverage": MIN_CANDLE_COVERAGE,
            "contiguous_chart_days": MIN_CONTINUOUS_DAYS,
            "structural_observations_in_chart": MIN_STRUCTURAL_OBSERVATIONS,
            "recorded_trades_with_complete_chart_candles": MIN_REPLAYABLE_TRADES,
        },
        "blockers": blockers,
        "limitations": [
            "Settled next-candle OHLC is not an executable price path from a mid-candle signal.",
            "The recent chart may come from a different venue than the execution quote.",
            "Repeated observations and overlapping positions require a causal portfolio engine.",
            "Coverage floors are diagnostic prerequisites, not statistical proof of profitability.",
            "Historical policy comparisons must use immutable decisions and chronologically separated tests.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        json.loads(args.history.read_text(encoding="utf-8")),
        json.loads(args.candles.read_text(encoding="utf-8")),
        json.loads(args.trades.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "status", "history_records", "target_candle_timing", "settled_historical_ohlc",
        "recent_contiguous_chart", "structural_event_records_in_chart",
        "closed_trades_with_complete_chart_candles", "blockers",
    )}, indent=2))


if __name__ == "__main__":
    main()
