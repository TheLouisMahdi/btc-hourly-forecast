"""Paired hourly exit-path replay for fixed recorded paper entries.

Run once with main's imported btc_ema_trader package, once with the review
package. This is a limited exit-only diagnostic, not a strategy backtest.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from btc_ema_trader.config import Settings
from btc_ema_trader.execution_path import (
    first_full_candle_open,
    resolve_open_trades_after_entry,
)


def _utc(value: str) -> datetime:
    ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError("Naive timestamp")
    return ts.astimezone(timezone.utc)


def replay_entries(
    trades: list[dict],
    candles: list[dict],
) -> dict:
    """Reprice only recorded entries fully covered by hourly candle history."""
    frame = pd.DataFrame(candles)
    if frame.empty:
        raise ValueError("No price history")
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame = frame.sort_values("open_time").reset_index(drop=True)
    timestamps = frame["open_time"]
    if timestamps.duplicated().any() or not timestamps.diff().dropna().eq(
        pd.Timedelta(hours=1)
    ).all():
        raise ValueError("Hourly chart candles must be contiguous and unique")
    start = timestamps.iloc[0]
    end = timestamps.iloc[-1] + pd.Timedelta(hours=1)
    settings = Settings(
        root=Path.cwd(),
        values={"trade_lifecycle": {"same_bar_policy": "STOP_FIRST"}},
    )
    results = []
    skipped = []
    for source in trades:
        if source.get("status") != "CLOSED":
            continue
        identifier = str(source["trade_id"])
        entry_at = pd.Timestamp(_utc(source["opened_at"]))
        historical_exit = pd.Timestamp(_utc(source["closed_at"]))
        first_open = first_full_candle_open(entry_at)
        if entry_at < start or historical_exit > end or first_open >= end:
            skipped.append(identifier)
            continue
        if not (pd.notna(source.get("entry_price")) and source.get("initial_stop_price")):
            skipped.append(identifier)
            continue
        case = copy.deepcopy(source)
        case["status"] = "OPEN"
        case["current_stop_price"] = case["initial_stop_price"]
        case["max_favorable_r"] = 0.0
        case["max_adverse_r"] = 0.0
        case["breakeven_armed"] = False
        case["trailing_armed"] = False
        case["adaptive_learned"] = False
        for field in (
            "outcome", "closed_at", "exit_price", "fill_reason",
            "gross_pnl_usd", "realized_net_pnl_usd", "realized_r",
            "exit_time_basis", "exit_evidence_available_at",
        ):
            case.pop(field, None)
        resolved = resolve_open_trades_after_entry([case], frame, settings)
        results.append(
            {
                "trade_id": identifier,
                "event_type": source.get("event_type"),
                "original_recorded_net_pnl_usd": source["realized_net_pnl_usd"],
                "original_recorded_outcome": source.get("outcome"),
                "resolved": bool(resolved),
                "replayed_outcome": case.get("outcome"),
                "replayed_exit_price": case.get("exit_price"),
                "replayed_net_pnl_usd": case.get("realized_net_pnl_usd"),
                "replayed_closed_at": case.get("closed_at"),
                "replayed_fill_reason": case.get("fill_reason"),
            }
        )
    return {
        "analysis_type": "FIXED_ENTRY_HOURLY_EXIT_PATH_DIAGNOSTIC",
        "assumed_same_bar_policy": "STOP_FIRST",
        "coverage_start": start.isoformat(),
        "coverage_end": end.isoformat(),
        "eligible_recorded_entries": len(results),
        "skipped_recorded_entries": len(skipped),
        "skipped_ids": skipped,
        "trades": results,
    }


def compare_replays(baseline: dict, candidate: dict) -> dict:
    before = {t["trade_id"]: t for t in baseline["trades"]}
    after = {t["trade_id"]: t for t in candidate["trades"]}
    if set(before) != set(after):
        raise ValueError("Versions must replay identical trade IDs")
    paired = []
    for ident in sorted(before):
        original, changed = before[ident], after[ident]
        both = bool(original["resolved"] and changed["resolved"])
        a = original.get("replayed_net_pnl_usd")
        b = changed.get("replayed_net_pnl_usd")
        delta = float(b) - float(a) if both else None
        if delta is not None and not math.isfinite(delta):
            raise ValueError("Non-finite replay difference")
        paired.append(
            {
                "trade_id": ident,
                "event_type": original.get("event_type"),
                "both_resolved": both,
                "baseline_net_pnl_usd": a,
                "candidate_net_pnl_usd": b,
                "delta_net_pnl_usd": delta,
                "baseline_outcome": original["replayed_outcome"],
                "candidate_outcome": changed["replayed_outcome"],
                "baseline_fill": original["replayed_fill_reason"],
                "candidate_fill": changed["replayed_fill_reason"],
                "baseline_closed_at": original["replayed_closed_at"],
                "candidate_closed_at": changed["replayed_closed_at"],
                "recorded_observed_pnl_usd": original["original_recorded_net_pnl_usd"],
            }
        )
    resolved = [t for t in paired if t["both_resolved"]]
    return {
        "analysis_type": "SAME_RECORDED_ENTRIES_EXIT_REPLAY_ONLY",
        "warning": (
            "This is NOT a full strategy backtest: only historical entries already "
            "observed in the ledger are replayed; entry policy, skipped signals, "
            "online learning and missed partial-hour stop/target hits are NOT modeled. "
            "Live OHLC and historical paper execution may have different venues and "
            "fees. PnL deltas do not forecast future returns."
        ),
        "paired_resolved": len(resolved),
        "eligible_entries": len(paired),
        "candidate_minus_baseline_net_pnl_usd": round(
            sum(t["delta_net_pnl_usd"] for t in resolved), 6
        ),
        "baseline_replay_net_pnl_usd": round(
            sum(t["baseline_net_pnl_usd"] for t in resolved), 6
        ),
        "candidate_replay_net_pnl_usd": round(
            sum(t["candidate_net_pnl_usd"] for t in resolved), 6
        ),
        "paired": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path)
    parser.add_argument("--candles", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline and args.candidate:
        result = compare_replays(
            json.loads(args.baseline.read_text("utf-8")),
            json.loads(args.candidate.read_text("utf-8")),
        )
    elif args.trades and args.candles:
        result = replay_entries(
            json.loads(args.trades.read_text("utf-8")),
            json.loads(args.candles.read_text("utf-8")),
        )
    else:
        parser.error("Supply --trades and --candles, or --baseline and --candidate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {key: value for key, value in result.items() if key not in ("trades", "paired")},
        indent=2,
    ))


if __name__ == "__main__":
    main()
