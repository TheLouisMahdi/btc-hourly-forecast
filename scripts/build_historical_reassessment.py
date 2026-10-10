"""Publish a read-only historical trade reassessment beside the original ledger.

Recorded fills and PnL are immutable; the independent replay estimate is
research-only, and is never used to train online models or overwrite trades.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compare_observed_paper_trades import STRUCTURAL_EVENTS, metrics, validate_ledger
from replay_recorded_exits import replay_entries

SCHEMA_VERSION = 1
REPLAY_METHOD = "SAME_RECORDED_ENTRY_HOURLY_EXIT_REPLAY_V1"


def _identity(trade: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match before carrying a previous replay forward."""
    return {
        "opened_at": trade["opened_at"],
        "closed_at": trade["closed_at"],
        "original_net_pnl_usd": float(trade["realized_net_pnl_usd"]),
    }


def build_reassessment(
    ledger: list[dict[str, Any]],
    candles: list[dict[str, Any]],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    closed = validate_ledger(ledger)
    structural = [t for t in closed if t.get("event_type") in STRUCTURAL_EVENTS]
    model_only = [t for t in closed if t.get("event_type") not in STRUCTURAL_EVENTS]
    previous = previous if isinstance(previous, dict) else {}
    existing = {
        row.get("trade_id"): row
        for row in previous.get("entries", [])
        if isinstance(row, dict)
    }
    by_id: dict[str, dict[str, Any]] = {}
    for trade in closed:
        ident = trade["trade_id"]
        is_structural = trade.get("event_type") in STRUCTURAL_EVENTS
        row = {
            "trade_id": ident,
            "event_type": trade.get("event_type", "NONE"),
            "recorded_direction": trade.get("direction"),
            **_identity(trade),
            "historical_policy_class": (
                "STRUCTURAL_ENTRY_CANDIDATE_RECHECK_OTHER_GATES"
                if is_structural
                else "MODEL_ONLY_ENTRY_BLOCKED_BY_NEW_POLICY"
            ),
            "recorded_result_is_immutable": True,
        }
        prior = existing.get(ident, {})
        if (
            prior.get("replay_method") == REPLAY_METHOD
            and all(prior.get(k) == value for k, value in _identity(trade).items())
            and isinstance(prior.get("exit_replay_estimate"), dict)
        ):
            row["replay_method"] = REPLAY_METHOD
            row["exit_replay_estimate"] = prior["exit_replay_estimate"]
        by_id[ident] = row

    replay_warning = None
    try:
        replay = replay_entries(ledger, candles)
        for item in replay["trades"]:
            if not item.get("resolved"):
                continue
            row = by_id[item["trade_id"]]
            row["replay_method"] = REPLAY_METHOD
            row["exit_replay_estimate"] = {
                "net_pnl_usd": item["replayed_net_pnl_usd"],
                "exit_price": item["replayed_exit_price"],
                "closed_at": item["replayed_closed_at"],
                "outcome": item["replayed_outcome"],
                "fill_reason": item["replayed_fill_reason"],
            }
    except ValueError as exc:
        replay_warning = f"Recent price series is not safely replayable: {exc}"

    entries = list(by_id.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_type": "HISTORICAL_REASSESSMENT_NOT_COUNTERFACTUAL_BACKTEST",
        "recorded_ledger_untouched": True,
        "current_policy": (
            "Direction-only forecasts cannot open new entries without a structural event. "
            "Previously opened positions continue through their existing exit lifecycle."
        ),
        "original_closed_trades": metrics(closed),
        "historically_structural_subset": metrics(structural),
        "historically_model_only_subset": metrics(model_only),
        "historically_profitable_but_now_blocked_entries": sum(
            float(t["realized_net_pnl_usd"]) > 0 for t in model_only
        ),
        "currently_open_trades_are_not_repriced": [
            t.get("trade_id") for t in ledger if t.get("status") == "OPEN"
        ],
        "previously_replayed_trades_retained_when_candles_roll_off": True,
        "closed_trades_with_hourly_exit_replay": sum(
            isinstance(t.get("exit_replay_estimate"), dict) for t in entries
        ),
        "replay_warning": replay_warning,
        "entries": entries,
        "limitations": [
            "Recorded outcomes, fills, historical forecasts and live-learning labels are NOT rewritten.",
            "Exit replay is based on immutable recorded entries and available hourly candles.",
            "This is not a counterfactual trading-strategy backtest: blocked entries change later opportunities and learning.",
            "The currently open position remains managed by the live runtime, regardless of its original entry class.",
            "No intrahour execution timing or missing older candle paths can be reconstructed from hourly OHLC.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = {}
    if args.previous and args.previous.exists():
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
    report = build_reassessment(
        json.loads(args.trades.read_text(encoding="utf-8")),
        json.loads(args.candles.read_text(encoding="utf-8")),
        previous=previous,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "report_type": report["report_type"],
        "closed": report["original_closed_trades"]["closed_trades"],
        "open": len(report["currently_open_trades_are_not_repriced"]),
        "replayed": report["closed_trades_with_hourly_exit_replay"],
        "warning": report["replay_warning"],
    }))


if __name__ == "__main__":
    main()
