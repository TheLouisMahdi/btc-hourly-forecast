"""Forward-only audit of the existing SMA200 paper bot; never place or change trades.

Measures paper quote drift versus the next-day UTC opening quote and marks
SMA200 paper equity versus a same-timestamp, fee-aware buy/hold and cash.
All metrics are descriptive: Coinbase paper quotes are NOT real exchange fills.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import sma200_shadow as bot
from rule_based import request_json, load, save, utc

DAY = bot.DAY
HOUR = 3600
INITIAL = bot.INITIAL_CASH


def as_seconds(value: str) -> int:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Missing timezone on paper event")
    return int(dt.timestamp())


def positive(value) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Nonpositive or invalid paper price")
    return number


def config() -> dict:
    fields = {
        "strategy": "SMA200_DAILY_SPOT_V1",
        "period_days": bot.PERIOD,
        "fee_bps_per_side": bot.FEE_BPS,
        "slippage_bps_per_side": bot.SLIPPAGE_BPS,
        "max_allocation": bot.MAX_ALLOCATION,
        "initial_cash": bot.INITIAL_CASH,
        "state_version": bot.STATE_VERSION,
    }
    fields["sha256"] = hashlib.sha256(
        json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return fields


def paper_events(state: dict) -> list[dict]:
    """Extract authoritative execution events, not proposed signals."""
    all_positions = list(state["trades"])
    if state["qty"] > 0:
        if not isinstance(state["position"], dict):
            raise ValueError("Missing open position")
        all_positions.append(state["position"])
    elif state["position"] is not None:
        raise ValueError("Unexpected open position with zero BTC")
    events = []
    for pos in all_positions:
        for side, prefix in (("BUY", "entry"), ("SELL", "exit")):
            if prefix == "exit" and "exit_time" not in pos:
                continue
            at = as_seconds(pos[prefix + "_time"])
            candle = as_seconds(pos[prefix + "_candle"])
            planned = candle + DAY
            if at < planned:
                raise ValueError("Paper execution precedes signal's next-day boundary")
            size = positive(pos["quantity"])
            fill = positive(pos[prefix + "_price"])
            slip = bot.SLIPPAGE_BPS / 10000
            paper_quote = fill / (1 + slip if side == "BUY" else 1 - slip)
            events.append({
                "id": f"{side}:{pos['entry_time']}:{pos[prefix + '_time']}",
                "side": side,
                "signal_candle": utc(candle),
                "planned_open_time": utc(planned),
                "paper_execution_time": utc(at),
                "delay_minutes": round((at - planned) / 60, 2),
                "quantity": size,
                "paper_quote_usd": paper_quote,
                "modeled_fill_usd": fill,
                "modeled_fee_usd": float(pos[prefix + "_fee"]),
            })
    events.sort(key=lambda item: (item["paper_execution_time"], item["side"]))
    if len({x["id"] for x in events}) != len(events):
        raise ValueError("Duplicate paper execution IDs")
    return events


def hourly_open(utc_start: int, now: int) -> float | None:
    """Get the completed first hour of a UTC day, never an unfinished candle."""
    if utc_start % DAY or utc_start + HOUR > now - 60:
        return None
    raw = request_json(
        f"/products/{bot.PRODUCT}/candles",
        {"start": utc(utc_start), "end": utc(utc_start + HOUR),
         "granularity": HOUR},
    )
    if not isinstance(raw, list):
        raise ValueError("Invalid Coinbase opening-hour response")
    matching = [x for x in raw if isinstance(x, list) and len(x) == 6
                and int(x[0]) == utc_start]
    if len(matching) != 1:
        return None
    return positive(matching[0][3])


def max_drawdown(values: list[float]) -> float:
    peak = INITIAL
    dd = 0.0
    for value in values:
        peak = max(peak, value)
        dd = max(dd, (peak - value) / peak)
    return round(100 * dd, 3)


def audit(shadow: dict, old: dict, now: int, fetch_open=hourly_open) -> dict:
    state = shadow["state"]
    latest = shadow["latest"]
    history = shadow["history"]
    candles = shadow["candles"]
    trades = shadow["trades"]
    signature = config()
    if state.get("version") != bot.STATE_VERSION or latest.get("strategy") != signature["strategy"]:
        raise ValueError("Wrong SMA state or strategy; refuse to audit")
    if latest.get("mode") != "PAPER_ONLY" or latest.get("run_status") != "OK":
        raise ValueError("Shadow snapshot is not a successful paper decision")
    if not history or latest != history[-1]:
        raise ValueError("Latest decision and daily history disagree")
    if not isinstance(state.get("trades"), list) or trades != state["trades"]:
        raise ValueError("State and authoritative trade ledger disagree")
    if not isinstance(candles, list) or not candles:
        raise ValueError("Missing audited daily Coinbase candles")
    if (latest.get("fee_bps_per_side") != bot.FEE_BPS
            or latest.get("slippage_bps_per_side") != bot.SLIPPAGE_BPS
            or latest.get("allocation_cap") != bot.MAX_ALLOCATION):
        raise ValueError("SMA paper parameter drift detected")
    if old.get("config_sha256") not in (None, signature["sha256"]):
        raise ValueError("Existing audit uses different SMA settings")
    if not math.isclose(float(latest["cash_usd"]), float(state["cash"]), abs_tol=.01):
        raise ValueError("Cash ledger mismatch")
    if not math.isclose(float(latest["btc_quantity"]), float(state["qty"]), abs_tol=1e-9):
        raise ValueError("BTC ledger mismatch")
    if as_seconds(latest["time"]) > now + 60:
        raise ValueError("Paper event timestamp is in the future")

    # Historical paper observations are actual recorded events only.
    # Do NOT reconstruct the missing days or substitute historical candles.
    snapshots = list(old.get("snapshots", []))
    seen = {x["candle_time"]: x for x in snapshots}
    for record in history:
        if record.get("mode") != "PAPER_ONLY" or record.get("run_status") != "OK":
            raise ValueError("Malformed paper observation")
        if record["candle_time"] in seen:
            previous = seen[record["candle_time"]]
            if (record["time"] != previous["observed_time"]
                    or not math.isclose(float(record["equity_usd"]),
                                        float(previous["paper_equity_usd"]), abs_tol=.01)):
                raise ValueError("Historical paper snapshot was modified")
            continue
        t = as_seconds(record["time"])
        if snapshots and t <= as_seconds(snapshots[-1]["observed_time"]):
            raise ValueError("New snapshot is earlier than stored forward ledger")
        row = {"candle_time": record["candle_time"],
               "observed_time": record["time"],
               "paper_quote_usd": positive(record["price"]),
               "paper_equity_usd": positive(record["equity_usd"]),
               "paper_cash_usd": float(record["cash_usd"]),
               "btc_quantity": float(record["btc_quantity"]),
               "action": record["action"],
               "missed_days": int(record.get("missed_days", 0))}
        snapshots.append(row)
        seen[row["candle_time"]] = row
    if not snapshots or snapshots[-1]["candle_time"] != latest["candle_time"]:
        raise ValueError("Latest shadow candle not present in audit snapshots")

    first_quote = positive(snapshots[0]["paper_quote_usd"])
    start_time = snapshots[0]["observed_time"]
    fee = bot.FEE_BPS / 10000
    slip = bot.SLIPPAGE_BPS / 10000
    buy_hold_qty = INITIAL / (first_quote * (1 + slip) * (1 + fee))
    for item in snapshots:
        item["sma_return_pct"] = round(
            100 * (item["paper_equity_usd"] / INITIAL - 1), 4)
        item["hold_equity_usd"] = round(
            buy_hold_qty * item["paper_quote_usd"], 4)
        item["hold_return_pct"] = round(
            100 * (item["hold_equity_usd"] / INITIAL - 1), 4)

    # A rolling closed-daily bar contains its day's official opening quote.
    # For an execution during the still-open day, read its completed 00:00 hour.
    daily_open = {x["time"]: positive(x["open"]) for x in candles}
    saved_events = {x["id"]: x for x in old.get("executions", [])}
    events = []
    for event in paper_events(state):
        planned = as_seconds(event["planned_open_time"])
        prior = saved_events.get(event["id"])
        ref = daily_open.get(planned)
        if ref is None and prior and prior.get("reference_open_usd") is not None:
            ref = positive(prior["reference_open_usd"])
        if ref is None:
            ref = fetch_open(planned, now)
        outcome = dict(event)
        outcome["reference_open_usd"] = ref
        if ref is None:
            outcome["status"] = "PENDING_REFERENCE"
            outcome["adverse_price_drift_bps"] = None
            outcome["incremental_cost_usd_at_recorded_qty"] = None
        else:
            signed = 1 if event["side"] == "BUY" else -1
            drift = signed * (event["paper_quote_usd"] / ref - 1)
            outcome["status"] = "MEASURED"
            outcome["adverse_price_drift_bps"] = round(10_000 * drift, 3)
            outcome["incremental_cost_usd_at_recorded_qty"] = round(
                signed * event["quantity"] * (event["paper_quote_usd"] - ref), 4)
        if prior and prior.get("reference_open_usd") is not None:
            if outcome["reference_open_usd"] != prior["reference_open_usd"]:
                raise ValueError("Reference price was modified in a later audit")
        events.append(outcome)

    measured = [x for x in events if x["status"] == "MEASURED"]
    sma_values = [float(x["paper_equity_usd"]) for x in snapshots]
    hold_values = [float(x["hold_equity_usd"]) for x in snapshots]
    last = snapshots[-1]
    current_paper_liquidation_equity = (
        float(state["cash"]) + float(state["qty"]) * last["paper_quote_usd"]
        * (1 - slip) * (1 - fee))
    report = {
        "status": "OK",
        "mode": "PAPER_ONLY",
        "research_only": True,
        "as_of": last["observed_time"],
        "first_observed": start_time,
        "config": signature,
        "observed_daily_decisions": len(snapshots),
        "calendar_days_since_start": 1 + (
            as_seconds(last["candle_time"]) - as_seconds(snapshots[0]["candle_time"])) // DAY,
        "paper_closed_trades": len(trades),
        "paper_latest_action": latest["action"],
        "paper_latest_signal": latest["signal"],
        "paper_mark_to_market_equity_usd": round(last["paper_equity_usd"], 2),
        "paper_liquidation_estimate_usd": round(current_paper_liquidation_equity, 2),
        "sma_net_return_pct": last["sma_return_pct"],
        "sma_max_observed_drawdown_pct": max_drawdown(sma_values),
        "hold_mark_to_market_equity_usd": round(last["hold_equity_usd"], 2),
        "hold_return_pct": last["hold_return_pct"],
        "hold_max_observed_drawdown_pct": max_drawdown(hold_values),
        "cash_equity_usd": INITIAL,
        "cash_return_pct": 0.0,
        "execution_events": len(events),
        "execution_events_measured": len(measured),
        "execution_events_reference_pending": len(events) - len(measured),
        "average_execution_delay_minutes": (
            round(sum(x["delay_minutes"] for x in measured) / len(measured), 2)
            if measured else None),
        "average_adverse_price_drift_bps": (
            round(sum(x["adverse_price_drift_bps"] for x in measured) / len(measured), 3)
            if measured else None),
        "total_incremental_cost_usd_at_recorded_qty": round(
            sum(x["incremental_cost_usd_at_recorded_qty"] for x in measured), 4),
        "notes": [
            "Shadow equity is measured only at actual daily paper decisions, not at constant clock times.",
            "Paper quote is the last reported Coinbase ticker, not a real exchange order fill.",
            "Opening reference is the historical UTC next-day open; adverse drift is timing-related, not measured spread/slippage.",
            "Positive adverse price drift means worse paper execution quote than historical next-open proxy.",
            "Realized trading fees and fixed slippage already affect paper equity; adverse drift is diagnostic, not another fee to deduct.",
            "Buy & Hold uses the first recorded paper quote with modeled entry costs and a mark-to-market, not a real executed BTC position.",
            "Paper benchmark and cash share the same observation timestamps; no unrecorded daily returns are fabricated.",
            "Paper liquidation estimate includes a hypothetical exit fee and assumed slippage not incurred yet.",
            "Drawdowns are based only on recorded paper decision snapshots and may understate intraday drawdowns.",
        ],
    }
    return {"report": report, "snapshots": snapshots, "executions": events}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-dir", type=Path, default=Path(".sma_shadow"))
    parser.add_argument("--out-dir", type=Path, default=Path(".sma_audit"))
    args = parser.parse_args()
    source = args.shadow_dir
    payload = {name: load(source / (name + ".json"), None)
               for name in ("latest", "state", "history", "trades", "candles")}
    if any(val is None for val in payload.values()):
        raise SystemExit("SMA shadow ledger incomplete: audit not published")
    prior = {
        "config_sha256": load(args.out_dir / "report.json", {}).get("config", {}).get("sha256"),
        "snapshots": load(args.out_dir / "snapshots.json", []),
        "executions": load(args.out_dir / "executions.json", []),
    }
    result = audit(payload, prior, int(time.time()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for file in ("snapshots", "executions", "report"):
        save(args.out_dir / (file + ".json"), result[file])
    print(json.dumps(result["report"], indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
