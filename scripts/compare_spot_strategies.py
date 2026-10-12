"""Fair chronological BTC-USD spot comparison of running hourly and SMA200 paper rules.

Pure research: no orders, no state-branch writes and no changes to production.
Signals use completed candles and next period OPEN as an execution proxy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import rule_based as hourly
import sma200_shadow as daily

START_HOUR = "2023-11-01"
START_DAY = "2023-01-01"
END = "2026-01-01"
HOUR = hourly.HOUR
DAY = daily.DAY
FEES_BPS = (15.0, 40.0, 60.0)
SLIPPAGE_BPS = 3.0
INITIAL = 10_000.0
FIELDS = ("time", "open", "high", "low", "close", "volume")


def epoch(date: str) -> int:
    return int(datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def stamp(t: int) -> str:
    return hourly.utc(t)


def validate(rows: list[dict], spacing: int) -> None:
    if not rows:
        raise ValueError("Empty Coinbase history")
    previous = None
    for r in rows:
        t = r["time"]
        if t % spacing or (previous is not None and t <= previous):
            raise ValueError("Unsorted or misaligned timestamps")
        values = [r[k] for k in FIELDS[1:]]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Nonfinite OHLCV")
        if not (0 < r["low"] <= min(r["open"], r["close"])
                <= max(r["open"], r["close"]) <= r["high"]
                and r["volume"] >= 0):
            raise ValueError("Invalid Coinbase OHLCV")
        previous = t


def fetch(start: int, end: int, granularity: int) -> list[dict]:
    if start >= end or start % granularity or end % granularity:
        raise ValueError("Invalid aligned candle request")
    indexed = {}
    for first in range(start, end, 288 * granularity):
        last = min(first + 288 * granularity, end)
        for attempt in range(5):
            try:
                raw = hourly.request_json(
                    f"/products/{hourly.PRODUCT}/candles",
                    {"start": stamp(first), "end": stamp(last),
                     "granularity": granularity})
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))
        if not isinstance(raw, list):
            raise ValueError("Invalid Coinbase response")
        for v in raw:
            if not isinstance(v, list) or len(v) != 6:
                raise ValueError("Invalid Coinbase bar payload")
            t, low, high, opened, closed, volume = v
            t = int(t)
            if not first <= t < last:
                continue
            record = dict(zip(FIELDS, (t, float(opened), float(high),
                                        float(low), float(closed), float(volume))))
            if t in indexed and indexed[t] != record:
                raise ValueError("Conflicting duplicate candle")
            indexed[t] = record
        time.sleep(0.25)
    # Retry a limited number of genuine missing candles, never fabricate price bars.
    missing = [t for t in range(start, end, granularity) if t not in indexed]
    for t in missing[:72]:
        try:
            raw = hourly.request_json(
                f"/products/{hourly.PRODUCT}/candles",
                {"start": stamp(t), "end": stamp(t + granularity),
                 "granularity": granularity})
        except (HTTPError, URLError, TimeoutError):
            continue
        if not isinstance(raw, list):
            raise ValueError("Malformed narrow retry response")
        for v in raw:
            if len(v) != 6 or int(v[0]) != t:
                continue
            _, lo, hi, op, cl, vol = v
            indexed[t] = dict(zip(FIELDS, (t, float(op), float(hi),
                                           float(lo), float(cl), float(vol))))
    rows = [indexed[t] for t in sorted(indexed)]
    validate(rows, granularity)
    if rows[0]["time"] != start or rows[-1]["time"] != end - granularity:
        raise ValueError("Missing first/last required Coinbase candles")
    return rows


def write_csv(path: Path, bars: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(bars)


def read_csv(path: Path, interval: int) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing required OHLCV columns")
        bars = [{k: int(row[k]) if k == "time" else float(row[k])
                 for k in FIELDS} for row in reader]
    validate(bars, interval)
    return bars


def missing_intervals(bars: list[dict], step: int) -> list[dict]:
    return [{"first_missing": stamp(a["time"] + step),
             "resume": stamp(b["time"]),
             "count": (b["time"] - a["time"]) // step - 1}
            for a, b in zip(bars, bars[1:])
            if b["time"] - a["time"] != step]


def day_ceiling(t: int) -> int:
    return (t + DAY - 1) // DAY * DAY


def day_floor(t: int) -> int:
    return t // DAY * DAY


def windows(hour_bars: list[dict], day_bars: list[dict]) -> list[dict]:
    """Exclude missing-data intervals, require 1200 fresh hours of warmup.

    Each segment is a new 10k portfolio. Never splice PnL across a data gap.
    """
    htimes = [r["time"] for r in hour_bars]
    dset = {r["time"] for r in day_bars}
    consecutive = []
    begin = 0
    for i in range(1, len(htimes)):
        if htimes[i] - htimes[i - 1] != HOUR:
            consecutive.append((begin, i))
            begin = i
    consecutive.append((begin, len(htimes)))
    cases = []
    for label, lower, upper in (("2024", epoch("2024-01-01"), epoch("2025-01-01")),
                                ("2025", epoch("2025-01-01"), epoch("2026-01-01"))):
        for a, b in consecutive:
            first_usable = htimes[a] + hourly.WINDOW * HOUR
            # Start on a UTC day boundary. The first trade uses yesterday's
            # closed daily candle / last closed hourly candle.
            start = max(lower, day_ceiling(first_usable))
            stop = min(upper, day_floor(htimes[b - 1] + HOUR))
            if stop - start < 7 * DAY:
                continue
            if any(t not in dset for t in range(start - 200 * DAY, stop, DAY)):
                raise ValueError("Missing daily SMA200 history inside fair window")
            cases.append({"label": label, "start": start, "end": stop})
    if not cases or not {x["label"] for x in cases} == {"2024", "2025"}:
        raise ValueError("Expected full 2024 and partial 2025 comparable windows")
    return cases


def summarize(initial: float, ending: float, curve: list[float],
              trades: list[dict], allocation_time: int, n_steps: int) -> dict:
    peak = initial
    dd = 0.0
    for value in curve + [ending]:
        peak = max(peak, value)
        dd = max(dd, 1 - value / peak)
    profit = [x["pnl_usd"] for x in trades if x["pnl_usd"] > 0]
    losses = [x["pnl_usd"] for x in trades if x["pnl_usd"] <= 0]
    gross = sum(profit)
    negative = -sum(losses)
    return {
        "ending_equity_usd": round(ending, 2),
        "net_return_pct": round(100 * (ending / initial - 1), 3),
        "max_daily_drawdown_pct": round(100 * dd, 3),
        "closed_trades": len(trades),
        "win_rate_pct": round(100 * len(profit) / len(trades), 2) if trades else None,
        "profit_factor": round(gross / negative, 3) if negative else None,
        "expectancy_usd_per_closed_trade":
            round(sum(x["pnl_usd"] for x in trades) / len(trades), 2) if trades else None,
        "time_in_market_pct": round(100 * allocation_time / n_steps, 2) if n_steps else 0,
    }


def run_sma(dbar: list[dict], first: int, end: int,
            fee: float, slip: float) -> dict:
    idx = {bar["time"]: i for i, bar in enumerate(dbar)}
    state = daily.new_state()
    curve = []
    active = 0
    with patch.object(daily, "FEE_BPS", fee), patch.object(daily, "SLIPPAGE_BPS", slip):
        for t in range(first, end, DAY):
            i = idx[t]
            if i < daily.PERIOD:
                raise ValueError("Insufficient SMA warmup")
            # The last item is yesterday's closed candle; today's OPEN is the quote.
            previous = dbar[i - daily.PERIOD:i]
            if previous[-1]["time"] != t - DAY:
                raise ValueError("Daily discontinuity before SMA execution")
            current = dbar[i]
            result = daily.step(state, previous, current["open"], t)
            if result is None or result["run_status"] != "OK":
                raise RuntimeError("SMA200 replay did not advance")
            active += int(state["qty"] > 0)
            curve.append(state["cash"] + state["qty"] * current["close"])
    last_close = dbar[idx[end - DAY]]["close"]
    ending = state["cash"] + state["qty"] * last_close * (
        1 - slip / 10000) * (1 - fee / 10000)
    return summarize(INITIAL, ending, curve, state["trades"],
                     active, (end - first) // DAY)


def run_hourly(hbar: list[dict], first: int, end: int,
               fee: float, slip: float) -> dict:
    idx = {r["time"]: i for i, r in enumerate(hbar)}
    start_i = idx[first]
    end_i = idx[end - HOUR] + 1
    if start_i < hourly.WINDOW:
        raise ValueError("Insufficient hourly warmup")
    state = hourly.new_state()
    curve = []
    active_days = 0
    with patch.object(hourly, "FEE_BPS", fee), patch.object(hourly, "SLIPPAGE_BPS", slip):
        for i in range(start_i - 1, end_i - 1):
            history = hbar[i - hourly.WINDOW + 1:i + 1]
            if len(history) != hourly.WINDOW:
                raise ValueError("Incomplete 1200-bar hourly history")
            quote = hbar[i + 1]
            if quote["time"] != hbar[i]["time"] + HOUR:
                raise ValueError("Cannot replay across hourly data gap")
            result = hourly.step(state, history, quote["open"], quote["time"])
            if result.get("action") == "UNCHANGED":
                raise RuntimeError("Duplicate hourly decision in backtest")
            if quote["time"] % DAY == 23 * HOUR:
                active_days += int(state["qty"] > 0)
                curve.append(state["cash"] + state["qty"] * quote["close"])
        last_price = hbar[end_i - 1]["close"]
        ending = state["cash"] + state["qty"] * last_price * (
            1 - slip / 10000) * (1 - fee / 10000)
    if len(curve) != (end - first) // DAY:
        raise RuntimeError("Hourly and daily valuation calendars differ")
    return summarize(INITIAL, ending, curve, state["trades"],
                     active_days, len(curve))


def run_hold(dbar: list[dict], first: int, end: int,
             fee: float, slip: float) -> dict:
    idx = {r["time"]: i for i, r in enumerate(dbar)}
    entry = dbar[idx[first]]["open"] * (1 + slip / 10000)
    qty = INITIAL / (entry * (1 + fee / 10000))
    curve = [qty * dbar[idx[t]]["close"] for t in range(first, end, DAY)]
    end_price = dbar[idx[end - DAY]]["close"]
    ending = qty * end_price * (1 - slip / 10000) * (1 - fee / 10000)
    result = summarize(INITIAL, ending, curve, [], len(curve), len(curve))
    result["closed_trades"] = 0
    result["note"] = "Liquidation-cost NAV, buy-and-hold position remains open"
    return result


def compare(hbars: list[dict], dbars: list[dict],
            fees: tuple[float, ...] = FEES_BPS, slip: float = SLIPPAGE_BPS) -> dict:
    validate(hbars, HOUR)
    validate(dbars, DAY)
    if not fees or not all(math.isfinite(f) and f >= 0 for f in fees):
        raise ValueError("Invalid fees")
    if not math.isfinite(slip) or slip < 0:
        raise ValueError("Invalid slippage")
    if hbars[0]["time"] != epoch(START_HOUR) or hbars[-1]["time"] != epoch(END) - HOUR:
        raise ValueError("Expected complete hourly download range boundaries")
    if dbars[0]["time"] != epoch(START_DAY) or dbars[-1]["time"] != epoch(END) - DAY:
        raise ValueError("Expected complete daily download range boundaries")
    periods = windows(hbars, dbars)
    output = []
    for w in periods:
        tests = []
        for fee in fees:
            tests.append({
                "fee_bps_per_side": fee,
                "slippage_bps_per_side": slip,
                "strategies": {
                    "SMA200_DAILY_50PCT": run_sma(dbars, w["start"], w["end"], fee, slip),
                    "DONCHIAN55_HOURLY": run_hourly(hbars, w["start"], w["end"], fee, slip),
                    "BUY_HOLD_100PCT": run_hold(dbars, w["start"], w["end"], fee, slip),
                    "CASH_100PCT": summarize(INITIAL, INITIAL, [], [], 0, 1),
                },
            })
        output.append({"period": w["label"],
                       "start": stamp(w["start"]),
                       "end_exclusive": stamp(w["end"]),
                       "days": (w["end"] - w["start"]) // DAY,
                       "new_paper_account": True,
                       "cost_scenarios": tests})
    hs = missing_intervals(hbars, HOUR)
    ds = missing_intervals(dbars, DAY)
    return {
        "dataset": "Coinbase Exchange BTC-USD spot",
        "history_hours": len(hbars),
        "history_days": len(dbars),
        "missing_hour_intervals": hs,
        "missing_daily_intervals": ds,
        "hourly_input_sha256": hashlib.sha256(
            json.dumps(hbars, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "daily_input_sha256": hashlib.sha256(
            json.dumps(dbars, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "capital_usd_each_independent_account": INITIAL,
        "capital_reset_at_start_of_each_period": True,
        "valuation": "Daily close NAV, open BTC marked to liquidation net of fee/slippage at end",
        "execution_proxy": "Signal on closed previous candle; execute at next daily/hourly OPEN",
        "holding_policy": "SMA200 actual shadow.step at 50% cash budget per BUY, no rebalancing",
        "exclusions": [
            "Missing hourly candles excluded with new >=1200-bar warmup",
            "Incomplete UTC days around gaps excluded from all comparisons",
            "Hourly intrabar stop polling in the live 15-minute job cannot be reconstructed",
            "Fee tiers and slippage are simulated, not actual exchange fills",
            "This is a historical descriptive comparison, not fresh unseen out-of-sample proof",
        ],
        "periods": output,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Coinbase Spot: Fair SMA200 / Hourly / Buy & Hold / Cash Comparison",
        "",
        "Paper-only historical replay, not a promise of profit.",
        "Each row is a separate $10,000 portfolio. Do not add returns across segments.",
        "Drawdowns use daily-close observations for every strategy.",
        "",
        "| Period (UTC) | Days | Fee bps/side | Strategy | Net return % | Max daily DD % | Closed trades | Profit factor |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for seg in report["periods"]:
        for scenario in seg["cost_scenarios"]:
            for label, metrics in scenario["strategies"].items():
                lines.append("| " + " | ".join(
                    [seg["start"][:10] + " to " + seg["end_exclusive"][:10],
                     str(seg["days"]), str(scenario["fee_bps_per_side"]),
                     label, str(metrics["net_return_pct"]),
                     str(metrics["max_daily_drawdown_pct"]),
                     str(metrics["closed_trades"]), str(metrics["profit_factor"] or "n/a")]) + " |")
    lines.extend(["", "## Data integrity",
                  "Hourly gaps: " + json.dumps(report["missing_hour_intervals"]),
                  "Daily gaps: " + json.dumps(report["missing_daily_intervals"]),
                  "Hourly data SHA256: " + report["hourly_input_sha256"],
                  "Daily data SHA256: " + report["daily_input_sha256"],
                  "",
                  "## Limitations",
                  *("- " + x for x in report["exclusions"])])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch-coinbase", action="store_true")
    group.add_argument("--csv-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("research-output/comparison"))
    parser.add_argument("--fees", type=float, nargs="+", default=list(FEES_BPS))
    parser.add_argument("--slippage-bps", type=float, default=SLIPPAGE_BPS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fetch_coinbase:
        hbars = fetch(epoch(START_HOUR), epoch(END), HOUR)
        dbars = fetch(epoch(START_DAY), epoch(END), DAY)
    else:
        hbars = read_csv(args.csv_dir / "coinbase_hourly.csv", HOUR)
        dbars = read_csv(args.csv_dir / "coinbase_daily.csv", DAY)
    result = compare(hbars, dbars, tuple(args.fees), args.slippage_bps)
    write_csv(args.output_dir / "coinbase_hourly.csv", hbars)
    write_csv(args.output_dir / "coinbase_daily.csv", dbars)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "comparison.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result))


if __name__ == "__main__":
    main()
