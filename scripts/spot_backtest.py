"""Reproducible Coinbase BTC-USD spot backtest using the live paper rule engine.

Research only. Never places orders or changes the running paper-trade ledger.
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

import rule_based as engine

FIELDS = ("time", "open", "high", "low", "close", "volume")
RESEARCH_WINDOW = engine.WINDOW
DEFAULT_FEES = (15.0, 40.0, 60.0)


def timestamp(value: str) -> int:
    return int(datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def validate(rows: list[dict], start: int | None = None, end: int | None = None) -> None:
    if len(rows) < RESEARCH_WINDOW + 2:
        raise ValueError("At least 1,202 consecutive hourly candles are required")
    for i, row in enumerate(rows):
        t = row["time"]
        values = [row[field] for field in FIELDS[1:]]
        if t % engine.HOUR or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Invalid timestamp or nonfinite price at row {i}")
        if (min(row["open"], row["close"]) < row["low"]
                or max(row["open"], row["close"]) > row["high"]
                or row["low"] <= 0 or row["volume"] < 0):
            raise ValueError(f"Invalid OHLCV at row {i}")
        if i and t - rows[i - 1]["time"] != engine.HOUR:
            raise ValueError(f"Missing or duplicate hourly candle at {engine.utc(t)}")
    if start is not None and rows[0]["time"] != start:
        raise ValueError("History begins after requested start: refuse partial sample")
    if end is not None and rows[-1]["time"] + engine.HOUR != end:
        raise ValueError("History ends before requested end: refuse partial sample")


def download_coinbase(start: int, end: int) -> list[dict]:
    """Fetch 288-hour pages (below Coinbase's 300-bucket limit)."""
    if start % engine.HOUR or end % engine.HOUR or start >= end:
        raise ValueError("UTC start and end must be increasing hour boundaries")
    indexed = {}
    for first in range(start, end, 288 * engine.HOUR):
        last = min(first + 288 * engine.HOUR, end)
        for attempt in range(5):
            try:
                payload = engine.request_json(
                    f"/products/{engine.PRODUCT}/candles",
                    {"start": engine.utc(first), "end": engine.utc(last),
                     "granularity": engine.HOUR},
                )
                break
            except (HTTPError, URLError, TimeoutError) as exc:
                if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                    raise
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))
        if not isinstance(payload, list):
            raise ValueError("Invalid Coinbase OHLCV response")
        for raw in payload:
            if not isinstance(raw, list) or len(raw) != 6:
                raise ValueError("Unexpected Coinbase candle structure")
            t, low, high, opened, closed, volume = raw
            t = int(t)
            if not first <= t < last:
                continue
            candle = {"time": t, "open": float(opened), "high": float(high),
                      "low": float(low), "close": float(closed),
                      "volume": float(volume)}
            if t in indexed and indexed[t] != candle:
                raise ValueError("Conflicting duplicate candle from Coinbase")
            indexed[t] = candle
        time.sleep(0.25)
    rows = [indexed[t] for t in sorted(indexed)]
    validate(rows, start=start, end=end)
    return rows


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError("CSV requires time, open, high, low, close, volume")
        rows = [{key: int(row[key]) if key == "time" else float(row[key])
                 for key in FIELDS} for row in reader]
    validate(rows)
    return rows


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def replay(rows: list[dict], start: int, end: int, fee_bps: float,
           slippage_bps: float) -> dict:
    """Use production step() unmodified; one signal on closed bar, next-open fill.

    Paper stop monitoring between candle closes cannot be reconstructed from 1h bars.
    This replay does not assert historical fills at exact stop or quote timestamps.
    """
    if not (RESEARCH_WINDOW - 1 <= start < end - 1 <= len(rows) - 1):
        raise ValueError("Invalid evaluation range or inadequate warmup")
    if (not math.isfinite(fee_bps) or not math.isfinite(slippage_bps)
            or fee_bps < 0 or slippage_bps < 0):
        raise ValueError("Costs must be nonnegative and finite")
    state = engine.new_state()
    initial_cash = state["cash"]
    equity_curve = [initial_cash]
    max_drawdown = 0.0
    peak = initial_cash
    active_bars = 0
    executions = 0
    with (patch.object(engine, "FEE_BPS", fee_bps),
          patch.object(engine, "SLIPPAGE_BPS", slippage_bps)):
        for i in range(start, end - 1):
            current = rows[i]
            next_bar = rows[i + 1]
            # Live engine has at most 1,200 historical candles.
            prior = rows[i - RESEARCH_WINDOW + 1:i + 1]
            result = engine.step(state, prior, next_bar["open"], next_bar["time"])
            if result["action"] == "FAIL_SAFE":
                raise RuntimeError(result["reason"])
            if result["action"] in {"BUY", "EXIT"}:
                executions += 1
            active_bars += int(state["qty"] > 0)
            value = result["equity_usd"]
            peak = max(peak, value)
            max_drawdown = max(max_drawdown, 1 - value / peak)
            equity_curve.append(value)
        last_close = rows[end - 1]["close"]
        liquidation = last_close * (1 - slippage_bps / 10_000)
        final_equity = (state["cash"]
                        + state["qty"] * liquidation * (1 - fee_bps / 10_000))
    peak = max(peak, final_equity)
    max_drawdown = max(max_drawdown, 1 - final_equity / peak)
    completed = state["trades"]
    winners = [t["pnl_usd"] for t in completed if t["pnl_usd"] > 0]
    losers = [t["pnl_usd"] for t in completed if t["pnl_usd"] <= 0]
    gains, losses = sum(winners), -sum(losers)
    entry = rows[start + 1]["open"] * (1 + (fee_bps + slippage_bps) / 10_000)
    exit_price = last_close * (1 - (fee_bps + slippage_bps) / 10_000)
    benchmark = exit_price / entry - 1
    trades_count = len(completed)
    return {
        "start": engine.utc(rows[start]["time"]),
        "end": engine.utc(rows[end - 1]["time"]),
        "fee_bps_per_side": fee_bps,
        "slippage_bps_per_side": slippage_bps,
        "bars_evaluated": end - start - 1,
        "open_trade": state["qty"] > 0,
        "closed_trades": trades_count,
        "executions": executions,
        "win_rate_pct": round(100 * len(winners) / trades_count, 2)
                        if trades_count else None,
        "expectancy_usd_per_closed_trade": round(sum(t["pnl_usd"] for t in completed)
                                                  / trades_count, 2)
                                            if trades_count else None,
        "profit_factor": round(gains / losses, 3) if losses else None,
        "equity_usd": round(final_equity, 2),
        "net_return_pct": round(100 * (final_equity / initial_cash - 1), 3),
        "buy_hold_return_pct": round(100 * benchmark, 3),
        "max_drawdown_pct": round(100 * max_drawdown, 3),
        "time_in_market_pct": round(100 * active_bars / (end - start - 1), 2),
    }


def report(rows: list[dict], fees: tuple[float, ...] = DEFAULT_FEES,
           slippage_bps: float = 3.0) -> dict:
    validate(rows)
    first = RESEARCH_WINDOW - 1
    split = next((i for i, b in enumerate(rows)
                  if b["time"] >= timestamp("2025-01-01")), None)
    if split is None or split <= first or len(rows) - split < 2:
        raise ValueError("Both 2024 and 2025 are required for fixed year holdout")
    if rows[0]["time"] > timestamp("2024-01-01") or rows[-1]["time"] < timestamp("2025-12-31"):
        raise ValueError("Must cover 2024 and 2025 without missing hours")
    groups = (("2024", first, split), ("2025_holdout", split, len(rows)),
              ("full", first, len(rows)))
    return {
        "source": "Coinbase Exchange BTC-USD 1h spot" ,
        "data_start": engine.utc(rows[0]["time"]),
        "data_end": engine.utc(rows[-1]["time"]),
        "sha256_canonical_json": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "strategy": "Exact rule_based.step; 1200-bar causal warmup, next-hour open",
        "cost_model": "Taker on entry and exit; fixed slippage per side",
        "limitations": [
            "Hourly OHLCV has no bid-ask, depth, intra-hour stop timing or true fills",
            "No retroactive re-entry or parameter fitting; 2025 is a fixed holdout",
            "Exchange live paper may fill at a later quote than next open",
            "The reported expectancy covers CLOSED trades only; open position is marked to liquidation",
        ],
        "scenarios": {
            name: [replay(rows, start, stop, fee, slippage_bps) for fee in fees]
            for name, start, stop in groups
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fetch-coinbase", action="store_true")
    group.add_argument("--csv", type=Path)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive UTC end date")
    parser.add_argument("--save-csv", type=Path)
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/spot_backtest.json"))
    parser.add_argument("--fees", type=float, nargs="+", default=list(DEFAULT_FEES),
                        help="Per-side taker fee in bps")
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    args = parser.parse_args()
    rows = (download_coinbase(timestamp(args.start), timestamp(args.end))
            if args.fetch_coinbase else load_csv(args.csv))
    if args.save_csv:
        save_csv(args.save_csv, rows)
    result = report(rows, tuple(args.fees), args.slippage_bps)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
