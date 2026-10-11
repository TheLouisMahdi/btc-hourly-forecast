"""Chronological, next-open paper backtest of the exact rule engine."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rule_based import HOUR, INITIAL_CASH, new_state, step


def read_candles(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [{key: (int(value) if key == "time" else float(value))
                 for key, value in row.items() if key in
                 {"time", "open", "high", "low", "close", "volume"}}
                for row in csv.DictReader(stream)]
    if any(b["time"] - a["time"] != HOUR for a, b in zip(rows, rows[1:])):
        raise ValueError("Input must be hourly, UTC, sorted and gap-free")
    if len(rows) < 400:
        raise ValueError("Minimum 400 candles for evaluation")
    return rows


def replay(rows: list[dict], start: int = 200) -> dict:
    if start < 200 or start >= len(rows) - 1:
        raise ValueError("Invalid evaluation start")
    state = new_state()
    curves = []
    for i in range(start, len(rows) - 1):
        # Signal observes candle i after close. Fill uses next candle's OPEN.
        result = step(state, rows[max(0, i - 600):i + 1],
                      rows[i + 1]["open"], rows[i]["time"] + HOUR)
        if result["action"] == "FAIL_SAFE":
            raise RuntimeError(result["reason"])
        curves.append(result["equity_usd"])
    last_close = rows[-1]["close"]
    net = state["cash"] + state["qty"] * last_close * (1 - 0.0018)
    peak = INITIAL_CASH
    max_dd = 0.0
    for equity in curves + [net]:
        peak = max(peak, equity)
        max_dd = max(max_dd, 1 - equity / peak)
    first_buy = rows[start + 1]["open"]
    return {"start_time": rows[start]["time"], "end_time": rows[-1]["time"],
            "number_of_candles": len(rows) - start, "closed_trades": len(state["trades"]),
            "net_equity_usd": round(net, 2),
            "net_return_pct": round(100 * (net / INITIAL_CASH - 1), 3),
            "buy_hold_return_pct": round(100 * (last_close / first_buy - 1 - 0.0036), 3),
            "max_drawdown_pct": round(100 * max_dd, 3),
            "note": "Paper simulation; fees and slippage assumed; not verified live profit"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="UTC hourly Coinbase OHLCV with time epoch seconds")
    parser.add_argument("--out-of-sample-fraction", type=float, default=0.30)
    args = parser.parse_args()
    if not 0 < args.out_of_sample_fraction <= 0.50:
        parser.error("OOS fraction must be 0..0.50")
    rows = read_candles(args.csv)
    split = max(200, int(len(rows) * (1 - args.out_of_sample_fraction)))
    report = {"in_sample": replay(rows[:split + 1]),
              "out_of_sample": replay(rows, start=split),
              "data_separation": "chronological; no parameter optimization"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
