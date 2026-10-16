"""Reproducible chronological 1h BTC-USD spot replay of the regime shadow."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from regime_shadow import FEE_BPS, HOUR, INITIAL_CASH, SLIPPAGE_BPS, new_state, step


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: int(row[key]) if key == "time" else float(row[key])
             for key in ("time", "open", "high", "low", "close", "volume")}
            for row in csv.DictReader(stream)
        ]
    if len(rows) < 500:
        raise ValueError("At least 500 completed hourly bars required")
    if any(b["time"] - a["time"] != HOUR for a, b in zip(rows, rows[1:])):
        raise ValueError("Input must be UTC hourly and gap-free")
    return rows


def replay(rows: list[dict], start: int) -> dict:
    if start < 100 or start >= len(rows) - 1:
        raise ValueError("Invalid evaluation start")
    state = new_state()
    curve = [INITIAL_CASH]
    for i in range(start, len(rows) - 1):
        # Only historical candles through i are visible; execute at open[i+1].
        result = step(state, rows[max(0, i - 1199):i + 1],
                      rows[i + 1]["open"], rows[i + 1]["time"])
        curve.append(result["equity_usd"])
    last_price = rows[-1]["close"]
    liquidation_rate = (FEE_BPS + SLIPPAGE_BPS) / 10_000
    final = state["cash"] + state["qty"] * last_price * (1 - liquidation_rate)
    curve.append(final)
    peak, drawdown = INITIAL_CASH, 0.0
    for equity in curve:
        peak = max(peak, equity)
        drawdown = max(drawdown, 1 - equity / peak)
    pnl = [trade["pnl_usd"] for trade in state["trades"]]
    profit = sum(value for value in pnl if value > 0)
    loss = -sum(value for value in pnl if value < 0)
    hold_net = (last_price / rows[start + 1]["open"] *
                (1 - liquidation_rate) / (1 + liquidation_rate) - 1)
    return {
        "start": rows[start]["time"], "end": rows[-1]["time"],
        "candles_evaluated": len(rows) - start - 1,
        "closed_trades": len(pnl),
        "win_rate_pct": round(100 * sum(value > 0 for value in pnl) / len(pnl), 3)
                        if pnl else None,
        "profit_factor": round(profit / loss, 3) if loss else None,
        "net_return_pct": round(100 * (final / INITIAL_CASH - 1), 3),
        "buy_hold_return_pct": round(100 * hold_net, 3),
        "max_drawdown_pct": round(100 * drawdown, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="Completed Coinbase BTC-USD 1h OHLCV CSV")
    parser.add_argument("--oos-fraction", type=float, default=0.30)
    args = parser.parse_args()
    if not 0.10 <= args.oos_fraction <= 0.50:
        parser.error("--oos-fraction must be between 0.10 and 0.50")
    rows = read_rows(args.csv)
    split = max(100, int(len(rows) * (1 - args.oos_fraction)))
    report = {
        "in_sample": replay(rows[:split + 1], 100),
        "out_of_sample": replay(rows, split),
        "assumed_fee_bps_per_side": FEE_BPS,
        "assumed_slippage_bps_per_side": SLIPPAGE_BPS,
        "method": "Closed 1h candles and next-hour open, same untouched rules",
        "caveat": "Exploratory historical paper replay; future profitability unverified",
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
