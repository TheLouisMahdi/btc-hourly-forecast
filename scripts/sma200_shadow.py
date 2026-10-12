"""SMA(200) daily BTC-USD spot shadow strategy. Paper only, no broker orders.

Uses one completed UTC daily candle and a fresh execution-price proxy. Its
position ledger and capital are isolated from the hourly baseline.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from rule_based import PRODUCT, request_json, save, load, utc

DAY = 86400
PERIOD = 200
FEE_BPS = 15.0
SLIPPAGE_BPS = 3.0
INITIAL_CASH = 10_000.0
MAX_ALLOCATION = 0.50
STATE_VERSION = 1


def daily_bars(now: int) -> list[dict]:
    boundary = now // DAY * DAY
    raw = request_json(
        f"/products/{PRODUCT}/candles",
        {"start": utc(boundary - 260 * DAY), "end": utc(boundary),
         "granularity": DAY},
    )
    if not isinstance(raw, list):
        raise ValueError("Invalid daily candle response")
    indexed = {}
    for entry in raw:
        if not isinstance(entry, list) or len(entry) != 6:
            raise ValueError("Malformed daily candle")
        stamp, low, high, opening, closing, volume = entry
        stamp = int(stamp)
        if stamp % DAY or not boundary - 260 * DAY <= stamp < boundary:
            continue
        bar = {"time": stamp, "open": float(opening),
               "high": float(high), "low": float(low),
               "close": float(closing), "volume": float(volume)}
        if not (all(math.isfinite(bar[k]) for k in ("open", "high", "low", "close", "volume"))
                and 0 < bar["low"] <= min(bar["open"], bar["close"])
                <= max(bar["open"], bar["close"]) <= bar["high"]
                and bar["volume"] >= 0):
            raise ValueError("Invalid daily OHLCV")
        if stamp in indexed and bar != indexed[stamp]:
            raise ValueError("Conflicting duplicate daily candle")
        indexed[stamp] = bar
    bars = [indexed[k] for k in sorted(indexed)]
    if len(bars) < PERIOD or bars[-1]["time"] != boundary - DAY:
        raise ValueError("Missing recent completed daily candle or insufficient history")
    if any(b["time"] - a["time"] != DAY for a, b in zip(bars[-PERIOD:], bars[-PERIOD+1:])):
        raise ValueError("Discontinuous last 200 daily candles")
    return bars


def fresh_quote(now: int) -> float:
    payload = request_json(f"/products/{PRODUCT}/ticker")
    price = float(payload["price"])
    stamp = datetime.fromisoformat(payload["time"].replace("Z", "+00:00")).timestamp()
    if not math.isfinite(price) or price <= 0 or abs(now - stamp) > 180:
        raise ValueError("Invalid or stale spot quote")
    return price


def new_state() -> dict:
    return {"version": STATE_VERSION, "cash": INITIAL_CASH, "qty": 0.0,
            "position": None, "last_candle": None,
            "trades": [], "history": []}


def step(state: dict, bars: list[dict], quote: float, now: int) -> dict | None:
    if state.get("version") != STATE_VERSION:
        raise ValueError("Incompatible shadow state version")
    if len(bars) < PERIOD or not math.isfinite(quote) or quote <= 0:
        raise ValueError("Invalid SMA inputs or quote")
    last = bars[-1]
    t = last["time"]
    previous = state.get("last_candle")
    if previous is not None and t <= previous:
        return None
    sma = sum(b["close"] for b in bars[-PERIOD:]) / PERIOD
    above = last["close"] > sma
    # The daily signal is based solely on the closed candle, not the new quote.
    missing_days = max(0, (t - previous) // DAY - 1) if previous is not None else 0
    action = "HOLD" if state["qty"] else "CASH"
    fee_rate, slip_rate = FEE_BPS / 10000, SLIPPAGE_BPS / 10000
    if above and state["qty"] == 0:
        fill = quote * (1 + slip_rate)
        budget = min(state["cash"] * MAX_ALLOCATION, state["cash"] / (1 + fee_rate))
        qty = budget / fill
        if qty <= 0:
            raise ValueError("No paper cash for shadow entry")
        entry_fee = qty * fill * fee_rate
        state["cash"] -= qty * fill + entry_fee
        state["qty"] = qty
        state["position"] = {"entry_time": utc(now), "entry_candle": utc(t),
                             "entry_price": fill, "entry_fee": entry_fee,
                             "quantity": qty}
        action = "BUY"
    elif not above and state["qty"] > 0:
        pos = state["position"]
        qty = state["qty"]
        fill = quote * (1 - slip_rate)
        exit_fee = qty * fill * fee_rate
        net_pnl = (fill - pos["entry_price"]) * qty - pos["entry_fee"] - exit_fee
        state["cash"] += qty * fill - exit_fee
        state["trades"].append({**pos, "exit_time": utc(now), "exit_candle": utc(t),
                                "exit_price": fill, "exit_fee": exit_fee,
                                "pnl_usd": net_pnl, "reason": "CLOSE_BELOW_SMA200"})
        state["qty"], state["position"] = 0.0, None
        action = "SELL"
    state["last_candle"] = t
    equity = state["cash"] + state["qty"] * quote
    result = {
        "run_status": "OK", "strategy": "SMA200_DAILY_SPOT_V1",
        "mode": "PAPER_ONLY", "action": action,
        "signal": "LONG" if above else "CASH",
        "time": utc(now), "candle_time": utc(t),
        "daily_close": last["close"], "sma200": sma,
        "price": quote, "cash_usd": state["cash"],
        "equity_usd": equity, "btc_quantity": state["qty"],
        "closed_trades": len(state["trades"]),
        "realized_pnl_usd": sum(x["pnl_usd"] for x in state["trades"]),
        "missed_days": missing_days,
        "fee_bps_per_side": FEE_BPS,
        "slippage_bps_per_side": SLIPPAGE_BPS,
        "allocation_cap": MAX_ALLOCATION,
    }
    state["history"] = (state["history"] + [result])[-540:]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=".sma_shadow")
    args = parser.parse_args()
    path = Path(args.state_dir)
    now = int(time.time())
    state = load(path / "state.json", new_state())
    try:
        bars = daily_bars(now)
        quote = fresh_quote(now)
        result = step(state, bars, quote, now)
        if result is None:
            print("Daily SMA200 candle already processed. No simulated order.")
            return 0
        save(path / "state.json", state)
        save(path / "latest.json", result)
        save(path / "history.json", state["history"])
        save(path / "trades.json", state["trades"])
        save(path / "candles.json", bars[-220:])
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        # Do not mutate or replace the last known-good persisted paper ledger.
        print(f"::error::Daily shadow failed safely: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
