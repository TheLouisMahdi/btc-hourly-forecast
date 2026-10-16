"""Independent BTC-USD spot paper research: 4h RSI regime, 1h entries.

Uses closed Coinbase candles only. Never places broker orders.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from rule_based import (
    FEE_BPS, HOUR, INITIAL_CASH, SLIPPAGE_BPS,
    fetch_candles, fetch_quote, load, save, utc,
)

FOUR_HOURS = 4 * HOUR
RSI_PERIOD = 20
RSI_THRESHOLD = 70.0
ENTRY_LOOKBACK = 12
PULLBACK = 0.0075
BREAKOUT = 0.0025
SOFT_STOP = 0.0125
HARD_STOP = 0.025
MAX_HOLD_HOURS = 24
MAX_ZONE_ENTRIES = 3
RISK_FRACTION = 0.005
MAX_ALLOCATION = 0.50
STRATEGY = "RSI4H_PULLBACK_BREAKOUT1H_SPOT_V1"


def completed_four_hour_closes(bars: list[dict]) -> list[tuple[int, float]]:
    """Aggregate whole UTC-aligned four-hour blocks, discarding partial blocks."""
    first = bars[0]["time"]
    offset = (FOUR_HOURS - first % FOUR_HOURS) % FOUR_HOURS
    closes = []
    for i in range(offset // HOUR, len(bars) - 3, 4):
        group = bars[i:i + 4]
        if group[0]["time"] % FOUR_HOURS == 0 and all(
            group[j]["time"] == group[0]["time"] + j * HOUR for j in range(4)
        ):
            closes.append((group[0]["time"] + FOUR_HOURS, group[-1]["close"]))
    return closes


def wilder_rsi(closes: list[float], period: int = RSI_PERIOD) -> float:
    if len(closes) < period + 1:
        raise ValueError("Insufficient closed 4h bars for RSI")
    changes = [current - previous for previous, current in zip(closes, closes[1:])]
    gain = sum(max(change, 0) for change in changes[:period]) / period
    loss = sum(max(-change, 0) for change in changes[:period]) / period
    for change in changes[period:]:
        gain = ((period - 1) * gain + max(change, 0)) / period
        loss = ((period - 1) * loss + max(-change, 0)) / period
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return 100.0 - 100.0 / (1 + gain / loss)


def features(bars: list[dict]) -> dict:
    if len(bars) < 100:
        raise ValueError("At least 100 closed hourly candles required")
    if any(b["time"] - a["time"] != HOUR for a, b in zip(bars, bars[1:])):
        raise ValueError("Discontinuous hourly candles")
    if any(
        not all(math.isfinite(float(b[k])) for k in ("open", "high", "low", "close"))
        or b["low"] <= 0 or b["low"] > min(b["open"], b["close"])
        or max(b["open"], b["close"]) > b["high"]
        for b in bars
    ):
        raise ValueError("Malformed candles")
    four_hour = completed_four_hour_closes(bars)
    rsi = wilder_rsi([close for _, close in four_hour])
    prior_high = max(bar["high"] for bar in bars[-ENTRY_LOOKBACK - 1:-1])
    current, previous = bars[-1], bars[-2]
    breakout = current["close"] > prior_high * (1 + BREAKOUT)
    pullback = (
        previous["close"] <= prior_high * (1 - PULLBACK)
        and current["close"] > previous["close"]
        and current["close"] > current["open"]
        and current["close"] < prior_high
    )
    return {
        "rsi4h": round(rsi, 5),
        "completed_4h_end": utc(four_hour[-1][0]),
        "prior_high12": prior_high,
        "entry_setup": "BREAKOUT" if breakout else "PULLBACK" if pullback else None,
        "regime_long": rsi > RSI_THRESHOLD,
    }


def new_state() -> dict:
    return {
        "cash": INITIAL_CASH, "qty": 0.0, "position": None,
        "last_candle": None, "regime_active": False, "zone_entries": 0,
        "trades": [], "history": [],
    }


def buy(state: dict, quote: float, candle_time: int, now: int, setup: str) -> bool:
    if state["qty"] > 0:
        return False
    fill = quote * (1 + SLIPPAGE_BPS / 10_000)
    fee_rate = FEE_BPS / 10_000
    size = min(
        state["cash"] * RISK_FRACTION / (fill * HARD_STOP),
        state["cash"] * MAX_ALLOCATION / (fill * (1 + fee_rate)),
    )
    if not math.isfinite(size) or size <= 0:
        return False
    fee = fill * size * fee_rate
    state["cash"] -= size * fill + fee
    state["qty"] = size
    state["position"] = {
        "entry_time": now, "entry_candle": candle_time,
        "entry_price": fill, "quantity": size, "entry_fee": fee,
        "hard_stop": fill * (1 - HARD_STOP),
        "soft_stop": fill * (1 - SOFT_STOP), "setup": setup,
    }
    state["zone_entries"] += 1
    return True


def sell(state: dict, price: float, reason: str, now: int, candle_time: int) -> None:
    pos = state["position"]
    fill = price * (1 - SLIPPAGE_BPS / 10_000)
    fee = state["qty"] * fill * FEE_BPS / 10_000
    pnl = (fill - pos["entry_price"]) * state["qty"] - pos["entry_fee"] - fee
    state["cash"] += state["qty"] * fill - fee
    state["trades"].append({
        **pos, "exit_time": now, "exit_candle": candle_time,
        "exit_price": fill, "exit_fee": fee, "pnl_usd": pnl,
        "reason": reason, "status": "CLOSED",
    })
    state["qty"], state["position"] = 0.0, None


def step(state: dict, bars: list[dict], quote: float, now: int) -> dict:
    """Decide on a closed hour; subsequent polls only check the quote hard stop."""
    if not math.isfinite(quote) or quote <= 0:
        raise ValueError("Invalid quote")
    if not bars or bars[-1]["time"] + HOUR > now:
        raise ValueError("Unfinished hourly candle")
    candle_time = bars[-1]["time"]
    last = state["last_candle"]
    if last is not None and candle_time <= last:
        if state["qty"] == 0 or quote > state["position"]["hard_stop"]:
            return {"action": "UNCHANGED", "reason": "ALREADY_PROCESSED"}
        sell(state, quote, "HARD_STOP_QUOTE", now, candle_time)
        action, reason, info = "EXIT", "HARD_STOP_QUOTE", None
    else:
        info = features(bars)
        missed = last is not None and candle_time - last > HOUR
        if missed:
            if state["qty"]:
                sell(state, quote, "MISSED_CANDLE_EXIT", now, candle_time)
                action, reason = "EXIT", "MISSED_CANDLE_EXIT"
            else:
                action, reason = "RESYNC", "MISSED_CANDLES"
            state["regime_active"] = False
            state["zone_entries"] = 0
        else:
            active = info["regime_long"]
            if not active or not state["regime_active"]:
                state["zone_entries"] = 0
            state["regime_active"] = active
            action, reason = "FLAT", "NO_SETUP"
            if state["qty"]:
                pos = state["position"]
                action, reason = "HOLD", "OPEN_POSITION"
                if quote <= pos["hard_stop"]:
                    sell(state, quote, "HARD_STOP_QUOTE", now, candle_time)
                    action, reason = "EXIT", "HARD_STOP_QUOTE"
                elif candle_time >= pos["entry_time"]:
                    if bars[-1]["low"] <= pos["hard_stop"]:
                        sell(state, min(bars[-1]["open"], pos["hard_stop"]),
                             "HARD_STOP_BAR", now, candle_time)
                        action, reason = "EXIT", "HARD_STOP_BAR"
                    elif bars[-1]["close"] <= pos["soft_stop"]:
                        sell(state, quote, "SOFT_CLOSE_STOP", now, candle_time)
                        action, reason = "EXIT", "SOFT_CLOSE_STOP"
                    elif candle_time - pos["entry_candle"] >= MAX_HOLD_HOURS * HOUR:
                        sell(state, quote, "TIME_EXIT", now, candle_time)
                        action, reason = "EXIT", "TIME_EXIT"
            elif active and state["zone_entries"] < MAX_ZONE_ENTRIES:
                if info["entry_setup"] and buy(
                    state, quote, candle_time, now, info["entry_setup"]
                ):
                    action, reason = "BUY", info["entry_setup"]
                else:
                    reason = "WAIT_SETUP"
            elif active:
                reason = "ZONE_ENTRY_LIMIT"
            else:
                reason = "RSI_REGIME_OFF"
        state["last_candle"] = candle_time
    result = {
        "time": utc(now), "candle_time": utc(candle_time), "run_status": "OK",
        "strategy": STRATEGY, "action": action, "reason": reason,
        "price": quote, "cash_usd": state["cash"],
        "equity_usd": state["cash"] + state["qty"] * quote,
        "btc_quantity": state["qty"],
        "hard_stop": state["position"]["hard_stop"] if state["position"] else None,
        "features": info, "zone_entries": state["zone_entries"],
        "completed_trades": len(state["trades"]), "mode": "PAPER_ONLY",
    }
    state["history"] = (state["history"] + [result])[-720:]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=".regime_shadow")
    args = parser.parse_args()
    directory = Path(args.state_dir)
    state = load(directory / "state.json", new_state())
    now = int(time.time())
    try:
        candles = fetch_candles(now)
        quote = fetch_quote(now)
        result = step(state, candles, quote, now)
        if result["action"] == "UNCHANGED":
            print("Completed candle already processed")
            return 0
        save(directory / "state.json", state)
        save(directory / "candles.json", candles[-168:])
        save(directory / "latest.json", result)
        save(directory / "history.json", state["history"])
        save(directory / "trades.json", state["trades"])
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, IndexError) as exc:
        save(directory / "latest.json", {
            "run_status": "FAIL_SAFE", "action": "NO_TRADE",
            "reason": f"{type(exc).__name__}: {exc}",
            "time": utc(now), "mode": "PAPER_ONLY",
        })
        print(f"::warning::Regime paper research entered FAIL_SAFE: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
