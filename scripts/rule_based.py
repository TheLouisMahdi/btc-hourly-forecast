"""No-ML BTC-USD hourly spot paper trader. No broker order placement."""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PRODUCT = "BTC-USD"
BASE = "https://api.exchange.coinbase.com"
HOUR = 3600
WINDOW = 1200
FEE_BPS = 15.0  # Explicit simulation assumption, not an exchange quote.
SLIPPAGE_BPS = 3.0
INITIAL_CASH = 10_000.0
RISK_FRACTION = 0.005
MAX_EXPOSURE = 0.50
ATR_MULTIPLIER = 3.0


def utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def request_json(path: str, params: dict | None = None):
    url = BASE + path + ("?" + urlencode(params) if params else "")
    request = Request(url, headers={"User-Agent": "btc-rule-research/1.0"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def fetch_candles(now: int) -> list[dict]:
    """Read only public Coinbase data; never include an unfinished candle."""
    boundary = (now // HOUR) * HOUR
    candles: dict[int, dict] = {}
    for offset in range(0, WINDOW, 290):
        end = boundary - offset * HOUR
        start = max(0, end - 290 * HOUR)
        rows = request_json(
            f"/products/{PRODUCT}/candles",
            {"start": utc(start), "end": utc(end), "granularity": HOUR},
        )
        if not isinstance(rows, list):
            raise ValueError("Invalid candles response")
        for raw in rows:
            ts, low, high, opening, closing, volume = raw
            ts = int(ts)
            if ts + HOUR > now - 60:
                continue
            candle = {"time": ts, "open": float(opening), "high": float(high),
                      "low": float(low), "close": float(closing), "volume": float(volume)}
            if not (0 < candle["low"] <= min(candle["open"], candle["close"])
                    <= max(candle["open"], candle["close"]) <= candle["high"]
                    and all(math.isfinite(v) for k, v in candle.items() if k != "time")):
                raise ValueError("Malformed OHLC candle")
            candles[ts] = candle
    bars = [candles[k] for k in sorted(candles)]
    if len(bars) < 250 or now - (bars[-1]["time"] + HOUR) > 2 * HOUR:
        raise ValueError("Insufficient or stale candle history")
    if any(b["time"] - a["time"] != HOUR for a, b in zip(bars, bars[1:])):
        raise ValueError("Gapped candle history: no entries")
    return bars


def fetch_quote(now: int) -> float:
    quote = request_json(f"/products/{PRODUCT}/ticker")
    timestamp = datetime.fromisoformat(quote["time"].replace("Z", "+00:00")).timestamp()
    price = float(quote["price"])
    if not math.isfinite(price) or price <= 0 or abs(now - timestamp) > 180:
        raise ValueError("Invalid or stale live ticker")
    return price


def indicators(bars: list[dict], index: int) -> dict:
    """Compute only from the candle that has already closed and its predecessors."""
    if index < 200:
        raise ValueError("At least 201 closed candles required")
    alpha = 2.0 / 201.0
    ema = bars[0]["close"]
    for row in bars[1:index + 1]:
        ema = alpha * row["close"] + (1 - alpha) * ema
    tr = []
    for pos in range(index - 13, index + 1):
        bar, prev = bars[pos], bars[pos - 1]
        tr.append(max(bar["high"] - bar["low"],
                      abs(bar["high"] - prev["close"]),
                      abs(bar["low"] - prev["close"])))
    return {"ema200": ema, "atr14": sum(tr) / len(tr),
            "entry_high55": max(row["high"] for row in bars[index - 55:index]),
            "exit_low20": min(row["low"] for row in bars[index - 20:index])}


def entry_signal(bar: dict, features: dict) -> bool:
    atr_ratio = features["atr14"] / bar["close"]
    return (bar["close"] > features["entry_high55"]
            and bar["close"] > features["ema200"]
            and 0.002 <= atr_ratio <= 0.06)


def new_state() -> dict:
    return {"cash": INITIAL_CASH, "qty": 0.0, "stop": None,
            "position": None, "last_candle": None, "last_action_candle": None,
            "trades": [], "history": []}


def buy(state: dict, price: float, atr: float, candle_time: int, now: int) -> bool:
    if state["qty"] > 0 or atr <= 0:
        return False
    cost_rate = FEE_BPS / 10_000
    fill = price * (1 + SLIPPAGE_BPS / 10_000)
    risk_per_btc = max(ATR_MULTIPLIER * atr, 0.001 * price)
    size = min(state["cash"] * RISK_FRACTION / risk_per_btc,
               state["cash"] * MAX_EXPOSURE / fill,
               state["cash"] / (fill * (1 + cost_rate)))
    if size <= 0 or not math.isfinite(size):
        return False
    fee = fill * size * cost_rate
    state["cash"] -= fill * size + fee
    state["qty"] = size
    state["stop"] = fill - ATR_MULTIPLIER * atr
    state["position"] = {"entry_time": now, "entry_candle": candle_time,
                         "entry_price": fill, "quantity": size,
                         "entry_fee": fee, "entry_stop": state["stop"]}
    state["last_action_candle"] = candle_time
    return True


def sell(state: dict, price: float, reason: str, now: int, candle_time: int) -> dict:
    if state["qty"] <= 0:
        raise ValueError("No active position")
    fill = price * (1 - SLIPPAGE_BPS / 10_000)
    fee = state["qty"] * fill * FEE_BPS / 10_000
    state["cash"] += state["qty"] * fill - fee
    position = state["position"]
    pnl = (fill - position["entry_price"]) * state["qty"] - position["entry_fee"] - fee
    trade = {**position, "exit_price": fill, "exit_fee": fee,
             "exit_time": now, "exit_candle": candle_time,
             "pnl_usd": pnl, "reason": reason, "status": "CLOSED"}
    state["trades"].append(trade)
    state["qty"] = 0.0
    state["stop"] = None
    state["position"] = None
    state["last_action_candle"] = candle_time
    return trade


def step(state: dict, bars: list[dict], quote: float, now: int) -> dict:
    """Idempotent on the latest closed hour; no retrospective fills on skipped bars."""
    index = len(bars) - 1
    bar = bars[index]
    candle_time = bar["time"]
    if state.get("last_candle") is not None and candle_time <= state["last_candle"]:
        return {"action": "UNCHANGED", "reason": "ALREADY_PROCESSED"}
    if state.get("last_candle") is not None and candle_time - state["last_candle"] > HOUR:
        return {"action": "FAIL_SAFE", "reason": "MISSED_CANDLE_MANUAL_REVIEW"}
    f = indicators(bars, index)
    action, reason = "FLAT", "NO_BREAKOUT"
    if state["qty"] > 0:
        action, reason = "HOLD", "OPEN_POSITION"
        # A full candle must start after entry; no use of earlier intrabar lows.
        if candle_time >= state["position"]["entry_time"]:
            if bar["low"] <= state["stop"]:
                sell(state, min(bar["open"], state["stop"]), "STOP", now, candle_time)
                action, reason = "EXIT", "STOP"
            elif bar["close"] < f["exit_low20"] or bar["close"] < f["ema200"]:
                sell(state, quote, "TREND_EXIT", now, candle_time)
                action, reason = "EXIT", "TREND_EXIT"
            else:
                state["stop"] = max(state["stop"], bar["close"] - ATR_MULTIPLIER * f["atr14"])
    elif entry_signal(bar, f) and state.get("last_action_candle") != candle_time:
        if buy(state, quote, f["atr14"], candle_time, now):
            action, reason = "BUY", "BREAKOUT_55"
    state["last_candle"] = candle_time
    equity = state["cash"] + state["qty"] * quote
    result = {"time": utc(now), "candle_time": utc(candle_time),
              "run_status": "OK", "action": action, "reason": reason,
              "strategy": "DONCHIAN55_EMA200_ATR14_SPOT_V1", "price": quote,
              "cash_usd": state["cash"], "equity_usd": equity,
              "btc_quantity": state["qty"], "stop_price": state["stop"],
              "features": f, "realized_pnl_usd": sum(t["pnl_usd"] for t in state["trades"]),
              "completed_trades": len(state["trades"]), "mode": "PAPER_ONLY"}
    state["history"] = (state.get("history", []) + [result])[-720:]
    return result


def load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def save(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", default=".rule_state")
    args = parser.parse_args()
    directory = Path(args.state_dir)
    state = load(directory / "state.json", new_state())
    now = int(time.time())
    try:
        bars = fetch_candles(now)
        quote = fetch_quote(now)
        result = step(state, bars, quote, now)
        if result["action"] == "FAIL_SAFE":
            raise RuntimeError(result["reason"])
        save(directory / "state.json", state)
        save(directory / "candles.json", bars[-168:])
        save(directory / "trades.json", state["trades"])
        save(directory / "history.json", state["history"])
        save(directory / "latest.json", result)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        # No position/equity changes or synthetic fills when data is unhealthy.
        error = {"run_status": "FAIL_SAFE", "action": "NO_TRADE",
                 "reason": f"{type(exc).__name__}: {exc}",
                 "time": utc(now), "mode": "PAPER_ONLY"}
        save(directory / "latest.json", error)
        print(json.dumps(error, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
