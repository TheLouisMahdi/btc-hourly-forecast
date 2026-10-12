"""Tests for the independent, closed-candle SMA200 shadow."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sma200_shadow as shadow


def bars(n=230, price=100.0, start=1704067200):
    return [{"time": start + i * shadow.DAY, "open": price, "high": price + 2,
             "low": price - 2, "close": price, "volume": 10.0} for i in range(n)]


class SMA200Tests(unittest.TestCase):
    def test_200_day_signal_and_first_paper_buy(self):
        candles = bars()
        candles[-1]["close"] = 103.0
        state = shadow.new_state()
        result = shadow.step(state, candles, 101.0, candles[-1]["time"] + shadow.DAY)
        self.assertEqual(result["action"], "BUY")
        self.assertEqual(result["signal"], "LONG")
        self.assertAlmostEqual(result["sma200"], 100.015)
        self.assertLessEqual(state["qty"] * 101.0, 5000.0)
        self.assertGreater(state["cash"], 0)

    def test_repeated_daily_candle_is_idempotent(self):
        candles = bars()
        candles[-1]["close"] = 103
        state = shadow.new_state()
        shadow.step(state, candles, 101, candles[-1]["time"] + shadow.DAY)
        cash, qty = state["cash"], state["qty"]
        self.assertIsNone(shadow.step(state, candles, 102, candles[-1]["time"] + shadow.DAY + 900))
        self.assertEqual((state["cash"], state["qty"]), (cash, qty))
        self.assertEqual(len(state["history"]), 1)

    def test_close_below_sma_exits_and_charges_both_fees(self):
        candles = bars()
        candles[-1]["close"] = 103
        state = shadow.new_state()
        shadow.step(state, candles, 101, candles[-1]["time"] + shadow.DAY)
        nxt = bars()
        for i, b in enumerate(nxt):
            b["time"] += shadow.DAY
        nxt[-1]["close"] = 97
        result = shadow.step(state, nxt, 100, nxt[-1]["time"] + shadow.DAY)
        self.assertEqual(result["action"], "SELL")
        self.assertEqual(state["qty"], 0)
        self.assertLess(state["cash"], shadow.INITIAL_CASH)
        self.assertLess(state["trades"][0]["pnl_usd"], 0)
        self.assertEqual(len(state["trades"]), 1)

    def test_cash_when_signal_is_below(self):
        c = bars()
        c[-1]["close"] = 97
        st = shadow.new_state()
        result = shadow.step(st, c, 100, c[-1]["time"] + shadow.DAY)
        self.assertEqual(result["action"], "CASH")
        self.assertEqual(st["qty"], 0)

    def test_missing_days_are_not_replayed(self):
        c = bars()
        st = shadow.new_state()
        st["last_candle"] = c[-1]["time"] - 3 * shadow.DAY
        c[-1]["close"] = 103
        res = shadow.step(st, c, 100, c[-1]["time"] + shadow.DAY)
        self.assertEqual(res["missed_days"], 2)
        self.assertEqual(len(st["history"]), 1)

    def test_fetch_excludes_open_daily_candle_and_rejects_gaps(self):
        now = bars()[-1]["time"] + shadow.DAY + 600
        historical = bars(240, start=now // shadow.DAY * shadow.DAY - 240 * shadow.DAY)
        payload = [[c["time"], c["low"], c["high"], c["open"], c["close"], c["volume"]]
                   for c in historical]
        payload += [[now // shadow.DAY * shadow.DAY, 1, 2, 1.5, 1.5, 1]]
        with patch.object(shadow, "request_json", return_value=payload):
            actual = shadow.daily_bars(now)
        self.assertEqual(actual[-1]["time"], now // shadow.DAY * shadow.DAY - shadow.DAY)
        self.assertEqual(len(actual), 240)
        payload.pop(-10)
        with patch.object(shadow, "request_json", return_value=payload):
            with self.assertRaisesRegex(ValueError, "Discontinuous"):
                shadow.daily_bars(now)

    def test_stale_ticker_refused(self):
        now = 1720000000
        with patch.object(shadow, "request_json", return_value={
            "price": "100", "time": datetime.fromtimestamp(now - 400, timezone.utc).isoformat()
        }):
            with self.assertRaisesRegex(ValueError, "stale"):
                shadow.fresh_quote(now)


if __name__ == "__main__":
    unittest.main()
