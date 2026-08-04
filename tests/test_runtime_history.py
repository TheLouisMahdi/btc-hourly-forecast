from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.runtime_history import latest_contiguous_tail


class RuntimeHistoryTests(unittest.TestCase):
    def test_latest_contiguous_tail_discards_older_gap_history(self) -> None:
        first = pd.date_range(
            "2026-01-01T00:00:00Z",
            periods=1200,
            freq="h",
        )
        second_start = first[-1] + pd.Timedelta(hours=6)
        second = pd.date_range(
            second_start,
            periods=1100,
            freq="h",
        )
        times = first.append(second)
        candles = pd.DataFrame(
            {
                "provider": ["coinbase_spot"] * len(times),
                "symbol": ["BTCUSDT"] * len(times),
                "open_time": times,
                "open": range(len(times)),
                "high": range(1, len(times) + 1),
                "low": range(len(times)),
                "close": range(len(times)),
                "volume": [1.0] * len(times),
            }
        )

        selected, audit = latest_contiguous_tail(candles)

        self.assertEqual(len(selected), 1100)
        self.assertEqual(selected["open_time"].iloc[0], second[0])
        self.assertEqual(selected["open_time"].iloc[-1], second[-1])
        self.assertEqual(audit["gap_count"], 1)
        self.assertEqual(audit["largest_gap_hours"], 6.0)
        self.assertEqual(audit["missing_candle_hours"], 5)
        self.assertEqual(audit["discarded_older_rows"], 1200)
        self.assertFalse(audit["synthetic_candles"])
        self.assertFalse(audit["interpolation"])
        self.assertFalse(audit["forward_fill"])

    def test_continuous_history_is_preserved(self) -> None:
        times = pd.date_range(
            "2026-01-01T00:00:00Z",
            periods=1200,
            freq="h",
        )
        candles = pd.DataFrame({"open_time": times})

        selected, audit = latest_contiguous_tail(candles)

        self.assertEqual(len(selected), len(candles))
        self.assertEqual(audit["gap_count"], 0)
        self.assertEqual(audit["discarded_older_rows"], 0)
        self.assertEqual(audit["largest_gap_hours"], 0.0)


if __name__ == "__main__":
    unittest.main()
