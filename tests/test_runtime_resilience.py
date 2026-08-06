from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.runtime_history import (
    expected_latest_closed_open,
    latest_candle_freshness,
    latest_contiguous_tail,
)


class RuntimeResilienceTests(unittest.TestCase):
    def test_latest_contiguous_tail_discards_pre_gap_history(self) -> None:
        times = pd.to_datetime(
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T05:00:00Z",
                "2026-01-01T06:00:00Z",
                "2026-01-01T07:00:00Z",
            ],
            utc=True,
        )
        frame = pd.DataFrame({"open_time": times, "close": range(5)})
        selected, audit = latest_contiguous_tail(frame)
        self.assertEqual(len(selected), 3)
        self.assertEqual(
            selected["open_time"].iloc[0],
            pd.Timestamp("2026-01-01T05:00:00Z"),
        )
        self.assertEqual(audit["missing_candle_hours"], 3)
        self.assertFalse(audit["synthetic_candles"])
        self.assertFalse(audit["interpolation"])

    def test_expected_latest_closed_candle_uses_completed_hour(self) -> None:
        expected = expected_latest_closed_open(
            pd.Timestamp("2026-01-01T12:17:00Z")
        )
        self.assertEqual(expected, pd.Timestamp("2026-01-01T11:00:00Z"))

    def test_freshness_accepts_current_closed_candle(self) -> None:
        frame = pd.DataFrame(
            {"open_time": [pd.Timestamp("2026-01-01T11:00:00Z")]}
        )
        audit = latest_candle_freshness(
            frame,
            now=pd.Timestamp("2026-01-01T12:17:00Z"),
            maximum_lag_hours=1.0,
        )
        self.assertTrue(audit["fresh"])
        self.assertEqual(audit["lag_hours"], 0.0)

    def test_freshness_rejects_multi_hour_stale_tail(self) -> None:
        frame = pd.DataFrame(
            {"open_time": [pd.Timestamp("2026-01-01T08:00:00Z")]}
        )
        audit = latest_candle_freshness(
            frame,
            now=pd.Timestamp("2026-01-01T12:17:00Z"),
            maximum_lag_hours=1.0,
        )
        self.assertFalse(audit["fresh"])
        self.assertEqual(audit["lag_hours"], 3.0)


if __name__ == "__main__":
    unittest.main()
