from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.contract_training import _contiguous_segments


class TrainingSegmentationTests(unittest.TestCase):
    def test_real_gap_creates_two_segments_without_fill(self) -> None:
        first = pd.date_range(
            "2024-01-01T00:00:00Z",
            periods=5,
            freq="h",
        )
        second = pd.date_range(
            "2024-01-01T20:00:00Z",
            periods=5,
            freq="h",
        )
        times = first.append(second)
        candles = pd.DataFrame(
            {
                "open_time": times,
                "open": range(10),
                "high": range(1, 11),
                "low": range(10),
                "close": range(10),
                "volume": [1.0] * 10,
            }
        )

        segments, continuity = _contiguous_segments(candles)

        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0]["frame"]), 5)
        self.assertEqual(len(segments[1]["frame"]), 5)
        self.assertEqual(continuity["gap_count"], 1)
        self.assertEqual(continuity["largest_gap_hours"], 16.0)
        self.assertEqual(continuity["missing_candle_hours"], 15)
        self.assertEqual(segments[1]["gap_before_hours"], 16.0)
        reconstructed = pd.concat(
            [item["frame"] for item in segments],
            ignore_index=True,
        )
        self.assertEqual(
            reconstructed["open_time"].tolist(),
            list(times),
        )

    def test_continuous_history_remains_one_segment(self) -> None:
        times = pd.date_range(
            "2024-01-01T00:00:00Z",
            periods=24,
            freq="h",
        )
        candles = pd.DataFrame({"open_time": times})

        segments, continuity = _contiguous_segments(candles)

        self.assertEqual(len(segments), 1)
        self.assertEqual(continuity["gap_count"], 0)
        self.assertEqual(continuity["largest_gap_hours"], 0.0)
        self.assertEqual(continuity["missing_candle_hours"], 0)


if __name__ == "__main__":
    unittest.main()