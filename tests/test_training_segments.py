from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from btc_ema_trader.contract_training import (
    _contiguous_segments,
    build_segmented_feature_set,
)
from btc_ema_trader.features import FeatureSet


class _Settings:
    def __init__(self, market: dict[str, object]) -> None:
        self._market = market

    def section(self, name: str) -> dict[str, object]:
        if name == "market":
            return self._market
        return {}


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

    def test_many_bounded_gaps_are_audited_not_rejected(self) -> None:
        start = pd.Timestamp("2024-01-01T00:00:00Z")
        times: list[pd.Timestamp] = []
        for segment_index in range(22):
            segment_start = start + pd.Timedelta(
                hours=segment_index * 3
            )
            times.extend(
                [segment_start, segment_start + pd.Timedelta(hours=1)]
            )
        candles = pd.DataFrame(
            {
                "open_time": times,
                "open": range(len(times)),
                "high": range(1, len(times) + 1),
                "low": range(len(times)),
                "close": range(len(times)),
                "volume": [1.0] * len(times),
            }
        )
        settings = _Settings(
            {
                "training_maximum_gap_hours": 24,
                "training_maximum_gap_count": 12,
                "training_maximum_missing_hours": 72,
                "training_minimum_segment_rows": 1,
            }
        )

        def fake_feature_set(
            frame: pd.DataFrame,
            news: pd.DataFrame,
            settings: _Settings,
            include_labels: bool = True,
        ) -> FeatureSet:
            del news, settings, include_labels
            return FeatureSet(
                frame=frame.copy(),
                feature_columns=["close"],
                horizons=[1],
            )

        with patch(
            "btc_ema_trader.contract_training.build_feature_set",
            side_effect=fake_feature_set,
        ):
            feature_set, audit = build_segmented_feature_set(
                candles,
                pd.DataFrame(),
                settings,  # type: ignore[arg-type]
                include_labels=True,
            )

        self.assertEqual(audit["gap_count"], 21)
        self.assertEqual(audit["gap_count_policy"], "AUDIT_ONLY")
        self.assertTrue(audit["gap_count_advisory_exceeded"])
        self.assertEqual(audit["used_segment_count"], 22)
        self.assertEqual(len(feature_set.frame), len(candles))
        self.assertFalse(audit["synthetic_candles"])
        self.assertFalse(audit["interpolation"])
        self.assertFalse(audit["forward_fill"])


if __name__ == "__main__":
    unittest.main()