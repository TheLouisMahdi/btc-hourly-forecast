from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_ema_trader.sample_policy import (
    apply_model_calendar,
    filter_history_for_model_policy,
    horizon_path_eligible,
    invalidate_ineligible_labels,
    mark_liquidity_eligibility,
)


class SamplePolicyTests(unittest.TestCase):
    def test_weekends_are_removed_before_features(self) -> None:
        candles = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2026-08-07T23:00:00Z",
                        "2026-08-08T00:00:00Z",
                        "2026-08-09T12:00:00Z",
                        "2026-08-10T00:00:00Z",
                    ],
                    utc=True,
                ),
                "close": [1.0, 2.0, 3.0, 4.0],
            }
        )
        filtered = apply_model_calendar(candles)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(
            filtered["open_time"].dt.dayofweek.tolist(),
            [4, 0],
        )

    def test_low_volume_uses_only_previous_eligible_samples(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2026-08-03T00:00:00Z",
                    periods=6,
                    freq="h",
                ),
                "close": [1.0] * 6,
                "volume": [100.0, 200.0, 300.0, 50.0, 150.0, 250.0],
            }
        )
        marked = mark_liquidity_eligibility(
            frame,
            lookback=3,
            quantile=0.50,
        )
        self.assertEqual(
            marked["model_sample_eligible"].tolist(),
            [True, True, True, False, False, True],
        )
        self.assertAlmostEqual(
            float(marked.loc[3, "model_liquidity_threshold"]),
            200.0,
        )
        self.assertAlmostEqual(
            float(marked.loc[4, "model_liquidity_threshold"]),
            200.0,
        )
        self.assertAlmostEqual(
            float(marked.loc[5, "model_liquidity_threshold"]),
            200.0,
        )

    def test_cross_weekend_target_is_invalid(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2026-08-07T22:00:00Z",
                        "2026-08-07T23:00:00Z",
                        "2026-08-10T00:00:00Z",
                        "2026-08-10T01:00:00Z",
                    ],
                    utc=True,
                ),
                "model_sample_eligible": [True, True, True, True],
            }
        )
        valid = horizon_path_eligible(frame, 1)
        self.assertEqual(valid.tolist(), [True, False, True, False])

    def test_low_liquidity_invalidates_source_and_target_paths(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2026-08-03T00:00:00Z",
                    periods=4,
                    freq="h",
                ),
                "model_sample_eligible": [True, False, True, True],
                "future_return_h1": [0.1, 0.1, 0.1, np.nan],
                "target_up_h1": [1.0, 1.0, 1.0, np.nan],
            }
        )
        output = invalidate_ineligible_labels(frame, [1])
        self.assertTrue(pd.isna(output.loc[0, "future_return_h1"]))
        self.assertTrue(pd.isna(output.loc[1, "future_return_h1"]))
        self.assertAlmostEqual(float(output.loc[2, "future_return_h1"]), 0.1)
        self.assertEqual(
            output["model_path_eligible_h1"].tolist(),
            [False, False, True, False],
        )

    def test_history_excludes_weekends_and_explicit_low_liquidity(self) -> None:
        history = [
            {
                "candle_time": "2026-08-07T22:00:00Z",
                "model_sample_eligible": True,
                "next_candle_forecast": {
                    "target_open_time": "2026-08-07T23:00:00Z"
                },
            },
            {
                "candle_time": "2026-08-08T10:00:00Z",
                "model_sample_eligible": True,
            },
            {
                "candle_time": "2026-08-07T23:00:00Z",
                "model_sample_eligible": True,
                "next_candle_forecast": {
                    "target_open_time": "2026-08-08T00:00:00Z"
                },
            },
            {
                "candle_time": "2026-08-10T10:00:00Z",
                "model_sample_eligible": False,
                "model_low_liquidity": True,
            },
        ]
        filtered = filter_history_for_model_policy(history)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["candle_time"], "2026-08-07T22:00:00Z")


if __name__ == "__main__":
    unittest.main()
