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
    def test_weekends_are_preserved_before_features(self) -> None:
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
        prepared = apply_model_calendar(candles)
        self.assertEqual(len(prepared), 4)
        self.assertEqual(
            prepared["open_time"].dt.dayofweek.tolist(),
            [4, 5, 6, 0],
        )

    def test_low_volume_is_downweighted_not_excluded(self) -> None:
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
            low_liquidity_weight=0.35,
        )
        self.assertEqual(
            marked["model_sample_eligible"].tolist(),
            [True] * 6,
        )
        self.assertEqual(
            marked["model_low_liquidity"].tolist(),
            [False, False, False, True, True, False],
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
        self.assertAlmostEqual(
            float(marked.loc[3, "model_sample_weight_multiplier"]),
            0.35,
        )
        self.assertAlmostEqual(
            float(marked.loc[4, "model_sample_weight_multiplier"]),
            0.35,
        )
        self.assertAlmostEqual(
            float(marked.loc[5, "model_sample_weight_multiplier"]),
            1.0,
        )

    def test_weekend_is_downweighted(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2026-08-07T23:00:00Z",
                        "2026-08-08T00:00:00Z",
                        "2026-08-09T00:00:00Z",
                        "2026-08-10T00:00:00Z",
                    ],
                    utc=True,
                ),
                "close": [1.0] * 4,
                "volume": [100.0] * 4,
            }
        )
        marked = mark_liquidity_eligibility(
            frame,
            liquidity_enabled=False,
            weekend_weight=0.25,
        )
        self.assertEqual(
            marked["model_weekend"].tolist(),
            [False, True, True, False],
        )
        self.assertEqual(
            marked["model_sample_weight_multiplier"].tolist(),
            [1.0, 0.25, 0.25, 1.0],
        )

    def test_weekend_low_volume_uses_minimum_weight_floor(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2026-08-07T21:00:00Z",
                        "2026-08-07T22:00:00Z",
                        "2026-08-07T23:00:00Z",
                        "2026-08-08T00:00:00Z",
                    ],
                    utc=True,
                ),
                "close": [1.0] * 4,
                "volume": [100.0, 200.0, 300.0, 50.0],
            }
        )
        marked = mark_liquidity_eligibility(
            frame,
            lookback=3,
            quantile=0.50,
            weekend_weight=0.25,
            low_liquidity_weight=0.35,
            minimum_weight=0.15,
        )
        self.assertTrue(bool(marked.loc[3, "model_weekend"]))
        self.assertTrue(bool(marked.loc[3, "model_low_liquidity"]))
        self.assertAlmostEqual(
            float(marked.loc[3, "model_sample_weight_multiplier"]),
            0.15,
        )

    def test_cross_weekend_target_is_valid_when_hourly_path_exists(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2026-08-07T23:00:00Z",
                        "2026-08-08T00:00:00Z",
                        "2026-08-08T01:00:00Z",
                    ],
                    utc=True,
                )
            }
        )
        valid = horizon_path_eligible(frame, 1)
        self.assertEqual(valid.tolist(), [True, True, False])

    def test_real_market_gap_still_invalidates_target(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2026-08-07T23:00:00Z",
                        "2026-08-10T00:00:00Z",
                    ],
                    utc=True,
                )
            }
        )
        valid = horizon_path_eligible(frame, 1)
        self.assertEqual(valid.tolist(), [False, False])

    def test_low_liquidity_does_not_invalidate_labels(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2026-08-03T00:00:00Z",
                    periods=4,
                    freq="h",
                ),
                "model_low_liquidity": [False, True, False, False],
                "model_sample_eligible": [True, True, True, True],
                "future_return_h1": [0.1, 0.1, 0.1, np.nan],
                "target_up_h1": [1.0, 1.0, 1.0, np.nan],
            }
        )
        output = invalidate_ineligible_labels(frame, [1])
        self.assertAlmostEqual(float(output.loc[0, "future_return_h1"]), 0.1)
        self.assertAlmostEqual(float(output.loc[1, "future_return_h1"]), 0.1)
        self.assertAlmostEqual(float(output.loc[2, "future_return_h1"]), 0.1)
        self.assertEqual(
            output["model_path_eligible_h1"].tolist(),
            [True, True, True, False],
        )

    def test_history_keeps_weekends_and_low_liquidity(self) -> None:
        history = [
            {
                "candle_time": "2026-08-07T22:00:00Z",
                "model_sample_eligible": True,
            },
            {
                "candle_time": "2026-08-08T10:00:00Z",
                "model_weekend": True,
            },
            {
                "candle_time": "2026-08-10T10:00:00Z",
                "model_low_liquidity": True,
            },
        ]
        filtered = filter_history_for_model_policy(history)
        self.assertEqual(filtered, history)


if __name__ == "__main__":
    unittest.main()
