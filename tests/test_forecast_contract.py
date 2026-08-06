from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.forecast_contract import (
    attach_close_based_general_labels,
    build_next_candle_forecast,
)


class ForecastContractTests(unittest.TestCase):
    def test_contract_targets_next_close_and_preserves_model_fusion(self) -> None:
        record = {
            "candle_time": "2026-01-01T00:00:00Z",
            "price": 100.0,
            "general_probabilities": {"1": 0.61},
            "general_return_estimates": {"1": 0.012},
            "price_forecast_model": {
                "source": "BATCH_AND_ONLINE",
                "batch_probability_up": 0.58,
                "online_probability_up": 0.67,
                "fused_probability_up": 0.61,
                "direction_blend_weight": 0.33,
                "batch_return": 0.008,
                "online_return": 0.02,
                "fused_return": 0.012,
                "return_blend_weight": 0.33,
            },
        }
        result = build_next_candle_forecast(
            record,
            {
                "1": {
                    "return_mae": 0.01,
                    "close_interval_oof_samples": 300,
                    "close_interval_residual_low": -0.008,
                    "close_interval_residual_high": 0.009,
                }
            },
            pd.DataFrame(),
            [],
        )
        expected_magnitude = 0.67 * 0.008 + 0.33 * 0.02
        self.assertEqual(result["contract_version"], 3)
        self.assertEqual(
            result["target_close_time"],
            "2026-01-01T02:00:00+00:00",
        )
        self.assertEqual(result["direction"], "UP")
        self.assertEqual(result["forecast_source"], "BATCH_AND_ONLINE")
        self.assertAlmostEqual(result["median_return"], expected_magnitude)
        self.assertAlmostEqual(result["raw_fused_return"], 0.012)
        self.assertAlmostEqual(result["batch_probability_up"], 0.58)
        self.assertAlmostEqual(result["online_probability_up"], 0.67)
        self.assertAlmostEqual(result["direction_blend_weight"], 0.33)
        self.assertEqual(
            result["interval_method"],
            "WALK_FORWARD_PRIOR_WITH_VOLATILITY_FLOOR",
        )
        self.assertNotIn("median_close", result)
        expected_center = 100.0 * (1.0 + expected_magnitude)
        self.assertLess(result["likely_close_low"], expected_center)
        self.assertGreater(result["likely_close_high"], expected_center)

    def test_internal_return_alignment_does_not_publish_exact_close(self) -> None:
        record = {
            "candle_time": "2026-01-01T00:00:00Z",
            "price": 100.0,
            "general_probabilities": {"1": 0.61},
            "general_return_estimates": {"1": -0.012},
            "price_forecast_model": {
                "source": "BATCH_AND_ONLINE",
                "batch_probability_up": 0.58,
                "online_probability_up": 0.67,
                "direction_blend_weight": 0.33,
                "batch_return": -0.008,
                "online_return": -0.02,
                "return_blend_weight": 0.33,
            },
        }
        result = build_next_candle_forecast(
            record,
            {"1": {"return_mae": 0.005}},
            pd.DataFrame(),
            [],
        )
        expected_magnitude = 0.67 * 0.008 + 0.33 * 0.02
        self.assertEqual(result["direction"], "UP")
        self.assertAlmostEqual(result["median_return"], expected_magnitude)
        self.assertAlmostEqual(result["raw_fused_return"], -0.012)
        self.assertFalse(result["return_direction_consistent"])
        self.assertTrue(result["direction_alignment_applied"])
        self.assertNotIn("median_close", result)

    def test_direction_is_always_up_or_down(self) -> None:
        record = {
            "candle_time": "2026-01-01T00:00:00Z",
            "price": 100.0,
            "general_probabilities": {"1": 0.499},
            "general_return_estimates": {"1": 0.0},
        }
        result = build_next_candle_forecast(
            record,
            {"1": {"return_mae": 0.005}},
            pd.DataFrame(),
            [],
        )
        self.assertEqual(result["direction"], "DOWN")
        self.assertEqual(result["signal_strength"], "LOW")
        self.assertNotEqual(result["direction"], "RANGE")

    def test_close_based_labels_use_source_and_future_closes(self) -> None:
        frame = pd.DataFrame(
            {
                "close": [100.0, 110.0, 99.0],
                "future_return_h1": [999.0, 999.0, 999.0],
                "target_up_h1": [0.0, 0.0, 0.0],
            }
        )
        result = attach_close_based_general_labels(frame, [1])
        self.assertAlmostEqual(result.loc[0, "future_return_h1"], 0.10)
        self.assertAlmostEqual(result.loc[1, "future_return_h1"], -0.10)
        self.assertEqual(result.loc[0, "target_up_h1"], 1.0)
        self.assertEqual(result.loc[1, "target_up_h1"], 0.0)
        self.assertTrue(pd.isna(result.loc[2, "future_return_h1"]))
        self.assertTrue(pd.isna(result.loc[2, "target_up_h1"]))

    def test_live_conformal_interval_requires_stable_sample_count(self) -> None:
        history = []
        for index in range(80):
            predicted = 0.001
            actual = predicted + (-0.004 + index * 0.0001)
            history.append(
                {
                    "direction_result": "DIRECTION_CORRECT",
                    "interval_result": "IN_RANGE",
                    "actual_close_return": actual,
                    "next_candle_forecast": {
                        "median_return": predicted
                    },
                }
            )
        record = {
            "candle_time": "2026-01-01T00:00:00Z",
            "price": 100.0,
            "general_probabilities": {1: 0.40},
            "general_return_estimates": {1: -0.002},
        }
        result = build_next_candle_forecast(
            record,
            {"1": {"return_mae": 0.002}},
            pd.DataFrame(),
            history,
        )
        self.assertEqual(
            result["interval_method"],
            "LIVE_CONFORMAL_WITH_VOLATILITY_FLOOR",
        )
        self.assertEqual(result["calibration_samples"], 80)
        self.assertEqual(result["direction"], "DOWN")

    def test_recent_volatility_prevents_an_unrealistically_narrow_range(self) -> None:
        closes = [100.0]
        for index in range(48):
            closes.append(closes[-1] * (1.01 if index % 2 == 0 else 0.99))
        candles = pd.DataFrame({"close": closes})
        record = {
            "candle_time": "2026-01-01T00:00:00Z",
            "price": 100.0,
            "general_probabilities": {1: 0.51},
            "general_return_estimates": {1: 0.0001},
        }
        result = build_next_candle_forecast(
            record,
            {"1": {"return_mae": 0.0005}},
            candles,
            [],
        )
        width_return = (
            result["likely_return_high"] - result["likely_return_low"]
        )
        self.assertGreaterEqual(result["volatility_samples"], 24)
        self.assertGreater(width_return, 0.015)


if __name__ == "__main__":
    unittest.main()
