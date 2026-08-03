from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_ema_trader.price_adaptive import (
    PriceAdaptiveEngine,
    price_vector,
)


class PriceAdaptiveTests(unittest.TestCase):
    def test_price_vector_is_finite_and_bounded(self) -> None:
        row = pd.Series(
            {
                "atr_pct": 0.012,
                "adx": 28.0,
                "rsi_centered": 0.15,
                "price_vs_kama": 0.009,
                "return_1": 0.003,
                "return_3": -0.002,
                "realized_vol_24": 0.025,
                "volume_z_24": 1.2,
                "event_score": 0.7,
                "event_direction": 1,
                "regime_code": 2,
                "news_weighted_sent_6h": np.nan,
                "news_relevance_6h": 0.6,
                "news_age_hours": np.inf,
                "hour_sin": 0.5,
                "hour_cos": -0.5,
                "weekday_sin": 0.3,
                "weekday_cos": 0.7,
            }
        )
        vector = price_vector(row, 0.61, 0.002)
        self.assertEqual(vector.ndim, 1)
        self.assertTrue(np.isfinite(vector).all())
        self.assertTrue((np.abs(vector) <= 8.0).all())

    def test_online_model_has_no_weight_before_minimum_samples(self) -> None:
        engine = object.__new__(PriceAdaptiveEngine)
        engine.config = {
            "online_minimum_samples": 72,
            "online_maximum_direction_weight": 0.45,
            "online_maximum_return_weight": 0.45,
        }
        direction_weight, return_weight = engine._blend_weights(
            {
                "samples": 71,
                "base_direction_brier": 0.25,
                "online_direction_brier": 0.20,
                "base_direction_accuracy": 0.51,
                "online_direction_accuracy": 0.56,
                "base_return_mae": 0.004,
                "online_return_mae": 0.003,
            }
        )
        self.assertEqual(direction_weight, 0.0)
        self.assertEqual(return_weight, 0.0)

    def test_online_model_receives_weight_when_it_is_better(self) -> None:
        engine = object.__new__(PriceAdaptiveEngine)
        engine.config = {
            "online_minimum_samples": 72,
            "online_maximum_direction_weight": 0.45,
            "online_maximum_return_weight": 0.45,
            "online_brier_tolerance": 0.005,
            "online_accuracy_tolerance": 0.01,
            "online_return_mae_tolerance": 0.0001,
        }
        direction_weight, return_weight = engine._blend_weights(
            {
                "samples": 200,
                "base_direction_brier": 0.25,
                "online_direction_brier": 0.22,
                "base_direction_accuracy": 0.51,
                "online_direction_accuracy": 0.55,
                "base_return_mae": 0.004,
                "online_return_mae": 0.003,
            }
        )
        self.assertGreater(direction_weight, 0.0)
        self.assertGreater(return_weight, 0.0)
        self.assertLessEqual(direction_weight, 0.45)
        self.assertLessEqual(return_weight, 0.45)


if __name__ == "__main__":
    unittest.main()
