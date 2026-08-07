from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_ema_trader.price_adaptive import PriceAdaptiveEngine, price_vector


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
            "online_maximum_direction_weight": 0.35,
            "online_maximum_return_weight": 0.35,
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

    def test_online_model_receives_weight_only_when_better(self) -> None:
        engine = object.__new__(PriceAdaptiveEngine)
        engine.config = {
            "online_minimum_samples": 72,
            "online_maximum_direction_weight": 0.35,
            "online_maximum_return_weight": 0.35,
            "online_minimum_brier_improvement": 0.0015,
            "online_minimum_accuracy_improvement": 0.002,
            "online_minimum_return_mae_improvement": 0.00002,
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
        self.assertLessEqual(direction_weight, 0.35)
        self.assertLessEqual(return_weight, 0.35)

    def test_worse_online_model_gets_zero_primary_weight(self) -> None:
        engine = object.__new__(PriceAdaptiveEngine)
        engine.config = {
            "online_minimum_samples": 72,
            "online_maximum_direction_weight": 0.35,
            "online_maximum_return_weight": 0.35,
            "online_minimum_brier_improvement": 0.0015,
            "online_minimum_accuracy_improvement": 0.002,
            "online_minimum_return_mae_improvement": 0.00002,
        }
        direction_weight, return_weight = engine._blend_weights(
            {
                "samples": 504,
                "base_direction_brier": 0.249,
                "online_direction_brier": 0.431,
                "base_direction_accuracy": 0.514,
                "online_direction_accuracy": 0.516,
                "base_return_mae": 0.002193,
                "online_return_mae": 0.002206,
            }
        )
        self.assertEqual(direction_weight, 0.0)
        self.assertEqual(return_weight, 0.0)

    def test_support_only_weight_is_small_when_directions_agree(self) -> None:
        engine = object.__new__(PriceAdaptiveEngine)
        engine.config = {
            "online_minimum_samples": 168,
            "online_support_direction_weight": 0.05,
            "online_support_minimum_accuracy": 0.50,
            "online_support_maximum_accuracy_regression": 0.03,
        }
        weight = engine._support_direction_weight(
            {
                "samples": 504,
                "base_direction_accuracy": 0.524,
                "online_direction_accuracy": 0.502,
            },
            base_probability_up=0.522,
            online_probability_up=0.95,
        )
        self.assertAlmostEqual(weight, 0.05)

    def test_support_only_weight_is_zero_when_directions_disagree(self) -> None:
        engine = object.__new__(PriceAdaptiveEngine)
        engine.config = {
            "online_minimum_samples": 168,
            "online_support_direction_weight": 0.05,
            "online_support_minimum_accuracy": 0.50,
            "online_support_maximum_accuracy_regression": 0.03,
        }
        weight = engine._support_direction_weight(
            {
                "samples": 504,
                "base_direction_accuracy": 0.524,
                "online_direction_accuracy": 0.502,
            },
            base_probability_up=0.522,
            online_probability_up=0.20,
        )
        self.assertEqual(weight, 0.0)

    def test_support_only_rejects_weak_online_accuracy(self) -> None:
        engine = object.__new__(PriceAdaptiveEngine)
        engine.config = {
            "online_minimum_samples": 168,
            "online_support_direction_weight": 0.05,
            "online_support_minimum_accuracy": 0.50,
            "online_support_maximum_accuracy_regression": 0.03,
        }
        weight = engine._support_direction_weight(
            {
                "samples": 504,
                "base_direction_accuracy": 0.54,
                "online_direction_accuracy": 0.49,
            },
            base_probability_up=0.48,
            online_probability_up=0.30,
        )
        self.assertEqual(weight, 0.0)

    def test_support_probability_caps_overconfidence(self) -> None:
        engine = object.__new__(PriceAdaptiveEngine)
        engine.config = {"online_support_probability_cap": 0.65}
        self.assertAlmostEqual(engine._support_probability(0.95), 0.65)
        self.assertAlmostEqual(engine._support_probability(0.05), 0.35)
        self.assertAlmostEqual(engine._support_probability(0.61), 0.61)


if __name__ == "__main__":
    unittest.main()
