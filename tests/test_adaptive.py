from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_ema_trader.adaptive import OnlineBinaryLearner, adaptive_vector


class AdaptiveLearningTests(unittest.TestCase):
    def test_binary_learner_updates_incrementally(self) -> None:
        learner = OnlineBinaryLearner()
        negative = np.asarray([-1.0, -0.5, 0.0], dtype=float)
        positive = np.asarray([1.0, 0.5, 0.0], dtype=float)
        for _ in range(80):
            learner.update(negative, 0)
            learner.update(positive, 1)
        self.assertTrue(learner.initialized)
        self.assertEqual(learner.samples_seen, 160)
        self.assertLess(
            learner.predict_probability(negative, 0.5),
            learner.predict_probability(positive, 0.5),
        )

    def test_adaptive_vector_is_finite_and_stable(self) -> None:
        row = pd.Series(
            {
                "atr_pct": 0.01,
                "adx": 24.0,
                "rsi_centered": 0.2,
                "price_vs_kama": 0.015,
                "realized_vol_24": 0.03,
                "volume_z_24": 1.1,
                "event_score": 0.8,
                "event_direction": 1,
                "regime_code": 1,
                "bars_since_event": np.nan,
                "news_age_hours": np.inf,
            }
        )
        vector = adaptive_vector(
            row,
            {
                "p_up": 0.61,
                "general_return": 0.002,
                "p_continuation": 0.58,
                "p_tradeable": 0.57,
                "event_return": 0.004,
            },
        )
        self.assertEqual(vector.ndim, 1)
        self.assertTrue(np.isfinite(vector).all())
        self.assertTrue((np.abs(vector) <= 8.0).all())


if __name__ == "__main__":
    unittest.main()
