from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.price_adaptive import _sample_weight_multiplier


class PriceAdaptiveSampleWeightTests(unittest.TestCase):
    def test_multiplier_is_preserved(self) -> None:
        self.assertAlmostEqual(
            _sample_weight_multiplier(
                pd.Series({"model_sample_weight_multiplier": 0.25})
            ),
            0.25,
        )
        self.assertAlmostEqual(
            _sample_weight_multiplier(
                pd.Series({"model_sample_weight_multiplier": 0.35})
            ),
            0.35,
        )
        self.assertAlmostEqual(
            _sample_weight_multiplier(
                pd.Series({"model_sample_weight_multiplier": 0.15})
            ),
            0.15,
        )

    def test_missing_or_invalid_multiplier_defaults_to_one(self) -> None:
        self.assertAlmostEqual(_sample_weight_multiplier(pd.Series()), 1.0)
        self.assertAlmostEqual(
            _sample_weight_multiplier(
                pd.Series({"model_sample_weight_multiplier": float("nan")})
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
