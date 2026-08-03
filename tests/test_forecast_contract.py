from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.forecast_contract import build_next_candle_forecast


class ForecastContractTests(unittest.TestCase):
    def test_contract_targets_exactly_the_next_closed_hour(self) -> None:
        record = {
            "candle_time": "2026-01-01T00:00:00Z",
            "price": 100.0,
            "probabilities": {"1": 0.60},
            "returns": {"1": 0.01},
        }
        recent = pd.DataFrame(
            {
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 99.5, 100.0],
                "close": [100.0, 101.0, 102.0],
            }
        )
        result = build_next_candle_forecast(
            record,
            {"1": {"return_mae": 0.01}},
            recent,
            [],
        )
        self.assertEqual(result["source_close_time"], "2026-01-01T01:00:00+00:00")
        self.assertEqual(result["target_open_time"], "2026-01-01T01:00:00+00:00")
        self.assertEqual(result["target_close_time"], "2026-01-01T02:00:00+00:00")
        self.assertEqual(result["target"], "NEXT_CLOSED_1H_CANDLE")
        self.assertLess(result["likely_close_low"], result["median_close"])
        self.assertGreater(result["likely_close_high"], result["median_close"])

    def test_resolved_history_calibrates_the_interval(self) -> None:
        history = []
        for index in range(30):
            predicted = 0.001
            actual = predicted + (-0.004 + index * 0.0003)
            history.append(
                {
                    "prediction_result": "IN_RANGE",
                    "actual_close_return": actual,
                    "next_candle_forecast": {"median_return": predicted},
                }
            )
        record = {
            "candle_time": "2026-01-01T00:00:00Z",
            "price": 100.0,
            "probabilities": {1: 0.40},
            "returns": {1: -0.002},
        }
        recent = pd.DataFrame(
            {"high": [101.0], "low": [99.0], "close": [100.0]}
        )
        result = build_next_candle_forecast(
            record,
            {"1": {"return_mae": 0.02}},
            recent,
            history,
        )
        self.assertEqual(
            result["interval_method"],
            "EMPIRICAL_PREQUENTIAL_RESIDUAL",
        )
        self.assertEqual(result["calibration_samples"], 30)
        self.assertEqual(result["direction"], "DOWN")
        self.assertEqual(result["scenario"], "BEARISH_BIAS")


if __name__ == "__main__":
    unittest.main()
