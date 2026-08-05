from __future__ import annotations

import unittest

from btc_ema_trader.github_runtime import preserve_canonical_forecast


class HourlyForecastHistoryTests(unittest.TestCase):
    def test_existing_forecast_is_preserved_for_duplicate_run(self) -> None:
        existing = {
            "candle_time": "2026-01-01T00:00:00+00:00",
            "run_status": "OK",
            "model_id": "first-model",
            "next_candle_forecast": {
                "likely_close_low": 99.0,
                "likely_close_high": 103.0,
            },
            "prediction_result": "PENDING",
        }
        replacement = {
            "candle_time": "2026-01-01T00:00:00+00:00",
            "run_status": "OK",
            "model_id": "second-model",
            "next_candle_forecast": {
                "likely_close_low": 80.0,
                "likely_close_high": 120.0,
            },
            "prediction_result": "PENDING",
        }
        result = preserve_canonical_forecast(
            [existing],
            replacement,
        )
        self.assertEqual(result, existing)
        self.assertEqual(result["model_id"], "first-model")
        self.assertEqual(
            result["next_candle_forecast"]["likely_close_low"],
            99.0,
        )

    def test_legacy_record_can_be_replaced_by_new_contract(self) -> None:
        legacy = {
            "candle_time": "2026-01-01T00:00:00+00:00",
            "run_status": "OK",
            "forecast_direction": "UP",
        }
        replacement = {
            "candle_time": "2026-01-01T00:00:00+00:00",
            "run_status": "OK",
            "next_candle_forecast": {
                "likely_close_low": 99.0,
                "likely_close_high": 103.0,
            },
        }
        result = preserve_canonical_forecast(
            [legacy],
            replacement,
        )
        self.assertEqual(result, replacement)


if __name__ == "__main__":
    unittest.main()
