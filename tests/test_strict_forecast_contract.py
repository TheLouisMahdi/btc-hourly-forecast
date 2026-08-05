from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.strict_forecast_contract import (
    build_strict_next_candle_forecast,
)


class StrictForecastContractTests(unittest.TestCase):
    def _candles(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T01:00:00Z",
                    ],
                    utc=True,
                ),
                "open": [99.0, 100.0],
                "high": [101.0, 102.0],
                "low": [98.0, 99.0],
                "close": [100.0, 101.0],
            }
        )

    def _record(self, created_at: str) -> dict:
        return {
            "candle_time": "2026-01-01T00:00:00Z",
            "created_at": created_at,
            "run_finished_at": created_at,
            "price": 100.0,
            "general_probabilities": {1: 0.61},
            "general_return_estimates": {1: 0.01},
        }

    def test_contract_is_bound_to_exact_future_close(self) -> None:
        result = build_strict_next_candle_forecast(
            self._record("2026-01-01T01:17:00Z"),
            {"1": {"return_mae": 0.005}},
            self._candles(),
            [],
        )
        self.assertEqual(result["contract_version"], 3)
        self.assertEqual(
            result["target_open_time"],
            "2026-01-01T01:00:00+00:00",
        )
        self.assertEqual(
            result["target_close_time"],
            "2026-01-01T02:00:00+00:00",
        )
        self.assertEqual(result["forecast_horizon_seconds"], 43 * 60)
        self.assertEqual(result["timing_status"], "EXACT_NEXT_CLOSED_CANDLE")
        self.assertFalse(result["retroactive_forecast"])

    def test_forecast_before_source_close_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "source candle close"):
            build_strict_next_candle_forecast(
                self._record("2026-01-01T00:59:00Z"),
                {"1": {"return_mae": 0.005}},
                self._candles(),
                [],
            )

    def test_forecast_after_target_close_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "retroactive"):
            build_strict_next_candle_forecast(
                self._record("2026-01-01T02:00:00Z"),
                {"1": {"return_mae": 0.005}},
                self._candles(),
                [],
            )

    def test_source_close_must_match_runtime_record(self) -> None:
        record = self._record("2026-01-01T01:10:00Z")
        record["price"] = 100.5
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_strict_next_candle_forecast(
                record,
                {"1": {"return_mae": 0.005}},
                self._candles(),
                [],
            )


if __name__ == "__main__":
    unittest.main()
