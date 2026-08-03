from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

from github_dashboard import resolve_outcomes


def contract() -> dict[str, object]:
    return {
        "contract_version": 2,
        "target": "NEXT_CLOSED_1H_CANDLE",
        "interval_probability": 0.80,
        "source_open_time": "2026-01-01T00:00:00Z",
        "source_close_time": "2026-01-01T01:00:00Z",
        "target_open_time": "2026-01-01T01:00:00Z",
        "target_close_time": "2026-01-01T02:00:00Z",
        "reference_close": 100.0,
        "median_return": 0.01,
        "likely_return_low": -0.01,
        "likely_return_high": 0.03,
        "median_close": 101.0,
        "likely_close_low": 99.0,
        "likely_close_high": 103.0,
        "probability_up": 0.62,
        "probability_down": 0.38,
        "direction": "UP",
        "direction_confidence": 0.62,
        "signal_strength": "MODERATE",
        "scenario": "BULLISH_BIAS",
        "interval_method": "TEST",
        "calibration_samples": 0,
        "forecast_source": "BATCH_CHAMPION",
        "batch_probability_up": 0.62,
        "online_probability_up": 0.62,
        "direction_blend_weight": 0.0,
        "batch_return": 0.01,
        "online_return": 0.01,
        "return_blend_weight": 0.0,
        "return_direction_consistent": True,
    }


def candles(close: float = 102.0) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "open_time": "2026-01-01T00:00:00Z",
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
            },
            {
                "open_time": "2026-01-01T01:00:00Z",
                "open": 100.5,
                "high": 103.0,
                "low": 99.5,
                "close": close,
            },
        ]
    )
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    return frame


class DashboardOutcomeTests(unittest.TestCase):
    def test_target_stays_pending_before_candle_close(self) -> None:
        history = [
            {
                "candle_time": "2026-01-01T00:00:00Z",
                "run_status": "OK",
                "next_candle_forecast": contract(),
            }
        ]
        result = resolve_outcomes(
            history,
            candles(),
            now=pd.Timestamp("2026-01-01T01:59:59Z"),
        )[0]
        self.assertEqual(result["prediction_result"], "PENDING")
        self.assertIsNone(result["actual_close"])

    def test_direction_is_primary_and_interval_is_secondary(self) -> None:
        history = [
            {
                "candle_time": "2026-01-01T00:00:00Z",
                "run_status": "OK",
                "next_candle_forecast": contract(),
            }
        ]
        result = resolve_outcomes(
            history,
            candles(),
            now=pd.Timestamp("2026-01-01T02:01:31Z"),
        )[0]
        self.assertEqual(
            result["prediction_result"],
            "DIRECTION_CORRECT",
        )
        self.assertEqual(
            result["direction_result"],
            "DIRECTION_CORRECT",
        )
        self.assertEqual(result["interval_result"], "IN_RANGE")
        self.assertEqual(result["actual_close"], 102.0)
        self.assertAlmostEqual(result["actual_close_return"], 0.02)
        self.assertIsNotNone(result["resolved_at"])

    def test_wrong_direction_can_still_be_inside_interval(self) -> None:
        bearish = contract()
        bearish["direction"] = "DOWN"
        bearish["probability_up"] = 0.40
        bearish["probability_down"] = 0.60
        history = [
            {
                "candle_time": "2026-01-01T00:00:00Z",
                "run_status": "OK",
                "next_candle_forecast": bearish,
            }
        ]
        result = resolve_outcomes(
            history,
            candles(),
            now=pd.Timestamp("2026-01-01T02:01:31Z"),
        )[0]
        self.assertEqual(
            result["prediction_result"],
            "DIRECTION_WRONG",
        )
        self.assertEqual(result["interval_result"], "IN_RANGE")

    def test_resolved_outcome_is_immutable(self) -> None:
        resolved = {
            "candle_time": "2026-01-01T00:00:00Z",
            "run_status": "OK",
            "next_candle_forecast": contract(),
            "prediction_result": "DIRECTION_CORRECT",
            "direction_result": "DIRECTION_CORRECT",
            "interval_result": "IN_RANGE",
            "actual_close": 102.0,
            "actual_close_return": 0.02,
            "resolved_at": "2026-01-01T02:01:31Z",
        }
        result = resolve_outcomes(
            [resolved],
            candles(close=120.0),
            now=pd.Timestamp("2026-01-01T04:00:00Z"),
        )[0]
        self.assertEqual(
            result["prediction_result"],
            "DIRECTION_CORRECT",
        )
        self.assertEqual(result["actual_close"], 102.0)
        self.assertEqual(
            result["resolved_at"],
            "2026-01-01T02:01:31Z",
        )

    def test_legacy_direction_only_forecast_is_not_scored(self) -> None:
        history = [
            {
                "candle_time": "2026-01-01T00:00:00Z",
                "forecast_direction": "UP",
                "selected_horizon": 2,
                "run_status": "OK",
                "prediction_result": "CORRECT",
            }
        ]
        result = resolve_outcomes(
            history,
            candles(),
            now=pd.Timestamp("2026-01-01T04:00:00Z"),
        )[0]
        self.assertEqual(
            result["prediction_result"],
            "LEGACY_NOT_SCORED",
        )


if __name__ == "__main__":
    unittest.main()
