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


class DashboardOutcomeTests(unittest.TestCase):
    def test_outcome_uses_next_open_and_horizon_close(self) -> None:
        candles = pd.DataFrame(
            [
                {
                    "open_time": "2026-01-01T00:00:00Z",
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 100.0,
                },
                {
                    "open_time": "2026-01-01T01:00:00Z",
                    "open": 110.0,
                    "high": 111.0,
                    "low": 104.0,
                    "close": 105.0,
                },
            ]
        )
        candles["open_time"] = pd.to_datetime(
            candles["open_time"],
            utc=True,
        )
        history = [
            {
                "candle_time": "2026-01-01T00:00:00Z",
                "forecast_direction": "DOWN",
                "selected_horizon": 1,
                "run_status": "OK",
            }
        ]
        result = resolve_outcomes(history, candles)[0]
        self.assertEqual(result["prediction_result"], "CORRECT")
        self.assertEqual(result["entry_price"], 110.0)
        self.assertEqual(result["actual_price"], 105.0)
        self.assertLess(result["actual_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
