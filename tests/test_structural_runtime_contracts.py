from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from github_structural_forecast import preserve_canonical_forecast


class StructuralRuntimeContractTests(unittest.TestCase):
    def test_rerun_preserves_forecast_but_refreshes_execution_metadata(self) -> None:
        existing = {
            "candle_time": "2026-01-01T00:00:00Z",
            "run_status": "OK",
            "next_candle_forecast": {
                "contract_version": 3,
                "median_close": 101.0,
            },
            "prediction_result": "DIRECTION_CORRECT",
            "direction_result": "DIRECTION_CORRECT",
            "interval_result": "IN_RANGE",
            "actual_close": 102.0,
            "resolved_at": "2026-01-01T02:01:31Z",
            "execution_quote": {"price": 100.5},
        }
        fresh = {
            "candle_time": "2026-01-01T00:00:00Z",
            "run_status": "OK",
            "run_finished_at": "2026-01-01T02:10:00Z",
            "next_candle_forecast": {
                "contract_version": 3,
                "median_close": 999.0,
            },
            "prediction_result": "PENDING",
            "direction_result": "PENDING",
            "interval_result": "PENDING",
            "resolved_at": None,
            "execution_quote": {"price": 103.0},
            "candidate_trade_plan": {"entry_reference": 103.0},
            "event_candle_context": {"complete": True},
        }
        merged = preserve_canonical_forecast([existing], fresh)
        self.assertEqual(
            merged["next_candle_forecast"]["median_close"],
            101.0,
        )
        self.assertEqual(merged["direction_result"], "DIRECTION_CORRECT")
        self.assertEqual(merged["actual_close"], 102.0)
        self.assertEqual(merged["resolved_at"], "2026-01-01T02:01:31Z")
        self.assertEqual(merged["execution_quote"]["price"], 103.0)
        self.assertEqual(
            merged["candidate_trade_plan"]["entry_reference"],
            103.0,
        )
        self.assertTrue(merged["event_candle_context"]["complete"])
        self.assertEqual(
            merged["run_finished_at"],
            "2026-01-01T02:10:00Z",
        )


if __name__ == "__main__":
    unittest.main()
