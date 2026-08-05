from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from github_structural_forecast import (
    attach_optional_forecast,
    build_optional_secondary_forecast,
)


class SecondaryForecastIsolationTests(unittest.TestCase):
    def test_missed_target_window_does_not_create_retroactive_forecast(self) -> None:
        record = {
            "candle_time": "2026-01-01T00:00:00Z",
            "run_finished_at": "2026-01-01T02:05:00Z",
            "price": 100.0,
            "price_forecast_model": {
                "source": "BATCH_CHAMPION",
                "fused_probability_up": 0.6,
                "fused_return": 0.01,
            },
            "general_probabilities": {1: 0.6},
            "general_return_estimates": {1: 0.01},
        }
        contract = build_optional_secondary_forecast(
            record,
            {"1": {"return_mae": 0.005}},
            pd.DataFrame(),
            [],
        )
        self.assertEqual(contract["status"], "NOT_CREATED")
        self.assertEqual(contract["contract_version"], 0)
        self.assertFalse(contract["retroactive_forecast"])

        attached = attach_optional_forecast(
            {
                **record,
                "action": "LONG",
                "trade_plan": {"status": "ACTIONABLE"},
            },
            contract,
        )
        self.assertEqual(attached["action"], "LONG")
        self.assertEqual(attached["trade_plan"]["status"], "ACTIONABLE")
        self.assertEqual(attached["secondary_forecast_status"], "NOT_CREATED")
        self.assertNotIn("next_candle_forecast", attached)
        self.assertEqual(attached["prediction_result"], "NOT_SCORED")

    def test_unavailable_price_model_skips_secondary_contract(self) -> None:
        contract = build_optional_secondary_forecast(
            {
                "candle_time": "2026-01-01T00:00:00Z",
                "price_forecast_model": {
                    "status": "UNAVAILABLE",
                    "error": "test failure",
                },
            },
            {},
            pd.DataFrame(),
            [],
        )
        self.assertEqual(contract["status"], "NOT_CREATED")
        self.assertEqual(contract["timing_status"], "PRICE_MODEL_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
