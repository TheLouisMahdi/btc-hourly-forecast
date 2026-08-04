from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_ema_trader.economic_validation import (
    CalibrationMap,
    apply_calibration,
    compare_with_incumbent,
    evaluate_oof_economics,
)


class EconomicValidationTests(unittest.TestCase):
    def test_calibration_mapping_is_bounded(self) -> None:
        value = apply_calibration(
            0.8,
            {"coefficient": 0.5, "intercept": -0.2},
        )
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)
        identity = CalibrationMap(1.0, 0.0).transform(
            np.array([0.2, 0.8])
        )
        np.testing.assert_allclose(
            identity,
            np.array([0.2, 0.8]),
            atol=1e-8,
        )

    def test_failed_candidate_never_replaces_incumbent(self) -> None:
        candidate = {
            "qualification": {"passed": False},
            "aggregate_holdout_objective_bps": None,
        }
        result = compare_with_incumbent(candidate, None)
        self.assertEqual(result["decision"], "KEEP_INCUMBENT")

    def test_candidate_requires_promotion_margin(self) -> None:
        candidate = {
            "qualification": {"passed": True},
            "aggregate_holdout_objective_bps": 5.1,
        }
        incumbent = {
            "qualification": {"passed": True},
            "aggregate_holdout_objective_bps": 5.0,
        }
        result = compare_with_incumbent(candidate, incumbent)
        self.assertEqual(result["decision"], "KEEP_INCUMBENT")

    def test_economic_gate_can_qualify_stable_cost_aware_edge(self) -> None:
        rows = []
        start = pd.Timestamp("2020-01-01", tz="UTC")
        for index in range(900):
            good = index % 3 != 0
            rows.append(
                {
                    "record_type": "EVENT",
                    "open_time": start + pd.Timedelta(hours=index * 7),
                    "event_id": f"e{index}",
                    "direction_name": "LONG",
                    "horizon": 3,
                    "event_score": 0.75 if good else 0.25,
                    "actual_continuation": 1 if good else 0,
                    "actual_tradeable": 1 if good else 0,
                    "actual_event_net_return": 0.004 if good else -0.003,
                    "p_continuation": 0.78 if good else 0.35,
                    "p_tradeable": 0.76 if good else 0.30,
                    "predicted_event_gross_return": (
                        0.006 if good else 0.001
                    ),
                }
            )
        report = {
            "execution_costs": {
                "entry_fee_bps": 2.0,
                "exit_fee_bps": 5.0,
                "entry_slippage_bps": 1.5,
                "exit_slippage_bps": 2.5,
                "base_cost_bps": 11.0,
                "stress_cost_bps": 16.5,
            }
        }
        result = evaluate_oof_economics(
            report,
            pd.DataFrame(rows),
        )
        self.assertTrue(result["qualification"]["passed"])
        self.assertIn(
            3,
            result["qualification"][
                "qualified_directions"
            ]["LONG"],
        )


if __name__ == "__main__":
    unittest.main()
