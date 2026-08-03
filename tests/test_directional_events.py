from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from btc_ema_trader.config import Settings
from btc_ema_trader.directional_events import (
    attach_directional_breakout_candidates,
    attach_directional_event_labels,
    event_inventory,
)


class DirectionalEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            root=Path("."),
            values={
                "event_mining": {
                    "structure_scales_hours": [24],
                    "minimum_event_separation_hours": 2,
                    "duplicate_level_similarity_atr": 0.35,
                },
                "event_inventory": {
                    "minimum_events_per_direction": 2000,
                    "minimum_years": 6,
                    "minimum_quarters": 24,
                    "minimum_structure_scales": 4,
                    "minimum_diversity_keys": 48,
                    "minimum_volatility_buckets": 3,
                    "minimum_regimes": 3,
                },
                "long_breakout": {
                    "crossing_tolerance_atr": 0.08,
                    "candidate_minimum_cross_atr": 0.015,
                    "candidate_maximum_extension_atr": 2.20,
                    "candidate_minimum_body_atr": 0.04,
                    "candidate_minimum_close_location": 0.52,
                    "candidate_minimum_volume_z": -1.50,
                    "candidate_minimum_touches": 1,
                    "touch_tolerance_atr": 0.35,
                    "minimum_triangle_quality": 0.25,
                    "invalidation_atr": 0.55,
                    "label_hold_buffer_atr": 0.04,
                    "target_atr_by_horizon": {3: 0.90},
                    "minimum_hold_ratio_by_horizon": {3: 0.34},
                },
                "short_breakdown": {
                    "crossing_tolerance_atr": 0.10,
                    "candidate_minimum_cross_atr": 0.020,
                    "candidate_maximum_extension_atr": 2.40,
                    "candidate_minimum_body_atr": 0.05,
                    "candidate_maximum_close_location": 0.48,
                    "candidate_minimum_volume_z": -1.30,
                    "candidate_minimum_touches": 1,
                    "touch_tolerance_atr": 0.38,
                    "minimum_triangle_quality": 0.25,
                    "invalidation_atr": 0.65,
                    "label_hold_buffer_atr": 0.05,
                    "target_atr_by_horizon": {3: 1.00},
                    "minimum_hold_ratio_by_horizon": {3: 0.34},
                },
                "strategy": {
                    "maker_fee_bps": 2.0,
                    "taker_fee_bps": 5.0,
                    "entry_slippage_bps": 1.5,
                    "exit_slippage_bps": 2.5,
                    "fallback_round_trip_cost_bps": 11.0,
                    "minimum_profit_buffer_bps": 8.0,
                    "stress_cost_multiplier": 1.5,
                },
            },
        )

    def test_resistance_crossing_creates_one_long_event(self) -> None:
        frame = self._base_frame()
        frame.loc[4, "close"] = 99.9
        frame.loc[5, "open"] = 99.9
        frame.loc[5, "close"] = 100.5
        frame.loc[5, "high"] = 100.7
        frame.loc[5, "body_atr"] = 0.60
        frame.loc[5, "close_location"] = 0.90
        frame.loc[6, "close"] = 100.8
        frame.loc[6, "body_atr"] = 0.30
        frame.loc[6, "close_location"] = 0.80

        result = attach_directional_breakout_candidates(
            frame,
            self.settings,
        )
        self.assertEqual(result.loc[5, "event_direction"], 1)
        self.assertEqual(
            result.loc[5, "event_type"],
            "RESISTANCE_BREAKOUT_LONG",
        )
        self.assertEqual(result.loc[5, "event_scale_hours"], 24)
        self.assertAlmostEqual(result.loc[5, "breakout_level"], 100.0)
        self.assertEqual(result.loc[6, "event_direction"], 0)

    def test_support_crossing_creates_one_short_event(self) -> None:
        frame = self._base_frame()
        frame.loc[4, "close"] = 100.1
        frame.loc[5, "open"] = 100.1
        frame.loc[5, "close"] = 99.4
        frame.loc[5, "low"] = 99.2
        frame.loc[5, "body_atr"] = -0.70
        frame.loc[5, "close_location"] = 0.08

        result = attach_directional_breakout_candidates(
            frame,
            self.settings,
        )
        self.assertEqual(result.loc[5, "event_direction"], -1)
        self.assertEqual(
            result.loc[5, "event_type"],
            "SUPPORT_BREAKDOWN_SHORT",
        )
        self.assertAlmostEqual(result.loc[5, "breakout_level"], 100.0)

    def test_mining_is_deterministic(self) -> None:
        frame = self._base_frame()
        frame.loc[4, "close"] = 99.9
        frame.loc[5, "close"] = 100.5
        frame.loc[5, "body_atr"] = 0.60
        frame.loc[5, "close_location"] = 0.90
        first = attach_directional_breakout_candidates(
            frame,
            self.settings,
        )
        second = attach_directional_breakout_candidates(
            frame,
            self.settings,
        )
        columns = [
            "event_id",
            "event_direction",
            "event_type",
            "event_score",
            "breakout_level",
            "event_diversity_key",
        ]
        pd.testing.assert_frame_equal(first[columns], second[columns])

    def test_directional_labels_use_separate_formulas(self) -> None:
        frame = self._base_frame(count=8)
        frame.loc[1, "event_direction"] = 1
        frame.loc[1, "event_type"] = "RESISTANCE_BREAKOUT_LONG"
        frame.loc[1, "event_id"] = "LONG-1"
        frame.loc[1, "is_event"] = 1
        frame.loc[1, "breakout_level"] = 100.0
        frame.loc[1, "breakout_invalidation_level"] = 99.45
        frame.loc[2:4, "open"] = [100.1, 100.4, 100.7]
        frame.loc[2:4, "high"] = [100.5, 100.9, 101.2]
        frame.loc[2:4, "low"] = [99.9, 100.2, 100.5]
        frame.loc[2:4, "close"] = [100.4, 100.8, 101.0]

        labeled = attach_directional_event_labels(
            frame,
            self.settings,
            [3],
        )
        self.assertEqual(labeled.loc[1, "breakout_success_h3"], 1.0)
        self.assertEqual(labeled.loc[1, "false_breakout_h3"], 0.0)
        self.assertGreater(labeled.loc[1, "event_gross_return_h3"], 0.0)

    def test_inventory_fails_instead_of_oversampling(self) -> None:
        frame = self._base_frame()
        frame.loc[1, "event_direction"] = 1
        frame.loc[1, "event_id"] = "LONG-1"
        frame.loc[1, "event_diversity_key"] = "A"
        frame.loc[1, "event_scale_hours"] = 24
        frame.loc[1, "event_type"] = "RESISTANCE_BREAKOUT_LONG"
        frame.loc[1, "breakout_source"] = "RESISTANCE_24H"
        frame.loc[2, "event_direction"] = -1
        frame.loc[2, "event_id"] = "SHORT-1"
        frame.loc[2, "event_diversity_key"] = "B"
        frame.loc[2, "event_scale_hours"] = 24
        frame.loc[2, "event_type"] = "SUPPORT_BREAKDOWN_SHORT"
        frame.loc[2, "breakout_source"] = "SUPPORT_24H"
        with self.assertRaisesRegex(ValueError, "at least 2000"):
            event_inventory(frame, self.settings)

    @staticmethod
    def _base_frame(count: int = 10) -> pd.DataFrame:
        timestamps = pd.date_range(
            "2020-01-01",
            periods=count,
            freq="h",
            tz="UTC",
        )
        return pd.DataFrame(
            {
                "open_time": timestamps,
                "open": np.full(count, 100.0),
                "high": np.full(count, 100.3),
                "low": np.full(count, 99.7),
                "close": np.full(count, 100.0),
                "atr": np.full(count, 1.0),
                "atr_pct": np.full(count, 0.01),
                "body_atr": np.zeros(count),
                "close_location": np.full(count, 0.5),
                "volume_z_24": np.zeros(count),
                "regime": np.full(count, "RANGE", dtype=object),
                "regime_code": np.zeros(count),
                "rsi_centered": np.zeros(count),
                "ema_168_slope_6": np.zeros(count),
                "price_vs_ema_168": np.zeros(count),
                "adx": np.full(count, 20.0),
                "structure_24h_resistance": np.full(count, 100.0),
                "structure_24h_support": np.full(count, 100.0),
                "structure_24h_resistance_touches": np.full(count, 2.0),
                "structure_24h_support_touches": np.full(count, 2.0),
                "structure_24h_resistance_slope_atr": np.zeros(count),
                "structure_24h_support_slope_atr": np.zeros(count),
                "structure_24h_resistance_r2": np.full(count, 0.5),
                "structure_24h_support_r2": np.full(count, 0.5),
                "triangle_type": np.full(count, "NONE", dtype=object),
                "triangle_quality": np.zeros(count),
                "triangle_upper": np.full(count, np.nan),
                "triangle_lower": np.full(count, np.nan),
                "event_direction": np.zeros(count, dtype=int),
                "event_type": np.full(count, "NONE", dtype=object),
                "event_id": np.full(count, None, dtype=object),
                "event_diversity_key": np.full(count, None, dtype=object),
                "event_scale_hours": np.zeros(count, dtype=int),
                "is_event": np.zeros(count, dtype=int),
                "breakout_source": np.full(count, "NONE", dtype=object),
                "breakout_level": np.full(count, np.nan),
                "breakout_invalidation_level": np.full(count, np.nan),
            }
        )


if __name__ == "__main__":
    unittest.main()
