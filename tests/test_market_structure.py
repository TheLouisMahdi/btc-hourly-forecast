from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from btc_ema_trader.config import Settings
from btc_ema_trader.market_structure import (
    Pivot,
    build_market_structure,
    confirmed_pivots,
    detect_breakout_events,
    detect_triangle,
)


class MarketStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            root=Path("."),
            values={
                "structure": {
                    "lookback_hours": [24, 48, 96],
                    "pivot_left_bars": 2,
                    "pivot_right_bars": 2,
                    "maximum_pivots_per_line": 6,
                    "level_touch_tolerance_atr": 0.30,
                    "triangle_lookback_hours": 80,
                    "triangle_minimum_contraction": 0.10,
                    "triangle_maximum_width_percent": 0.15,
                    "triangle_minimum_line_r2": 0.20,
                    "triangle_minimum_slope_atr": 0.005,
                    "triangle_flat_slope_atr": 0.02,
                    "triangle_minimum_quality": 0.35,
                    "breakout_buffer_atr": 0.10,
                    "breakout_crossing_tolerance_atr": 0.05,
                    "breakout_maximum_extension_atr": 1.50,
                    "breakout_minimum_body_atr": 0.20,
                    "breakout_minimum_volume_z": -0.25,
                    "breakout_invalidation_atr": 0.35,
                    "long_minimum_close_location": 0.65,
                    "short_maximum_close_location": 0.35,
                    "event_cooldown_hours": 4,
                }
            },
        )

    def test_pivot_requires_right_side_confirmation(self) -> None:
        high = np.asarray([1.0, 2.0, 5.0, 3.0, 2.0, 1.0])
        low = np.asarray([0.5, 0.7, 1.0, 0.8, 0.6, 0.4])
        high_pivots, _ = confirmed_pivots(
            high,
            low,
            left=2,
            right=2,
        )
        pivot = next(
            item
            for item in high_pivots
            if item.pivot_at == 2
        )
        self.assertEqual(pivot.confirmed_at, 4)

    def test_converging_pivots_form_a_triangle(self) -> None:
        high_pivots = [
            Pivot(20, 22, 110.0),
            Pivot(40, 42, 108.0),
            Pivot(60, 62, 106.0),
            Pivot(80, 82, 104.0),
        ]
        low_pivots = [
            Pivot(25, 27, 90.0),
            Pivot(45, 47, 92.0),
            Pivot(65, 67, 94.0),
            Pivot(85, 87, 96.0),
        ]
        triangle = detect_triangle(
            index=100,
            high_pivots=high_pivots,
            low_pivots=low_pivots,
            atr=2.0,
            price=100.0,
            lookback=90,
            max_pivots=6,
            cfg=self.settings.section("structure"),
        )
        self.assertIsNotNone(triangle)
        assert triangle is not None
        self.assertEqual(triangle.pattern, "SYMMETRICAL")
        self.assertGreater(triangle.quality, 0.35)
        self.assertGreater(triangle.upper, triangle.lower)

    def test_resistance_breakout_creates_long_event(self) -> None:
        frame = self._event_frame()
        frame.loc[5, "close"] = 101.4
        frame.loc[5, "body_atr"] = 0.75
        frame.loc[5, "close_location"] = 0.92
        event = detect_breakout_events(
            frame,
            self.settings.section("structure"),
        )
        self.assertEqual(event["event_direction"][5], 1)
        self.assertIn("LONG", event["event_type"][5])
        self.assertAlmostEqual(
            event["breakout_level"][5],
            101.0,
        )

    def test_support_breakdown_creates_short_event(self) -> None:
        frame = self._event_frame()
        frame.loc[5, "close"] = 98.6
        frame.loc[5, "body_atr"] = -0.75
        frame.loc[5, "close_location"] = 0.08
        event = detect_breakout_events(
            frame,
            self.settings.section("structure"),
        )
        self.assertEqual(event["event_direction"][5], -1)
        self.assertIn("SHORT", event["event_type"][5])
        self.assertAlmostEqual(
            event["breakout_level"][5],
            99.0,
        )

    def test_price_already_above_resistance_is_not_a_new_breakout(self) -> None:
        frame = self._event_frame()
        frame.loc[4, "close"] = 101.3
        frame.loc[5, "close"] = 101.5
        frame.loc[5, "body_atr"] = 0.75
        frame.loc[5, "close_location"] = 0.92
        event = detect_breakout_events(
            frame,
            self.settings.section("structure"),
        )
        self.assertEqual(event["event_direction"][5], 0)
        self.assertEqual(event["event_type"][5], "NONE")

    def test_price_already_below_support_is_not_a_new_breakdown(self) -> None:
        frame = self._event_frame()
        frame.loc[4, "close"] = 98.7
        frame.loc[5, "close"] = 98.5
        frame.loc[5, "body_atr"] = -0.75
        frame.loc[5, "close_location"] = 0.08
        event = detect_breakout_events(
            frame,
            self.settings.section("structure"),
        )
        self.assertEqual(event["event_direction"][5], 0)
        self.assertEqual(event["event_type"][5], "NONE")

    def test_structure_features_are_prefix_stable(self) -> None:
        count = 180
        x = np.arange(count, dtype=float)
        close = 100.0 + 0.03 * x + 2.0 * np.sin(x / 5.0)
        open_ = close - 0.15 * np.cos(x / 3.0)
        high = np.maximum(open_, close) + 0.65
        low = np.minimum(open_, close) - 0.65
        frame = pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2026-01-01",
                    periods=count,
                    freq="h",
                    tz="UTC",
                ),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "atr": np.full(count, 1.4),
                "body_atr": (close - open_) / 1.4,
                "close_location": (close - low) / (high - low),
                "volume_z_24": np.zeros(count),
            }
        )
        full = build_market_structure(frame, self.settings)
        prefix = build_market_structure(
            frame.iloc[:140],
            self.settings,
        )
        for column in (
            "structure_resistance",
            "structure_support",
            "triangle_quality",
            "event_direction",
            "breakout_level",
        ):
            left = prefix.iloc[-1][column]
            right = full.iloc[139][column]
            if pd.isna(left) and pd.isna(right):
                continue
            self.assertAlmostEqual(
                float(left),
                float(right),
                places=10,
            )

    @staticmethod
    def _event_frame() -> pd.DataFrame:
        count = 8
        return pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2026-01-01",
                    periods=count,
                    freq="h",
                    tz="UTC",
                ),
                "close": np.full(count, 100.0),
                "atr": np.full(count, 2.0),
                "body_atr": np.zeros(count),
                "close_location": np.full(count, 0.5),
                "volume_z_24": np.full(count, 0.5),
                "structure_resistance": np.full(count, 101.0),
                "structure_support": np.full(count, 99.0),
                "resistance_strength": np.full(count, 6.0),
                "support_strength": np.full(count, 6.0),
                "triangle_upper": np.full(count, np.nan),
                "triangle_lower": np.full(count, np.nan),
                "triangle_quality": np.zeros(count),
                "triangle_type": np.full(
                    count,
                    "NONE",
                    dtype=object,
                ),
            }
        )


if __name__ == "__main__":
    unittest.main()
