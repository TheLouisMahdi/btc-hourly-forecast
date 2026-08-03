from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from btc_ema_trader.config import Settings
from btc_ema_trader.market_structure import (
    build_market_structure as build_reference_structure,
)
from btc_ema_trader.market_structure_fast import (
    build_market_structure as build_fast_structure,
)


class FastMarketStructureTests(unittest.TestCase):
    def test_fast_engine_matches_reference_engine(self) -> None:
        settings = Settings(
            root=Path("."),
            values={
                "structure": {
                    "lookback_hours": [24, 48, 96],
                    "pivot_left_bars": 3,
                    "pivot_right_bars": 3,
                    "maximum_pivots_per_line": 8,
                    "level_touch_tolerance_atr": 0.35,
                    "triangle_lookback_hours": 96,
                    "triangle_minimum_contraction": 0.08,
                    "triangle_maximum_width_percent": 0.16,
                    "triangle_minimum_line_r2": 0.15,
                    "triangle_minimum_slope_atr": 0.003,
                    "triangle_flat_slope_atr": 0.025,
                    "triangle_minimum_quality": 0.25,
                    "breakout_buffer_atr": 0.04,
                    "breakout_crossing_tolerance_atr": 0.08,
                    "breakout_maximum_extension_atr": 2.40,
                    "breakout_minimum_body_atr": 0.04,
                    "breakout_minimum_volume_z": -1.50,
                    "breakout_invalidation_atr": 0.60,
                    "long_minimum_close_location": 0.52,
                    "short_maximum_close_location": 0.48,
                    "event_cooldown_hours": 2,
                }
            },
        )
        frame = self._frame(260)
        reference = build_reference_structure(frame, settings)
        fast = build_fast_structure(frame, settings)
        columns = [
            "structure_resistance",
            "structure_support",
            "resistance_strength",
            "support_strength",
            "resistance_age_bars",
            "support_age_bars",
            "triangle_code",
            "triangle_quality",
            "triangle_upper",
            "triangle_lower",
            "event_direction",
            "event_score",
            "breakout_level",
        ]
        for scale in (24, 48, 96):
            columns.extend(
                [
                    f"structure_{scale}h_resistance",
                    f"structure_{scale}h_support",
                    f"structure_{scale}h_resistance_slope_atr",
                    f"structure_{scale}h_support_slope_atr",
                    f"structure_{scale}h_resistance_r2",
                    f"structure_{scale}h_support_r2",
                    f"structure_{scale}h_resistance_touches",
                    f"structure_{scale}h_support_touches",
                    f"structure_{scale}h_width_atr",
                ]
            )
        for column in columns:
            left = pd.to_numeric(reference[column], errors="coerce").to_numpy(
                dtype=float
            )
            right = pd.to_numeric(fast[column], errors="coerce").to_numpy(
                dtype=float
            )
            np.testing.assert_allclose(
                left,
                right,
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
                err_msg=column,
            )

    @staticmethod
    def _frame(count: int) -> pd.DataFrame:
        index = np.arange(count, dtype=float)
        close = (
            100.0
            + 0.025 * index
            + 2.2 * np.sin(index / 7.0)
            + 0.7 * np.sin(index / 19.0)
        )
        open_ = close - 0.18 * np.cos(index / 4.0)
        high = np.maximum(open_, close) + 0.55 + 0.08 * np.sin(index / 3.0)
        low = np.minimum(open_, close) - 0.55 - 0.08 * np.cos(index / 5.0)
        atr = np.full(count, 1.35)
        return pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2024-01-01",
                    periods=count,
                    freq="h",
                    tz="UTC",
                ),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "atr": atr,
                "body_atr": (close - open_) / atr,
                "close_location": (close - low) / (high - low),
                "volume_z_24": 0.4 * np.sin(index / 11.0),
            }
        )


if __name__ == "__main__":
    unittest.main()
