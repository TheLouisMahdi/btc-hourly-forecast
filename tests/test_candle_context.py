from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from btc_ema_trader.candle_context import (
    CONTEXT_CONTRACT,
    attach_causal_candle_context,
    extract_candle_context,
)


class CandleContextTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "open_time": pd.date_range(
                    "2026-01-01T00:00:00Z", periods=6, freq="h"
                ),
                "open": [100.0, 101.0, 100.5, 102.0, 103.0, 102.0],
                "high": [102.0, 103.0, 103.0, 104.0, 105.0, 106.0],
                "low": [99.0, 100.0, 100.0, 101.0, 101.5, 101.0],
                "close": [101.0, 100.5, 102.0, 103.0, 102.0, 105.0],
                "volume": [10.0, 12.0, 11.0, 15.0, 14.0, 20.0],
                "atr": [2.0] * 6,
                "volume_z_24": [-0.2, 0.1, 0.0, 0.4, 0.3, 1.0],
            }
        )
        return frame

    def test_event_and_two_previous_candles_are_attached(self) -> None:
        result = attach_causal_candle_context(self._frame())
        row = result.iloc[-1]
        self.assertAlmostEqual(row["candle_ctx_lag0_body_atr"], 1.5)
        self.assertAlmostEqual(row["candle_ctx_lag0_upper_wick_atr"], 0.5)
        self.assertAlmostEqual(row["candle_ctx_lag0_lower_wick_atr"], 0.5)
        self.assertAlmostEqual(row["candle_ctx_lag1_body_atr"], -0.5)
        self.assertAlmostEqual(row["candle_ctx_lag2_body_atr"], 0.5)
        self.assertEqual(row["candle_ctx_3bar_bullish_count"], 2.0)

    def test_features_are_prefix_stable_and_do_not_use_future_candles(self) -> None:
        frame = self._frame()
        short = attach_causal_candle_context(frame.iloc[:5])
        full = attach_causal_candle_context(frame)
        columns = [
            column for column in full.columns if column.startswith("candle_ctx_")
        ]
        for column in columns:
            left = short[column].to_numpy(dtype=float)
            right = full.iloc[:5][column].to_numpy(dtype=float)
            np.testing.assert_allclose(left, right, equal_nan=True)

    def test_raw_context_contains_shadows_and_exact_roles(self) -> None:
        context = extract_candle_context(
            self._frame(),
            "2026-01-01T05:00:00Z",
        )
        self.assertEqual(context["contract"], CONTEXT_CONTRACT)
        self.assertTrue(context["complete"])
        self.assertFalse(context["future_bars_used_as_features"])
        self.assertEqual(
            [item["role"] for item in context["bars"]],
            ["PREVIOUS_2", "PREVIOUS_1", "EVENT"],
        )
        event = context["bars"][-1]
        self.assertEqual(event["open"], 102.0)
        self.assertEqual(event["high"], 106.0)
        self.assertEqual(event["low"], 101.0)
        self.assertEqual(event["close"], 105.0)
        self.assertEqual(event["upper_wick"], 1.0)
        self.assertEqual(event["lower_wick"], 1.0)


if __name__ == "__main__":
    unittest.main()
