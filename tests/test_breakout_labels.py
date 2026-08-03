from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.features import _attach_labels


class BreakoutLabelTests(unittest.TestCase):
    def test_close_back_below_long_level_is_false_breakout(self) -> None:
        frame = self._frame(
            future_close=99.8,
            future_high=100.2,
            future_low=99.6,
        )
        self._label(frame)
        self.assertEqual(frame.loc[0, "breakout_hold_h1"], 0.0)
        self.assertEqual(frame.loc[0, "false_breakout_h1"], 1.0)
        self.assertEqual(frame.loc[0, "neutral_breakout_h1"], 0.0)
        self.assertEqual(frame.loc[0, "breakout_success_h1"], 0.0)

    def test_small_move_above_level_is_neutral_not_false(self) -> None:
        frame = self._frame(
            future_close=100.03,
            future_high=100.20,
            future_low=99.80,
        )
        self._label(frame)
        self.assertEqual(frame.loc[0, "breakout_hold_h1"], 0.0)
        self.assertEqual(frame.loc[0, "false_breakout_h1"], 0.0)
        self.assertEqual(frame.loc[0, "neutral_breakout_h1"], 1.0)
        self.assertEqual(frame.loc[0, "breakout_success_h1"], 0.0)

    def test_level_hold_with_positive_move_is_successful(self) -> None:
        frame = self._frame(
            future_close=100.8,
            future_high=101.0,
            future_low=99.8,
        )
        self._label(frame)
        self.assertEqual(frame.loc[0, "breakout_hold_h1"], 1.0)
        self.assertEqual(frame.loc[0, "false_breakout_h1"], 0.0)
        self.assertEqual(frame.loc[0, "neutral_breakout_h1"], 0.0)
        self.assertEqual(frame.loc[0, "breakout_success_h1"], 1.0)

    @staticmethod
    def _label(frame: pd.DataFrame) -> None:
        _attach_labels(
            frame,
            horizons=[1],
            strategy_cfg={
                "maker_fee_bps": 2.0,
                "taker_fee_bps": 5.0,
                "entry_slippage_bps": 1.5,
                "exit_slippage_bps": 2.5,
                "fallback_round_trip_cost_bps": 11.0,
                "minimum_profit_buffer_bps": 8.0,
                "label_target_atr_multiplier": 1.5,
                "label_stop_atr_multiplier": 1.0,
            },
            structure_cfg={"label_hold_buffer_atr": 0.05},
        )

    @staticmethod
    def _frame(
        future_close: float,
        future_high: float,
        future_low: float,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": [100.1, 100.0],
                "high": [100.6, future_high],
                "low": [99.9, future_low],
                "close": [100.5, future_close],
                "atr": [1.0, 1.0],
                "event_direction": [1, 0],
                "breakout_level": [100.0, float("nan")],
                "breakout_invalidation_level": [99.0, float("nan")],
            }
        )


if __name__ == "__main__":
    unittest.main()
