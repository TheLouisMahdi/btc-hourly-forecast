from __future__ import annotations

import unittest

import numpy as np

from btc_ema_trader.context_trade_features import (
    CONTEXT_TRADE_FEATURES,
    EXTENDED_TRADE_FEATURES,
    context_trade_feature_vector,
    migrate_trade_feature_vectors,
)
from btc_ema_trader.trade_lifecycle import TRADE_FEATURES


class ContextTradeFeatureTests(unittest.TestCase):
    def test_online_vector_appends_causal_candle_shape(self) -> None:
        record = {
            "action": "LONG",
            "confidence": 0.65,
            "tradeability_probability": 0.60,
            "event_candle_context": {
                "bars": [
                    {
                        "role": "PREVIOUS_2",
                        "open": 100.0,
                        "close": 99.0,
                        "volume": 10.0,
                        "body_percent": -0.01,
                        "range": 3.0,
                        "upper_wick": 1.0,
                        "lower_wick": 1.0,
                        "close_location": 0.33,
                    },
                    {
                        "role": "PREVIOUS_1",
                        "open": 99.0,
                        "close": 100.0,
                        "volume": 12.0,
                        "body_percent": 0.010101,
                        "range": 2.0,
                        "upper_wick": 0.5,
                        "lower_wick": 0.5,
                        "close_location": 0.75,
                    },
                    {
                        "role": "EVENT",
                        "open": 100.0,
                        "close": 103.0,
                        "volume": 30.0,
                        "body_percent": 0.03,
                        "range_percent": 0.05,
                        "range": 5.0,
                        "upper_wick": 1.0,
                        "lower_wick": 1.0,
                        "close_location": 0.80,
                    },
                ]
            },
        }
        plan = {
            "atr_pct": 0.01,
            "adx": 20.0,
            "rsi_centered": 0.1,
            "volume_z_24": 1.0,
            "regime_code": 1.0,
        }
        vector = context_trade_feature_vector(
            record,
            plan,
            base_stop_percent=0.01,
            base_reward_r=5.0,
            direction_code=1.0,
        )
        self.assertEqual(
            len(EXTENDED_TRADE_FEATURES),
            len(TRADE_FEATURES) + len(CONTEXT_TRADE_FEATURES),
        )
        self.assertEqual(vector.shape, (len(EXTENDED_TRADE_FEATURES),))
        self.assertTrue(np.isfinite(vector).all())
        extra = vector[-len(CONTEXT_TRADE_FEATURES) :]
        self.assertGreater(extra[0], 0.0)
        self.assertGreater(extra[9], 0.0)
        self.assertGreater(extra[10], 0.0)

    def test_missing_context_uses_finite_neutral_defaults(self) -> None:
        vector = context_trade_feature_vector(
            {"action": "SHORT"},
            {},
            base_stop_percent=0.01,
            base_reward_r=5.0,
            direction_code=-1.0,
        )
        self.assertEqual(vector.shape, (len(EXTENDED_TRADE_FEATURES),))
        self.assertTrue(np.isfinite(vector).all())

    def test_legacy_position_vector_is_migrated_once(self) -> None:
        trade = {
            "entry_feature_names": list(TRADE_FEATURES),
            "entry_feature_vector": [0.0] * len(TRADE_FEATURES),
        }
        self.assertEqual(migrate_trade_feature_vectors([trade]), 1)
        self.assertEqual(
            len(trade["entry_feature_vector"]),
            len(EXTENDED_TRADE_FEATURES),
        )
        self.assertEqual(
            trade["candle_context_migration"],
            "LEGACY_NEUTRAL_CONTEXT",
        )
        self.assertEqual(migrate_trade_feature_vectors([trade]), 0)


if __name__ == "__main__":
    unittest.main()
