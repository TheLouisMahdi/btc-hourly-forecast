from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from btc_ema_trader.meta_filter import (
    META_FEATURES,
    apply_precision_gate,
    meta_vector_from_record,
)
from btc_ema_trader.pattern_memory import (
    LiveCandlePatternMemory,
    adjust_forecast_with_pattern_memory,
    build_static_pattern_bundle,
)


class FakeSettings:
    def __init__(self) -> None:
        self.values = {
            "model": {"trade_horizons_hours": [3, 6, 12]},
            "trade_assistant": {
                "require_qualified_meta_for_position": True,
                "static_pattern_minimum_count": 2,
                "static_pattern_bad_rate": 0.75,
                "live_pattern_minimum_count": 3,
                "live_pattern_bad_rate": 0.67,
                "bloom_false_positive_rate": 0.005,
                "static_forecast_penalty_gain": 0.25,
                "live_forecast_penalty_gain": 0.30,
                "maximum_static_forecast_penalty": 0.08,
                "maximum_live_forecast_penalty": 0.10,
            },
        }

    def section(self, name: str):
        return self.values.get(name, {})


def candle_context() -> dict:
    bars = []
    for role, open_price, close_price, volume in (
        ("PREVIOUS_2", 100.0, 100.4, 10.0),
        ("PREVIOUS_1", 100.4, 100.2, 11.0),
        ("EVENT", 100.2, 100.8, 15.0),
    ):
        high = max(open_price, close_price) + 0.2
        low = min(open_price, close_price) - 0.2
        span = high - low
        bars.append(
            {
                "role": role,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": volume,
                "body_percent": (close_price - open_price) / open_price,
                "range_percent": span / open_price,
                "range": span,
                "upper_wick": high - max(open_price, close_price),
                "lower_wick": min(open_price, close_price) - low,
                "close_location": (close_price - low) / span,
            }
        )
    return {"complete": True, "bars": bars}


def forecast_record(source: str, result: str = "DIRECTION_WRONG") -> dict:
    return {
        "candle_time": source,
        "candle_context_complete": True,
        "event_candle_context": candle_context(),
        "regime": "RANGE",
        "direction_result": result,
        "next_candle_forecast": {
            "source_open_time": source,
            "direction": "UP",
        },
    }


class TradeAssistantTests(unittest.TestCase):
    def test_live_pattern_memory_requires_repeated_resolved_mistakes(self) -> None:
        settings = FakeSettings()
        with tempfile.TemporaryDirectory() as directory:
            memory = LiveCandlePatternMemory(
                settings,
                Path(directory) / "memory.joblib",
            )
            history = [
                forecast_record("2026-08-01T00:00:00+00:00"),
                forecast_record("2026-08-02T00:00:00+00:00"),
                forecast_record("2026-08-03T00:00:00+00:00"),
            ]
            self.assertEqual(memory.synchronize(history), 3)
            assessment = memory.assess(history[-1], direction="UP")
            self.assertTrue(assessment["bad_pattern"])
            self.assertEqual(assessment["count"], 3)
            self.assertEqual(assessment["wrong"], 3)

    def test_pattern_penalty_shrinks_confidence_without_flipping(self) -> None:
        settings = FakeSettings()
        adjusted = adjust_forecast_with_pattern_memory(
            {"fused_probability_up": 0.70},
            record={"event_direction": 1},
            static={
                "available": True,
                "direction_code": 1,
                "bloom_hit": True,
                "count": 4,
                "bad_rate": 1.0,
            },
            live=None,
            settings=settings,
        )
        probability = adjusted["fused_probability_up"]
        self.assertGreater(probability, 0.5)
        self.assertLess(probability, 0.70)
        self.assertTrue(
            adjusted["pattern_memory_adjustment"]["direction_preserved"]
        )

    def test_static_bloom_contains_repeated_false_breakout_pattern(self) -> None:
        settings = FakeSettings()
        rows = []
        for index in range(6):
            rows.append(
                {
                    "open_time": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(hours=index),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 10.0,
                    "event_direction": 1 if index >= 2 else 0,
                    "event_scale_hours": 24,
                    "event_score": 0.6,
                    "false_breakout_h1": 1.0 if index >= 2 else np.nan,
                    "false_breakout_h3": 1.0 if index >= 2 else np.nan,
                    "false_breakout_h6": 1.0 if index >= 2 else np.nan,
                    "false_breakout_h12": 1.0 if index >= 2 else np.nan,
                }
            )
        frame = pd.DataFrame(rows)
        bundle, report = build_static_pattern_bundle(
            frame,
            settings,
            model_id="model-x",
        )
        self.assertEqual(report["horizons"], [1, 3, 6, 12])
        head = bundle.heads["LONG"][1]
        self.assertGreaterEqual(
            sum(int(head.bloom.contains(key)) for key in head.stats),
            1,
        )

    def test_precision_gate_blocks_unqualified_position(self) -> None:
        settings = FakeSettings()
        record = {"action": "LONG"}
        plan, blockers = apply_precision_gate(
            record,
            {"status": "ACTIONABLE"},
            {
                "status": "READY",
                "qualified": False,
                "selected": False,
                "reason": "META_HEAD_NOT_QUALIFIED",
            },
            settings,
        )
        self.assertEqual(plan["status"], "BLOCKED")
        self.assertIn("POSITION_META_NOT_QUALIFIED", blockers)

    def test_qualified_gate_applies_horizon_aligned_exit(self) -> None:
        settings = FakeSettings()
        record = {"action": "LONG", "price": 100.0}
        plan, blockers = apply_precision_gate(
            record,
            {
                "status": "ACTIONABLE",
                "entry_reference": 100.0,
                "entry_atr": 2.0,
            },
            {
                "status": "READY",
                "qualified": True,
                "selected": True,
                "reason": "META_GATE_ACCEPTED",
                "exit_profile": {
                    "horizon_hours": 3.0,
                    "target_atr": 0.9,
                    "stop_atr": 0.6,
                    "breakeven_trigger_r": 0.9,
                    "trailing_trigger_r": 1.3,
                },
            },
            settings,
        )
        self.assertEqual(blockers, [])
        self.assertEqual(plan["maximum_holding_hours"], 3)
        self.assertAlmostEqual(plan["target_price"], 101.8)
        self.assertAlmostEqual(plan["stop_price"], 98.8)
        self.assertAlmostEqual(plan["risk_reward"], 1.5)

    def test_meta_vector_is_fixed_finite_and_uses_candle_context(self) -> None:
        record = {
            "event_direction": 1,
            "selected_horizon": 3,
            "trigger_score": 0.7,
            "event_scale_hours": 48,
            "event_candle_context": candle_context(),
            "base_model": {
                "probabilities": {"1": 0.62},
                "continuation": {"3": 0.66},
                "tradeability": {"3": 0.61},
                "event_returns": {"3": 0.004},
            },
        }
        vector = meta_vector_from_record(record, 1, 3)
        self.assertEqual(vector.shape, (len(META_FEATURES),))
        self.assertTrue(np.isfinite(vector).all())
        self.assertGreater(abs(vector[6]), 0.0)


if __name__ == "__main__":
    unittest.main()
