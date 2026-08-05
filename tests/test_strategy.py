from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from btc_ema_trader.config import Settings
from btc_ema_trader.strategy import make_decision


def _row(**overrides):
    values = {
        "close": 70_000.0,
        "atr": 500.0,
        "atr_pct": 0.007,
        "is_event": 1,
        "event_id": "V5-TEST-LONG",
        "event_type": "RESISTANCE_BREAKOUT_LONG",
        "event_direction": 1,
        "event_score": 0.30,
        "event_scale_hours": 168,
        "breakout_source": "RESISTANCE_168H",
        "breakout_level": 69_900.0,
        "breakout_invalidation_level": 69_300.0,
        "regime": "STRUCTURE_UP",
        "regime_code": 1.0,
        "news_shock": 0,
        "adx": 22.0,
        "rsi_centered": 0.10,
        "volume_z_24": 0.4,
    }
    values.update(overrides)
    return pd.Series(values)


def _prediction(
    *,
    success: float = 0.50,
    tradeability: float = 0.50,
    event_return: float = 0.0,
):
    return {
        "direction": "UP",
        "trade_direction": "UP",
        "confidence": success,
        "agreement": 1.0,
        "expected_return": event_return,
        "expected_event_aligned_return": event_return,
        "selected_horizon": 3,
        "qualified_trade_horizons": [],
        "probabilities": {1: 0.52, 3: 0.55, 6: 0.53, 12: 0.51},
        "continuation": {1: 0.50, 3: success, 6: 0.48, 12: 0.46},
        "tradeability": {1: 0.50, 3: tradeability, 6: 0.48, 12: 0.46},
        "event_returns": {1: 0.0, 3: event_return, 6: -0.001, 12: -0.002},
        "returns": {1: 0.0, 3: event_return, 6: -0.001, 12: -0.002},
        "absolute_event_returns": {
            1: 0.0,
            3: event_return,
            6: -0.001,
            12: -0.002,
        },
    }


def _unqualified_bundle():
    return SimpleNamespace(
        qualification={
            "passed": False,
            "qualified_directions": {"LONG": [], "SHORT": []},
            "economic_policy": {"LONG": {}, "SHORT": {}},
            "economic_stress_cost_bps": 21.0,
        }
    )


def _qualified_bundle():
    return SimpleNamespace(
        qualification={
            "passed": True,
            "qualified_directions": {"LONG": [3], "SHORT": []},
            "economic_stress_cost_bps": 15.0,
            "economic_policy": {
                "LONG": {
                    "3": {
                        "success_probability": 0.58,
                        "tradeability_probability": 0.56,
                        "minimum_event_score": 0.10,
                        "minimum_predicted_stress_edge_bps": 0.0,
                    }
                },
                "SHORT": {},
            },
        }
    )


class StrategyRiskPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(__file__).resolve().parents[1]
        values = deepcopy(
            yaml.safe_load(
                (root / "config" / "default.yaml").read_text(
                    encoding="utf-8"
                )
            )
        )
        temp_root = Path(self.temp.name)
        values["paths"] = {
            "database": "data/test.sqlite3",
            "model_dir": "artifacts/models",
            "report_dir": "artifacts/reports",
            "log_dir": "logs",
            "runtime_state": "data/runtime_state.json",
            "adaptive_state": "data/adaptive_state.joblib",
            "price_adaptive_state": "data/price_adaptive_state.joblib",
            "trade_adaptive_state": "data/trade_adaptive_state.joblib",
        }
        self.settings = Settings(temp_root, values)
        self.settings.ensure_runtime_dirs()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_unqualified_negative_edge_event_still_opens_with_scaled_risk(self) -> None:
        decision = make_decision(
            _row(),
            _prediction(success=0.50, tradeability=0.50, event_return=0.0),
            _unqualified_bundle(),
            self.settings,
        )

        self.assertEqual(decision.action, "LONG")
        self.assertEqual(decision.blockers, [])
        self.assertIn(
            "MODEL_NOT_QUALIFIED",
            decision.trade_plan["soft_risk_flags"],
        )
        self.assertIn(
            "SELECTED_DIRECTION_NOT_QUALIFIED",
            decision.trade_plan["soft_risk_flags"],
        )
        self.assertIn(
            "INSUFFICIENT_STRESS_NET_EDGE",
            decision.trade_plan["soft_risk_flags"],
        )
        self.assertEqual(
            decision.trade_plan["policy_name"],
            "AGGRESSIVE_STRUCTURAL_RISK_SCALED",
        )
        self.assertEqual(decision.trade_plan["policy_version"], 2)
        self.assertGreaterEqual(decision.trade_plan["risk_fraction"], 0.005)
        self.assertLessEqual(decision.trade_plan["risk_fraction"], 0.03)
        self.assertGreater(decision.trade_plan["risk_budget_usd"], 0.0)

    def test_strong_qualified_event_receives_more_risk_than_weak_event(self) -> None:
        weak = make_decision(
            _row(),
            _prediction(success=0.50, tradeability=0.50, event_return=0.0),
            _unqualified_bundle(),
            self.settings,
        )
        strong = make_decision(
            _row(event_score=0.95, volume_z_24=2.0, adx=34.0),
            _prediction(success=0.78, tradeability=0.76, event_return=0.005),
            _qualified_bundle(),
            self.settings,
        )

        self.assertEqual(strong.action, "LONG")
        self.assertEqual(strong.blockers, [])
        self.assertEqual(strong.trade_plan["soft_risk_flags"], [])
        self.assertGreater(
            strong.trade_plan["risk_fraction"],
            weak.trade_plan["risk_fraction"],
        )
        self.assertLessEqual(strong.trade_plan["risk_fraction"], 0.03)
        self.assertGreater(
            strong.trade_plan["risk_score"],
            weak.trade_plan["risk_score"],
        )

    def test_no_event_remains_a_hard_blocker(self) -> None:
        decision = make_decision(
            _row(
                is_event=0,
                event_direction=0,
                event_type="NONE",
                event_id=None,
                breakout_level=None,
                breakout_invalidation_level=None,
            ),
            _prediction(success=0.80, tradeability=0.80, event_return=0.01),
            _qualified_bundle(),
            self.settings,
        )

        self.assertEqual(decision.action, "WAIT")
        self.assertIn("NO_NEW_STRUCTURE_BREAKOUT", decision.blockers)

    def test_missing_invalidation_remains_a_hard_blocker(self) -> None:
        decision = make_decision(
            _row(breakout_invalidation_level=None),
            _prediction(success=0.80, tradeability=0.80, event_return=0.01),
            _qualified_bundle(),
            self.settings,
        )

        self.assertEqual(decision.action, "WAIT")
        self.assertIn("INVALIDATION_LEVEL_UNAVAILABLE", decision.blockers)

    def test_unhealthy_market_data_remains_a_hard_blocker(self) -> None:
        decision = make_decision(
            _row(),
            _prediction(success=0.80, tradeability=0.80, event_return=0.01),
            _qualified_bundle(),
            self.settings,
            data_health={
                "candles_ok": False,
                "quote_ok": True,
                "provider_mismatch": False,
                "model_stale": False,
                "news_stale": False,
            },
        )

        self.assertEqual(decision.action, "WAIT")
        self.assertIn("CANDLE_DATA_UNHEALTHY", decision.blockers)


if __name__ == "__main__":
    unittest.main()
