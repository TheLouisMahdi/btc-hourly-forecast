from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from btc_ema_trader.config import Settings
from btc_ema_trader.context_trade_features import EXTENDED_TRADE_FEATURES
from btc_ema_trader.github_runtime import CanonicalAdaptiveTradeEngine


class _IdentityScaler:
    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return matrix


class _FixedProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                [1.0 - self.probability, self.probability]
                for _ in range(len(matrix))
            ],
            dtype=float,
        )


class GithubRuntimeLossMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.settings = Settings(
            root=root,
            values={
                "paths": {
                    "database": str(root / "db.sqlite3"),
                    "runtime_state": str(root / "runtime.json"),
                    "adaptive_state": str(root / "adaptive.joblib"),
                    "price_adaptive_state": str(root / "price.joblib"),
                    "trade_adaptive_state": str(root / "trade.joblib"),
                    "model_dir": str(root / "models"),
                    "report_dir": str(root / "reports"),
                    "log_dir": str(root / "logs"),
                },
                "strategy": {},
                "trade_lifecycle": {
                    "enabled": True,
                    "minimum_online_samples": 20,
                    "maximum_online_weight": 0.65,
                    "stop_learning_weight": 1.75,
                    "loss_learning_weight": 1.75,
                    "entry_loss_guard_enabled": True,
                    "entry_loss_guard_minimum_samples": 20,
                    "entry_loss_probability_threshold": 0.70,
                    "entry_profit_probability_ceiling": 0.30,
                    "entry_predicted_r_ceiling": 0.0,
                    "entry_loss_probability_margin": 0.35,
                },
            },
        )
        self.settings.ensure_runtime_dirs()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_profitable_stop_is_learned_as_profit(self) -> None:
        engine = CanonicalAdaptiveTradeEngine(self.settings, "model-1")
        trade = {
            "trade_id": "profitable-stop",
            "status": "CLOSED",
            "outcome": "STOP",
            "closed_at": "2026-10-01T00:00:00+00:00",
            "direction": "LONG",
            "realized_r": 0.25,
            "realized_net_pnl_usd": 2.5,
            "entry_feature_names": list(EXTENDED_TRADE_FEATURES),
            "entry_feature_vector": [0.0] * len(EXTENDED_TRADE_FEATURES),
        }

        summary = engine.synchronize([trade])

        self.assertEqual(summary["samples_seen"], 1)
        self.assertEqual(
            engine.state.recent_outcomes[-1]["learning_result"],
            "PROFIT",
        )
        self.assertTrue(engine.state.profit_loss_initialized)
        self.assertFalse(engine.state.initialized)

    def test_high_confidence_loss_pattern_is_vetoed(self) -> None:
        engine = CanonicalAdaptiveTradeEngine(self.settings, "model-1")
        engine.state.samples_seen = 20
        engine.state.profit_loss_initialized = True
        engine.state.scaler = _IdentityScaler()
        engine.state.profit_model = _FixedProbabilityModel(0.20)
        engine.state.loss_model = _FixedProbabilityModel(0.80)
        engine._last_extended_vector = np.zeros(
            len(EXTENDED_TRADE_FEATURES),
            dtype=float,
        )
        engine._last_extended_vector[0] = 1.0
        engine.state.entry_memory = [
            {
                "trade_id": f"loss-{index}",
                "direction_code": 1.0,
                "profitable": False,
                "vector": engine._last_extended_vector.tolist(),
            }
            for index in range(4)
        ]
        record = {"action": "LONG", "blockers": []}
        plan = {
            "status": "ACTIONABLE",
            "adaptive_predicted_r": -0.25,
        }

        output = engine._apply_loss_memory_entry_guard(record, plan)

        self.assertEqual(record["action"], "WAIT")
        self.assertIn("ADAPTIVE_LOSS_MEMORY_VETO", record["blockers"])
        self.assertEqual(output["status"], "BLOCKED")
        self.assertTrue(output["loss_memory_veto"])

    def test_profitable_pattern_remains_actionable(self) -> None:
        engine = CanonicalAdaptiveTradeEngine(self.settings, "model-1")
        engine.state.samples_seen = 20
        engine.state.profit_loss_initialized = True
        engine.state.scaler = _IdentityScaler()
        engine.state.profit_model = _FixedProbabilityModel(0.80)
        engine.state.loss_model = _FixedProbabilityModel(0.20)
        engine._last_extended_vector = np.zeros(
            len(EXTENDED_TRADE_FEATURES),
            dtype=float,
        )
        engine._last_extended_vector[0] = 1.0
        engine.state.entry_memory = [
            {
                "trade_id": f"loss-{index}",
                "direction_code": 1.0,
                "profitable": False,
                "vector": engine._last_extended_vector.tolist(),
            }
            for index in range(4)
        ]
        record = {"action": "LONG", "blockers": []}
        plan = {
            "status": "ACTIONABLE",
            "adaptive_predicted_r": 0.40,
        }

        output = engine._apply_loss_memory_entry_guard(record, plan)

        self.assertEqual(record["action"], "LONG")
        self.assertEqual(record["blockers"], [])
        self.assertEqual(output["status"], "ACTIONABLE")
        self.assertFalse(output["loss_memory_guard"]["veto"])


    def test_loss_model_without_repeated_bad_neighbors_does_not_veto(self) -> None:
        engine = CanonicalAdaptiveTradeEngine(self.settings, "model-1")
        engine.state.samples_seen = 20
        engine.state.profit_loss_initialized = True
        engine.state.scaler = _IdentityScaler()
        engine.state.profit_model = _FixedProbabilityModel(0.10)
        engine.state.loss_model = _FixedProbabilityModel(0.90)
        engine._last_extended_vector = np.zeros(
            len(EXTENDED_TRADE_FEATURES),
            dtype=float,
        )
        engine._last_extended_vector[0] = 1.0
        engine.state.entry_memory = [
            {
                "trade_id": f"mixed-{index}",
                "direction_code": 1.0,
                "profitable": index >= 2,
                "vector": engine._last_extended_vector.tolist(),
            }
            for index in range(4)
        ]
        record = {"action": "LONG", "blockers": []}
        plan = {
            "status": "ACTIONABLE",
            "adaptive_predicted_r": -0.40,
        }

        output = engine._apply_loss_memory_entry_guard(record, plan)

        self.assertEqual(record["action"], "LONG")
        self.assertEqual(output["status"], "ACTIONABLE")
        self.assertFalse(output["loss_memory_guard"]["veto"])
        self.assertFalse(
            output["loss_memory_guard"]["neighbor_memory"][
                "supported_bad_pattern"
            ]
        )


if __name__ == "__main__":
    unittest.main()
