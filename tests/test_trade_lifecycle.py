from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from btc_ema_trader.config import Settings
from btc_ema_trader.trade_lifecycle import (
    AdaptiveTradeEngine,
    open_trade_from_record,
    resolve_open_trades,
)


class TradeLifecycleTests(unittest.TestCase):
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
                "strategy": {
                    "paper_only": True,
                    "aggressive_paper_mode": True,
                    "account_equity_usd": 1000.0,
                    "risk_per_trade_fraction": 0.01,
                    "maximum_leverage": 5.0,
                    "maker_fee_bps": 2.0,
                    "taker_fee_bps": 5.0,
                    "entry_slippage_bps": 1.5,
                    "exit_slippage_bps": 2.5,
                    "stress_cost_multiplier": 1.5,
                    "economic_execution_uncertainty_bps": 4.0,
                },
                "trade_lifecycle": {
                    "enabled": True,
                    "base_reward_r": 5.0,
                    "minimum_reward_r": 3.0,
                    "maximum_reward_r": 8.0,
                    "base_stop_atr_multiplier": 0.75,
                    "minimum_stop_percent": 0.0025,
                    "maximum_stop_percent": 0.025,
                    "base_maximum_holding_hours": 72,
                    "minimum_holding_hours": 12,
                    "maximum_holding_hours": 168,
                    "minimum_online_samples": 2,
                    "maximum_online_weight": 0.65,
                    "same_bar_policy": "NEAREST_TO_OPEN",
                },
            },
        )
        self.settings.ensure_runtime_dirs()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _record(self) -> dict:
        return {
            "action": "LONG",
            "forecast_direction": "UP",
            "trade_forecast_direction": "UP",
            "confidence": 0.65,
            "trade_confidence": 0.65,
            "tradeability_probability": 0.62,
            "trigger_score": 0.7,
            "expected_net_edge_bps": 20.0,
            "expected_return": 0.01,
            "event_id": "event-1",
            "event_type": "RESISTANCE_BREAKOUT_LONG",
            "model_id": "model-1",
            "candle_time": "2026-01-01T00:00:00+00:00",
            "run_finished_at": "2026-01-01T00:20:00+00:00",
        }

    def _plan(self) -> dict:
        return {
            "status": "ACTIONABLE",
            "entry_reference": 100.0,
            "stop_price": 99.0,
            "entry_atr": 1.0,
            "atr_pct": 0.01,
            "adx": 25.0,
            "rsi_centered": 0.1,
            "volume_z_24": 1.2,
            "regime_code": 1.0,
            "event_score": 0.7,
            "stress_execution_cost_bps": 15.0,
        }

    def test_initial_plan_uses_five_r_target(self) -> None:
        engine = AdaptiveTradeEngine(self.settings, "model-1")
        plan = engine.enrich_trade_plan(self._record(), self._plan())
        risk = plan["entry_reference"] - plan["stop_price"]
        reward = plan["target_price"] - plan["entry_reference"]
        self.assertAlmostEqual(plan["risk_reward"], 5.0, places=6)
        self.assertAlmostEqual(reward / risk, 5.0, places=6)
        self.assertGreater(plan["target_net_profit_usd"], 0.0)
        self.assertLess(plan["stop_net_loss_usd"], 0.0)

    def test_target_hit_closes_trade(self) -> None:
        engine = AdaptiveTradeEngine(self.settings, "model-1")
        record = self._record()
        record["trade_plan"] = engine.enrich_trade_plan(record, self._plan())
        trade = open_trade_from_record(record)
        self.assertIsNotNone(trade)
        assert trade is not None
        candle = pd.DataFrame(
            [
                {
                    "open_time": pd.Timestamp("2026-01-01T01:00:00Z"),
                    "open": 100.2,
                    "high": trade["target_price"] + 0.1,
                    "low": trade["current_stop_price"] + 0.1,
                    "close": trade["target_price"],
                }
            ]
        )
        self.assertEqual(resolve_open_trades([trade], candle, self.settings), 1)
        self.assertEqual(trade["status"], "CLOSED")
        self.assertEqual(trade["outcome"], "TARGET")
        self.assertGreater(trade["realized_r"], 0.0)

    def test_resolved_trade_updates_online_learner(self) -> None:
        engine = AdaptiveTradeEngine(self.settings, "model-1")
        record = self._record()
        record["trade_plan"] = engine.enrich_trade_plan(record, self._plan())
        trade = open_trade_from_record(record)
        assert trade is not None
        trade.update(
            {
                "status": "CLOSED",
                "outcome": "STOP",
                "closed_at": "2026-01-01T02:00:00+00:00",
                "realized_r": -1.1,
                "realized_net_pnl_usd": -11.0,
            }
        )
        summary = engine.synchronize([trade])
        self.assertTrue(trade["adaptive_learned"])
        self.assertEqual(summary["samples_seen"], 1)
        self.assertEqual(summary["learned_now"], 1)


if __name__ == "__main__":
    unittest.main()
