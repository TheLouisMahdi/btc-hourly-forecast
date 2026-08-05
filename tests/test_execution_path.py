from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from btc_ema_trader.config import Settings
from btc_ema_trader.execution_path import (
    first_full_candle_open,
    install_execution_path_contract,
    resolve_open_trades_after_entry,
)


class ExecutionPathTests(unittest.TestCase):
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
                "trade_lifecycle": {
                    "same_bar_policy": "NEAREST_TO_OPEN",
                },
            },
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _trade(self) -> dict:
        return install_execution_path_contract(
            {
                "status": "OPEN",
                "direction": "LONG",
                "opened_at": "2026-01-01T01:04:00Z",
                "signal_candle_time": "2026-01-01T00:00:00Z",
                "entry_price": 100.0,
                "target_price": 105.0,
                "initial_stop_price": 99.0,
                "current_stop_price": 99.0,
                "initial_risk_price": 1.0,
                "entry_atr": 1.0,
                "expires_at": "2026-01-02T01:04:00Z",
                "stress_execution_cost_bps": 0.0,
                "notional_usd": 1000.0,
                "risk_budget_usd": 10.0,
                "max_favorable_r": 0.0,
                "max_adverse_r": 0.0,
                "breakeven_trigger_r": 2.0,
                "trailing_trigger_r": 3.0,
                "trailing_atr_multiplier": 1.0,
            }
        )

    def test_partial_entry_candle_is_skipped(self) -> None:
        trade = self._trade()
        candles = pd.DataFrame(
            [
                {
                    "open_time": "2026-01-01T01:00:00Z",
                    "open": 100.0,
                    "high": 106.0,
                    "low": 99.5,
                    "close": 104.0,
                },
                {
                    "open_time": "2026-01-01T02:00:00Z",
                    "open": 104.0,
                    "high": 104.5,
                    "low": 100.0,
                    "close": 102.0,
                },
            ]
        )
        resolved = resolve_open_trades_after_entry(
            [trade], candles, self.settings
        )
        self.assertEqual(resolved, 0)
        self.assertEqual(trade["status"], "OPEN")
        self.assertEqual(
            trade["first_evaluable_candle_open"],
            "2026-01-01T02:00:00+00:00",
        )

    def test_first_full_post_entry_candle_can_resolve_target(self) -> None:
        trade = self._trade()
        candles = pd.DataFrame(
            [
                {
                    "open_time": "2026-01-01T01:00:00Z",
                    "open": 100.0,
                    "high": 106.0,
                    "low": 99.5,
                    "close": 104.0,
                },
                {
                    "open_time": "2026-01-01T02:00:00Z",
                    "open": 102.0,
                    "high": 105.5,
                    "low": 101.0,
                    "close": 105.0,
                },
            ]
        )
        resolved = resolve_open_trades_after_entry(
            [trade], candles, self.settings
        )
        self.assertEqual(resolved, 1)
        self.assertEqual(trade["status"], "CLOSED")
        self.assertEqual(trade["outcome"], "TARGET")

    def test_exact_hour_entry_can_use_that_hour(self) -> None:
        self.assertEqual(
            first_full_candle_open("2026-01-01T01:00:00Z"),
            pd.Timestamp("2026-01-01T01:00:00Z"),
        )


if __name__ == "__main__":
    unittest.main()
