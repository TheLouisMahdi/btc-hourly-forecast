"""Standard-library regression tests for chronological replay coverage."""
from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone

from scripts.audit_backtest_readiness import audit

UTC = timezone.utc
START = datetime(2026, 9, 1, tzinfo=UTC)


def stamp(offset: int) -> str:
    return (START + timedelta(hours=offset)).isoformat()


def candle(n: int) -> dict:
    return {"open_time": stamp(n), "open": 100, "high": 102, "low": 99, "close": 101}


def signal(n: int, *, finished_minutes: int = 30, event: bool = True) -> dict:
    out = {
        "candle_time": stamp(n - 1),
        "target_candle_open_time": stamp(n),
        "target_candle_close_time": stamp(n + 1),
        "run_finished_at": (START + timedelta(hours=n, minutes=finished_minutes)).isoformat(),
        "event_type": "RESISTANCE_BREAKOUT_LONG" if event else "NONE",
        "action": "WAIT",
        "actual_candle_open": 100,
        "actual_candle_high": 102,
        "actual_candle_low": 99,
        "actual_close": 101,
        "qualification_passed": False,
    }
    return out


def trade(n: int) -> dict:
    return {
        "status": "CLOSED",
        "opened_at": (START + timedelta(hours=n, minutes=15)).isoformat(),
        "closed_at": stamp(n + 3),
    }


class CausalBacktestReadinessTests(unittest.TestCase):
    def test_mid_candle_signal_is_not_a_preopen_execution(self) -> None:
        h = [signal(2, finished_minutes=-10), signal(3), signal(4, finished_minutes=60), signal(5)]
        original = copy.deepcopy(h)
        report = audit(h, [candle(n) for n in range(8)], [trade(1)])
        self.assertEqual(h, original)
        self.assertEqual(report["target_candle_timing"]["before_target_open"], 1)
        self.assertEqual(report["target_candle_timing"]["during_target_candle"], 2)
        self.assertEqual(report["target_candle_timing"]["at_or_after_target_close"], 1)
        self.assertFalse(report["hypothetical_profit_or_pnl_computed"])

    def test_discontinuous_history_fails_closed_despite_known_winner(self) -> None:
        h = [signal(i) for i in (0, 2, 4, 6)]
        report = audit(h, [candle(i) for i in range(8)], [trade(1)])
        self.assertEqual(report["settled_historical_ohlc"]["observed_hours"], 4)
        self.assertEqual(report["settled_historical_ohlc"]["missing_hours"], 3)
        self.assertEqual(report["status"], "INSUFFICIENT_CAUSAL_BACKTEST_DATA")
        self.assertIn("SETTLED_HISTORICAL_OHLC_IS_DISCONTINUOUS", report["blockers"])

    def test_full_chart_can_only_make_research_replay_ready(self) -> None:
        candles = [candle(i) for i in range(28 * 24 + 12)]
        history = [signal(i) for i in range(28 * 24 + 12)]
        trades = [trade(i * 10) for i in range(30)]
        report = audit(history, candles, trades)
        self.assertEqual(report["status"], "READY_FOR_RESEARCH_REPLAY_ONLY")
        self.assertFalse(report["hypothetical_profit_or_pnl_computed"])
        self.assertEqual(report["blockers"], [])

    def test_trade_with_intrabar_exit_requires_containing_candle(self) -> None:
        row = trade(1)
        row["closed_at"] = (START + timedelta(hours=3, minutes=30)).isoformat()
        eligible = audit([signal(1)], [candle(i) for i in range(5)], [row])
        self.assertEqual(eligible["closed_trades_with_complete_chart_candles"], 1)
        missing = audit([signal(1)], [candle(i) for i in (0, 2, 4)], [row])
        self.assertEqual(missing["closed_trades_with_complete_chart_candles"], 0)

    def test_unresolved_history_does_not_create_false_ohlc(self) -> None:
        row = signal(1)
        row["actual_close"] = None
        row["run_finished_at"] = None
        report = audit([row], [], [])
        self.assertEqual(report["settled_historical_ohlc"]["observed_hours"], 0)
        self.assertEqual(report["target_candle_timing"]["missing_or_invalid_timestamps"], 1)


if __name__ == "__main__":
    unittest.main()
