"""Offline tests for forward-only SMA200 execution auditing."""
from __future__ import annotations

import copy
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sma200_shadow as paper
import sma_forward_audit as monitor

START = 1791676800  # Oct 11, 2026 UTC, completed signal candle.


def source_state(price=100.0):
    data = [{"time": START - (219 - i) * paper.DAY,
             "open": 99.0, "high": 106.0, "low": 90.0,
             "close": 100.0, "volume": 10.0}
            for i in range(220)]
    data[-1]["close"] = 105.0
    state = paper.new_state()
    next_day = START + paper.DAY
    result = paper.step(state, data, price, next_day + 900)
    return {"latest": result, "state": state, "history": state["history"],
            "trades": state["trades"], "candles": data}


class SMAForwardAuditTests(unittest.TestCase):
    def test_first_buy_measures_delay_and_adverse_open_gap(self):
        source = source_state(price=101)
        original = copy.deepcopy(source)
        recorded = []
        def open_at(start, now):
            recorded.append(start)
            return 100.0
        payload = monitor.audit(source, {}, START + paper.DAY + 3600,
                                fetch_open=open_at)
        report = payload["report"]
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0], START + paper.DAY)
        self.assertEqual(report["execution_events"], 1)
        self.assertEqual(report["execution_events_measured"], 1)
        self.assertAlmostEqual(report["average_execution_delay_minutes"], 15.0)
        self.assertAlmostEqual(report["average_adverse_price_drift_bps"], 100.0)
        self.assertGreater(report["total_incremental_cost_usd_at_recorded_qty"], 0)
        self.assertLess(report["sma_net_return_pct"], 0)
        self.assertEqual(report["observed_daily_decisions"], 1)
        self.assertEqual(source, original, "audit must not mutate the actual paper ledger")

    def test_missing_open_reference_does_not_guess_price(self):
        payload = monitor.audit(source_state(), {}, START + paper.DAY + 1000,
                                fetch_open=lambda *_: None)
        row = payload["executions"][0]
        self.assertEqual(row["status"], "PENDING_REFERENCE")
        self.assertIsNone(row["adverse_price_drift_bps"])
        self.assertEqual(payload["report"]["execution_events_measured"], 0)

    def test_second_run_is_idempotent_and_caches_reference(self):
        data = source_state()
        first = monitor.audit(data, {}, START + paper.DAY + 3600,
                              fetch_open=lambda *_: 99.0)
        prior = {"config_sha256": first["report"]["config"]["sha256"],
                 "snapshots": first["snapshots"],
                 "executions": first["executions"]}
        later = monitor.audit(data, prior, START + paper.DAY + 4800,
                              fetch_open=lambda *_: self.fail("Should use saved opening quote"))
        self.assertEqual(later["report"], first["report"])
        self.assertEqual(later["snapshots"], first["snapshots"])
        self.assertEqual(later["executions"], first["executions"])

    def test_daily_replay_adds_new_observation_without_retrospective_orders(self):
        s = source_state()
        initial = monitor.audit(s, {}, START + paper.DAY + 3600,
                                fetch_open=lambda *_: 99.0)
        previous = copy.deepcopy(initial)
        bars = [dict(b, time=b["time"] + paper.DAY) for b in s["candles"]]
        bars[-1]["close"] = 107
        quote = 104.0
        now = START + 2 * paper.DAY + 600
        new = paper.step(s["state"], bars, quote, now)
        s["latest"] = new
        s["history"] = s["state"]["history"]
        s["trades"] = s["state"]["trades"]
        s["candles"] = bars
        updated = monitor.audit(
            s, {"config_sha256": initial["report"]["config"]["sha256"],
                "snapshots": initial["snapshots"],
                "executions": initial["executions"]},
            now + 500,
            fetch_open=lambda *_: 99.0)
        self.assertEqual(updated["report"]["observed_daily_decisions"], 2)
        self.assertEqual(updated["report"]["execution_events"], 1)
        self.assertEqual(updated["report"]["paper_latest_action"], "HOLD")
        self.assertEqual(previous["report"]["observed_daily_decisions"], 1)

    def test_sell_sign_measures_adverse_downward_drift(self):
        s = source_state()
        nxt = [dict(b, time=b["time"] + paper.DAY) for b in s["candles"]]
        nxt[-1]["close"] = 90
        new = paper.step(s["state"], nxt, 98, START + 2 * paper.DAY + 600)
        self.assertEqual(new["action"], "SELL")
        s.update(latest=new, history=s["state"]["history"],
                 trades=s["state"]["trades"], candles=nxt)
        expected = {START + paper.DAY: 99., START + 2 * paper.DAY: 100.}
        out = monitor.audit(s, {}, START + 2 * paper.DAY + 3600,
                            fetch_open=lambda t, now: expected[t])
        self.assertEqual(out["report"]["execution_events"], 2)
        sell = next(x for x in out["executions"] if x["side"] == "SELL")
        self.assertGreater(sell["adverse_price_drift_bps"], 0)
        self.assertAlmostEqual(sell["adverse_price_drift_bps"], 200.)
        self.assertGreater(sell["incremental_cost_usd_at_recorded_qty"], 0)
        self.assertEqual(out["report"]["paper_closed_trades"], 1)

    def test_modified_history_and_parameters_rejected(self):
        s = source_state()
        initial = monitor.audit(s, {}, START + paper.DAY + 3600,
                                fetch_open=lambda *_: 99.)
        s["latest"]["fee_bps_per_side"] += 1
        with self.assertRaisesRegex(ValueError, "parameter drift"):
            monitor.audit(s, {}, START + paper.DAY + 3600,
                          fetch_open=lambda *_: 99.)
        s = source_state()
        prior = {"config_sha256": "wrong",
                 "snapshots": initial["snapshots"],
                 "executions": initial["executions"]}
        with self.assertRaisesRegex(ValueError, "different SMA settings"):
            monitor.audit(s, prior, START + paper.DAY + 3600,
                          fetch_open=lambda *_: 99.)
        s["latest"] = dict(s["latest"], equity_usd=s["latest"]["equity_usd"] + 1)
        with self.assertRaisesRegex(ValueError, "Latest decision"):
            monitor.audit(s, {}, START + paper.DAY + 3600,
                          fetch_open=lambda *_: 99.)

    def test_refuse_an_unfinished_hour_and_bad_exchange_response(self):
        t = START + paper.DAY
        self.assertIsNone(monitor.hourly_open(t, t + 3500))
        with patch.object(monitor, "request_json", return_value=[[t, 99, 110, 100, 105, 1]]):
            self.assertEqual(monitor.hourly_open(t, t + 3601 + 60), 100.)
        with patch.object(monitor, "request_json", return_value=[]):
            self.assertIsNone(monitor.hourly_open(t, t + 3601 + 60))
        with patch.object(monitor, "request_json", return_value={"unexpected": 1}):
            with self.assertRaisesRegex(ValueError, "Invalid Coinbase"):
                monitor.hourly_open(t, t + 3601 + 60)

    def test_backward_and_duplicate_paper_events_fail_closed(self):
        s = source_state()
        s["state"]["position"]["entry_time"] = (
            datetime.fromtimestamp(START, tz=timezone.utc).isoformat())
        with self.assertRaisesRegex(ValueError, "precedes"):
            monitor.paper_events(s["state"])


if __name__ == "__main__":
    unittest.main()
