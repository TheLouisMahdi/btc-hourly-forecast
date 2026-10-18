"""Regression tests for evidence completeness in read-only entry-gate audit."""
from __future__ import annotations

import copy
import unittest

from scripts.audit_economic_entry_gates import audit


def trade(ident: str, event: str, pnl: float, *, expected_value: float = 1.0) -> dict:
    return {
        "trade_id": ident,
        "status": "CLOSED",
        "event_type": event,
        "direction": "LONG",
        "opened_at": "2026-01-01T00:00:00+00:00",
        "closed_at": "2026-01-01T03:00:00+00:00",
        "realized_net_pnl_usd": pnl,
        "notional_usd": 1000.0,
        "gross_aligned_return": (pnl + 1.0) / 1000.0,
        "realized_net_return": pnl / 1000.0,
        "simulated_execution_cost_usd": 1.0,
        "expected_value_usd": expected_value,
    }


class EconomicEntryGateAuditTests(unittest.TestCase):
    def test_qualified_positive_edge_retains_event_without_modifying_records(self) -> None:
        accepted = trade("accepted", "RESISTANCE_BREAKOUT_LONG", 3.0)
        accepted.update(
            qualification_passed=True,
            direction_qualified=True,
            economic_policy_available=True,
            expected_net_edge_bps=18.0,
        )
        rejected = trade("rejected", "NONE", -5.0)
        rejected.update(
            qualification_passed=True,
            direction_qualified=True,
            economic_policy_available=True,
            expected_net_edge_bps=20.0,
        )
        original = copy.deepcopy([accepted, rejected])
        report = audit([accepted, rejected])
        self.assertEqual([accepted, rejected], original)
        self.assertEqual(report["original"]["net_pnl_usd"], -2.0)
        self.assertEqual(report["strict_known_eligible"]["closed_trades"], 1)
        self.assertEqual(report["strict_known_eligible"]["net_pnl_usd"], 3.0)
        self.assertEqual(report["strict_known_rejected"]["closed_trades"], 1)
        self.assertEqual(report["strict_unknown"]["closed_trades"], 0)
        self.assertIn("NOT_BACKTEST", report["analysis_type"])

    def test_unknown_required_evidence_does_not_become_a_negative_or_positive(self) -> None:
        unknown = trade("unknown", "TRIANGLE_BREAKOUT_LONG", 7.0)
        rejected = trade("rejected", "SUPPORT_BREAKDOWN_SHORT", -4.0)
        rejected.update(qualification_passed=False, direction_qualified=False)
        report = audit([unknown, rejected])
        self.assertEqual(report["strict_unknown"]["closed_trades"], 1)
        self.assertEqual(report["strict_known_rejected"]["closed_trades"], 1)
        self.assertEqual(report["strict_known_eligible"]["closed_trades"], 0)
        self.assertEqual(report["unknown_recorded_winners"], 1)
        self.assertEqual(report["evidence_missing_counts"]["economic_policy_available"], 2)
        self.assertEqual(report["evidence_missing_counts"]["expected_net_edge_bps"], 2)

    def test_positive_expected_value_not_equivalent_to_positive_net_edge(self) -> None:
        row = trade("optimistic", "RESISTANCE_BREAKOUT_LONG", -6.0, expected_value=8.0)
        row.update(
            qualification_passed=True,
            direction_qualified=True,
            economic_policy_available=True,
            expected_net_edge_bps=-2.0,
        )
        report = audit([row])
        self.assertEqual(report["positive_recorded_expected_value"]["closed_trades"], 1)
        self.assertEqual(report["strict_known_rejected"]["closed_trades"], 1)

    def test_nonfinite_edge_remains_unknown(self) -> None:
        row = trade("missing", "TRIANGLE_BREAKDOWN_SHORT", 2.0)
        row.update(
            qualification_passed=True,
            direction_qualified=True,
            economic_policy_available=True,
            expected_net_edge_bps=float("nan"),
        )
        report = audit([row])
        self.assertEqual(report["strict_unknown"]["closed_trades"], 1)
        self.assertEqual(report["evidence_missing_counts"]["expected_net_edge_bps"], 1)


if __name__ == "__main__":
    unittest.main()
