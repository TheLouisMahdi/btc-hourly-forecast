"""Read-only recorded-entry audit; NOT a counterfactual trading backtest.

Do not change historical fills, online state, strategy gates, or position timing.
Unknown at-entry evidence is never treated as positive or negative evidence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from btc_ema_trader.strategy import STRUCTURAL_EVENTS
from scripts.compare_observed_paper_trades import metrics, validate_ledger

EVIDENCE_FIELDS = (
    "qualification_passed",
    "direction_qualified",
    "economic_policy_available",
    "expected_net_edge_bps",
)
LIMITATION = (
    "This is fixed-ledger attribution, NOT a causal or counterfactual backtest. "
    "Rejected trades would change future opportunity timing, capital availability, "
    "learning and exits. Recorded qualification fields can be missing or stale. "
    "Positive expected_value_usd is not proof of positive execution-adjusted edge."
)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _entry_evidence(trade: dict[str, Any]) -> dict[str, bool | None]:
    structural = str(trade.get("event_type") or "") in STRUCTURAL_EVENTS
    evidence: dict[str, bool | None] = {"structural_event_proxy": structural}
    for field in EVIDENCE_FIELDS[:3]:
        value = trade.get(field)
        evidence[field] = value if isinstance(value, bool) else None
    edge = _finite_number(trade.get("expected_net_edge_bps"))
    evidence["positive_predicted_stress_net_edge"] = (
        edge > 0 if edge is not None else None
    )
    return evidence


def _eligibility(evidence: dict[str, bool | None]) -> str:
    if False in evidence.values():
        return "REJECTED"
    if None in evidence.values():
        return "UNKNOWN"
    return "ELIGIBLE"


def audit(trades: list[dict[str, Any]]) -> dict[str, Any]:
    ledger = validate_ledger(trades)
    event = [trade for trade in ledger if trade.get("event_type") in STRUCTURAL_EVENTS]
    expected_value_positive: list[dict[str, Any]] = []
    expected_value_nonpositive: list[dict[str, Any]] = []
    expected_value_unknown: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {
        "ELIGIBLE": [], "REJECTED": [], "UNKNOWN": []
    }
    missing_evidence = {field: 0 for field in EVIDENCE_FIELDS}
    rows = []
    for trade in ledger:
        evidence = _entry_evidence(trade)
        status = _eligibility(evidence)
        grouped[status].append(trade)
        for field in EVIDENCE_FIELDS:
            if (
                field not in trade
                or trade[field] is None
                or (field == "expected_net_edge_bps"
                    and _finite_number(trade[field]) is None)
            ):
                missing_evidence[field] += 1
        ev = _finite_number(trade.get("expected_value_usd"))
        if ev is None:
            expected_value_unknown.append(trade)
        elif ev > 0:
            expected_value_positive.append(trade)
        else:
            expected_value_nonpositive.append(trade)
        rows.append({
            "trade_id": trade["trade_id"],
            "original_realized_net_pnl_usd": trade["realized_net_pnl_usd"],
            "event_type": trade.get("event_type", "NONE"),
            "observed_expected_value_usd": ev,
            "strict_evidence_status": status,
            "evidence": evidence,
        })
    return {
        "analysis_type": "FIXED_RECORDED_ENTRY_EVIDENCE_NOT_BACKTEST",
        "limitation": LIMITATION,
        "original": metrics(ledger),
        "structural_event_proxy": metrics(event),
        "strict_known_eligible": metrics(grouped["ELIGIBLE"]),
        "strict_known_rejected": metrics(grouped["REJECTED"]),
        "strict_unknown": metrics(grouped["UNKNOWN"]),
        "positive_recorded_expected_value": metrics(expected_value_positive),
        "nonpositive_recorded_expected_value": metrics(expected_value_nonpositive),
        "missing_recorded_expected_value": metrics(expected_value_unknown),
        "evidence_missing_counts": missing_evidence,
        "rejected_recorded_winners": sum(
            t["realized_net_pnl_usd"] > 0 for t in grouped["REJECTED"]
        ),
        "unknown_recorded_winners": sum(
            t["realized_net_pnl_usd"] > 0 for t in grouped["UNKNOWN"]
        ),
        "structural_proxy_excluded_winners": sum(
            t["realized_net_pnl_usd"] > 0
            for t in ledger if t.get("event_type") not in STRUCTURAL_EVENTS
        ),
        "strict_policy_candidate_not_live_enabled": True,
        "entry_evidence": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.trades.read_text(encoding="utf-8"))
    report = audit(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "analysis_type": report["analysis_type"],
        "original": report["original"],
        "structural_event_proxy": report["structural_event_proxy"],
        "strict_known_eligible": report["strict_known_eligible"],
        "strict_known_rejected": report["strict_known_rejected"],
        "strict_unknown": report["strict_unknown"],
        "evidence_missing_counts": report["evidence_missing_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
