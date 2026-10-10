"""Auditable historical attribution for a fixed paper-trade opportunity ledger.

This is NOT a counterfactual backtest: it neither regenerates missed entries
nor reconstructs exits, position overlap, or online learning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STRUCTURAL_EVENTS = frozenset(
    {"RESISTANCE_BREAKOUT_LONG", "SUPPORT_BREAKDOWN_SHORT"}
)
LIMITATION = (
    "Fixed-opportunity historical attribution only, NOT a counterfactual backtest. "
    "Filtering a completed ledger does not model newly available entry opportunities, "
    "capital reuse, online-learning changes, intrabar fills, or policy-dependent exits."
)


def _timestamp(value: Any) -> datetime:
    instant = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return instant.astimezone(timezone.utc)


def _amount(trade: dict[str, Any], field: str) -> float:
    value = float(trade[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} in trade {trade.get('trade_id')}")
    return value


def _execution_cost(trade: dict[str, Any]) -> float:
    if trade.get("simulated_execution_cost_usd") is not None:
        cost = _amount(trade, "simulated_execution_cost_usd")
    else:
        cost = (
            _amount(trade, "gross_aligned_return")
            - _amount(trade, "realized_net_return")
        ) * _amount(trade, "notional_usd")
    if cost < -1e-6:
        raise ValueError("Realized execution cost must not be negative")
    return max(0.0, cost)


def validate_ledger(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("Expected a JSON array of trades")
    closed = []
    ids: set[str] = set()
    for trade in payload:
        if not isinstance(trade, dict):
            raise ValueError("Every trade must be a JSON object")
        if trade.get("status") != "CLOSED":
            continue
        ident = str(trade.get("trade_id") or "")
        if not ident or ident in ids:
            raise ValueError("Missing or duplicate closed trade ID")
        ids.add(ident)
        if _timestamp(trade["closed_at"]) < _timestamp(trade["opened_at"]):
            raise ValueError(f"Closed before opened: {ident}")
        _amount(trade, "realized_net_pnl_usd")
        _execution_cost(trade)
        closed.append(trade)
    return sorted(closed, key=lambda t: (_timestamp(t["closed_at"]), t["trade_id"]))


def metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnl = [_amount(trade, "realized_net_pnl_usd") for trade in trades]
    positive = sum(value for value in pnl if value > 0)
    negative = sum(value for value in pnl if value < 0)
    equity = peak = maximum_drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
    return {
        "closed_trades": len(trades),
        "winning_trades": sum(value > 0 for value in pnl),
        "losing_or_flat_trades": sum(value <= 0 for value in pnl),
        "net_pnl_usd": round(sum(pnl), 6),
        "gross_wins_usd": round(positive, 6),
        "gross_losses_usd": round(negative, 6),
        "profit_factor": (
            round(positive / abs(negative), 6) if negative < 0 else None
        ),
        "realized_close_only_max_drawdown_usd": round(maximum_drawdown, 6),
        "execution_costs_usd": round(
            sum(_execution_cost(trade) for trade in trades), 6
        ),
    }


def candle_coverage(
    candles: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    times = sorted({_timestamp(c["open_time"]) for c in candles})
    if not times:
        return {"hourly_candles": 0, "fully_covered_closed_trades": 0}
    start = times[0]
    end = times[-1] + timedelta(hours=1)
    full = []
    for trade in trades:
        opening = _timestamp(trade["opened_at"])
        closing = _timestamp(trade["closed_at"])
        if opening >= start and closing <= end:
            full.append(trade)
    return {
        "hourly_candles": len(times),
        "first_candle_open": start.isoformat(),
        "last_candle_close": end.isoformat(),
        "fully_covered_closed_trades": len(full),
        "fully_covered_structural_trades": sum(
            trade.get("event_type") in STRUCTURAL_EVENTS for trade in full
        ),
        "warning": (
            "OHLC coverage does not validate intrabar execution timing, "
            "venue consistency, missing quotes, or model signals."
        ),
    }


def compare(
    trades: list[dict[str, Any]], candles: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    ledger = validate_ledger(trades)
    retained = [t for t in ledger if t.get("event_type") in STRUCTURAL_EVENTS]
    excluded = [t for t in ledger if t.get("event_type") not in STRUCTURAL_EVENTS]
    return {
        "analysis_type": "OBSERVED_LEDGER_ATTRIBUTION_NOT_BACKTEST",
        "limitation": LIMITATION,
        "original_recorded_opportunities": metrics(ledger),
        "retained_structural_opportunities": metrics(retained),
        "excluded_model_only_opportunities": metrics(excluded),
        "excluded_profitable_trade_count": sum(
            _amount(t, "realized_net_pnl_usd") > 0 for t in excluded
        ),
        "excluded_trade_ids": [t["trade_id"] for t in excluded],
        "retained_trade_ids": [t["trade_id"] for t in retained],
        "coverage": candle_coverage(candles, ledger) if candles is not None else None,
        "warnings": [
            "The structural-event proxy is not a replay of every live gate.",
            "Realized-close drawdown excludes unrealized intratrade drawdown.",
            "A small retained subset is not statistical evidence of profitability.",
            "Observed fills are frozen; changes to stop activation and timestamps "
            "are not re-priced by this report.",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    rows = [
        ("Original", report["original_recorded_opportunities"]),
        ("Retained structural", report["retained_structural_opportunities"]),
        ("Excluded model-only", report["excluded_model_only_opportunities"]),
    ]
    output = [
        "# Historical paper-policy attribution (not a backtest)",
        "",
        report["limitation"],
        "",
        "| Recorded opportunities | Count | Winners | Net PnL (USD) | "
        "Profit factor | Close-only max DD (USD) | Costs (USD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in rows:
        factor = "N/A" if m["profit_factor"] is None else f'{m["profit_factor"]:.3f}'
        output.append(
            f'| {name} | {m["closed_trades"]} | {m["winning_trades"]} | '
            f'{m["net_pnl_usd"]:.2f} | {factor} | '
            f'{m["realized_close_only_max_drawdown_usd"]:.2f} | '
            f'{m["execution_costs_usd"]:.2f} |'
        )
    output += [
        "",
        f'Excluded profitable opportunities: {report["excluded_profitable_trade_count"]}.',
        "",
        "## Hourly OHLC coverage",
        "",
        f'```json\n{json.dumps(report["coverage"], indent=2)}\n```',
        "",
        "## Limits",
        "",
        *[f"- {warning}" for warning in report["warnings"]],
        "",
    ]
    return "\n".join(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--candles", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.trades.read_bytes()
    trades = json.loads(raw)
    candles = json.loads(args.candles.read_text("utf-8")) if args.candles else None
    report = compare(trades, candles)
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "attribution.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "attribution.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(markdown_report(report))


if __name__ == "__main__":
    main()
