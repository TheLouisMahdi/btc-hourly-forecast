from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import github_boundary_dashboard
import github_dashboard
import github_timing_dashboard
import github_trade_dashboard


def main() -> int:
    status = github_dashboard.main()
    if status != 0:
        return status

    root = Path(__file__).resolve().parents[1]
    site_dir = root / "site"
    latest = _load_json(site_dir / "latest.json", {})
    history = _load_json(site_dir / "history.json", [])
    index_path = site_dir / "index.html"
    document = index_path.read_text(encoding="utf-8")
    document = document.replace(
        "BTC Next-Candle Forecast",
        "BTC Adaptive Directional Breakout Trader",
    )
    document = document.replace(
        "Direction-first forecasting with adaptive price learning",
        "Persistent paper positions with causal structure and immutable outcomes",
    )
    document = document.replace(
        "Adaptive next-candle BTC direction and price forecast",
        "Adaptive BTC paper positions and an exact secondary next-close forecast",
    )
    document = document.replace(
        "</style>",
        _extra_styles() + "\n</style>",
        1,
    )
    document = re.sub(
        r'<div class="health"><i></i>.*?</div>',
        _health_badges(latest),
        document,
        count=1,
    )
    marker = '<section class="panel ledger">'
    panels = _economic_panel(latest, history) + "\n" + _structure_panel(latest)
    if marker in document:
        document = document.replace(marker, panels + "\n" + marker, 1)
    index_path.write_text(document, encoding="utf-8")

    # Component renderers are intentionally orchestrated here so deployment has
    # one canonical dashboard command and one deterministic mutation order.
    github_boundary_dashboard.main()
    github_trade_dashboard.main()
    github_timing_dashboard.main()
    return 0


def _health_badges(latest: dict[str, Any]) -> str:
    pipeline_ok = latest.get("run_status") == "OK"
    health = latest.get("data_health")
    health = health if isinstance(health, dict) else {}
    data_ok = bool(health.get("candles_ok", False) and health.get("quote_ok", False))
    qualification_ok = bool(latest.get("qualification_passed", False))
    paper_only = bool(
        str(latest.get("paper_trade_mode") or "").upper()
        or _plan(latest).get("paper_only", True)
    )
    return (
        '<div class="health-stack">'
        + _badge("Pipeline", "HEALTHY" if pipeline_ok else "FAIL-SAFE", pipeline_ok)
        + _badge("Market data", "FRESH" if data_ok else "WARNING", data_ok)
        + _badge(
            "Economic qualification",
            "QUALIFIED" if qualification_ok else "UNQUALIFIED",
            qualification_ok,
        )
        + _badge("Execution", "PAPER ONLY" if paper_only else "UNKNOWN", paper_only)
        + "</div>"
    )


def _badge(label: str, value: str, ok: bool) -> str:
    return (
        f'<span class="health-badge {"ok" if ok else "warn"}">'
        f'<i></i><small>{_escape(label)}</small><strong>{_escape(value)}</strong>'
        "</span>"
    )


def _economic_panel(latest: dict[str, Any], history: list[Any]) -> str:
    active = latest.get("active_trade")
    active = active if isinstance(active, dict) else None
    current_plan = _plan(latest)
    candidate = latest.get("candidate_trade_plan") if active else current_plan
    candidate = candidate if isinstance(candidate, dict) else current_plan

    expected_gross = _number(candidate.get("predicted_gross_move_bps"))
    stress_cost = _number(candidate.get("stress_execution_cost_bps"))
    if stress_cost is None:
        stress_cost = _number(latest.get("actual_cost_bps"))
    net_edge = _number(
        candidate.get(
            "predicted_stress_net_edge_bps",
            latest.get("expected_net_edge_bps"),
        )
    )
    expected_value = _number(candidate.get("expected_value_usd"))
    ignored = candidate.get("ignored_soft_blockers")
    ignored = ignored if isinstance(ignored, list) else []
    hard = latest.get("candidate_blockers") if active else latest.get("blockers")
    hard = hard if isinstance(hard, list) else []
    mode = str(
        candidate.get("decision_mode")
        or latest.get("paper_trade_mode")
        or "PAPER"
    )
    candidate_action = str(
        latest.get("candidate_action")
        if active
        else latest.get("action")
        or "WAIT"
    )
    resolved_positions = _number(
        (latest.get("trade_lifecycle_summary") or {}).get("resolved_trades")
        if isinstance(latest.get("trade_lifecycle_summary"), dict)
        else None
    )
    resolved_forecasts = sum(
        1
        for item in history
        if isinstance(item, dict)
        and item.get("direction_result")
        in {"DIRECTION_CORRECT", "DIRECTION_WRONG"}
    )
    qualification = bool(latest.get("qualification_passed", False))
    explanation = (
        "The active position is managed from its frozen entry contract. "
        "This panel describes the newest candidate decision separately."
        if active
        else "The candidate is evaluated after stress execution costs. "
        "Aggressive paper mode may explore selected soft-gate failures, but "
        "hard timing, data, structure and duplication checks remain enforced."
    )
    return f'''
<section class="panel economic-panel">
  <div class="economic-heading">
    <div>
      <div class="structure-eyebrow">Candidate economics · separate from active position</div>
      <h2>Economic decision</h2>
      <p class="sub">{_escape(explanation)}</p>
    </div>
    <span class="economic-action">{_escape(candidate_action)}</span>
  </div>
  <div class="economic-grid">
    {_tile("Decision mode", _label(mode), "Economic-gated or aggressive paper exploration")}
    {_tile("Predicted gross move", _bps(expected_gross), "Candidate move before execution costs")}
    {_tile("Stress execution cost", _bps(stress_cost), "Fees, slippage and uncertainty buffer")}
    {_tile("Predicted net edge", _bps(net_edge), "Candidate gross move minus stress cost")}
    {_tile("Expected value", _money(expected_value), "Probability-weighted paper value")}
    {_tile("Ignored soft gates", str(len(ignored)), _list_text(ignored, "None"))}
    {_tile("Hard blockers", str(len(hard)), _list_text(hard, "None"))}
    {_tile("Qualification", "QUALIFIED" if qualification else "UNQUALIFIED", f"{int(resolved_positions or 0)} resolved positions · {resolved_forecasts} resolved secondary forecasts")}
  </div>
</section>'''


def _structure_panel(latest: dict[str, Any]) -> str:
    active = latest.get("active_trade")
    active = active if isinstance(active, dict) else None
    trade_plan = _plan(latest)
    source = trade_plan if active else latest.get("candidate_trade_plan", trade_plan)
    source = source if isinstance(source, dict) else trade_plan

    event_type = str(source.get("event_type") or latest.get("event_type") or "NONE")
    event_score = _number(source.get("event_score"))
    breakout_source = str(
        source.get("breakout_source")
        or latest.get("breakout_source")
        or "NONE"
    )
    breakout_level = _number(
        source.get("breakout_level", latest.get("breakout_level"))
    )
    invalidation_level = _number(
        source.get(
            "invalidation_level",
            latest.get("breakout_invalidation_level"),
        )
    )
    triangle_type = str(
        source.get("triangle_type")
        or latest.get("triangle_type")
        or "NONE"
    )
    selected_horizon = (
        source.get("selected_horizon")
        or latest.get("trade_selected_horizon")
        or latest.get("selected_horizon")
    )
    action = str(latest.get("action") or "WAIT")
    context_complete = bool(
        source.get(
            "candle_context_complete",
            latest.get("candle_context_complete", False),
        )
    )
    state = (
        "Managing the frozen structure that opened the active position"
        if active
        else "Waiting for or evaluating a fresh close-to-close structural crossing"
    )
    return f'''
<section class="panel structure-panel">
  <div class="structure-heading">
    <div>
      <div class="structure-eyebrow">Causal market structure</div>
      <h2>Breakout context</h2>
      <p class="sub">{_escape(state)}. New positions require a real crossing; an existing position may remain open without a new hourly event.</p>
    </div>
    <span class="structure-action">{_escape(action)}</span>
  </div>
  <div class="structure-grid">
    {_tile("Event", _label(event_type), "Structural setup classification")}
    {_tile("Source", _label(breakout_source), "Static, dynamic or triangle boundary")}
    {_tile("Breakout level", _price(breakout_level), "Confirmed crossed structure")}
    {_tile("Invalidation", _price(invalidation_level), "Structural risk reference")}
    {_tile("Triangle", _label(triangle_type), "Causal converging-boundary pattern")}
    {_tile("Event score", _percent(event_score), "Body, volume, close and structure quality")}
    {_tile("Signal horizon", f"{_escape(selected_horizon)}h" if selected_horizon else "—", "Direction-specific event horizon")}
    {_tile("Candle context", "COMPLETE" if context_complete else "UNAVAILABLE", "Event candle plus two previous closed candles")}
    {_tile("Regime", _label(source.get("regime", latest.get("regime"))), "Long-term structural environment")}
  </div>
</section>'''


def _plan(latest: dict[str, Any]) -> dict[str, Any]:
    value = latest.get("trade_plan")
    return value if isinstance(value, dict) else {}


def _tile(title: str, value: str, note: str) -> str:
    return f'''
<div class="structure-tile">
  <span>{_escape(title)}</span>
  <strong>{value}</strong>
  <small>{_escape(note)}</small>
</div>'''


def _extra_styles() -> str:
    return """
.health-stack{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:7px}
.health-badge{display:grid;grid-template-columns:8px auto;column-gap:7px;align-items:center;padding:8px 10px;border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.72)}
.health-badge i{grid-row:1/3;width:8px;height:8px;border-radius:50%;background:var(--ok)}
.health-badge.warn i{background:var(--wait)}
.health-badge small{font-size:8px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.health-badge strong{font-size:10px}
.structure-panel,.economic-panel{margin-top:18px;background:linear-gradient(135deg,rgba(255,255,255,.84),rgba(222,236,231,.54))}
.economic-panel{background:linear-gradient(135deg,rgba(255,255,255,.88),rgba(245,230,223,.52))}
.structure-heading,.economic-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:18px}
.structure-eyebrow{color:var(--sage2);font-size:10px;font-weight:850;letter-spacing:.12em;text-transform:uppercase;margin-bottom:7px}
.structure-action,.economic-action{display:inline-flex;padding:8px 12px;border-radius:999px;background:var(--mint);color:var(--sage2);font-size:11px;font-weight:850}
.economic-action{background:var(--peach2);color:#8f5f59}
.structure-grid,.economic-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.structure-tile{min-width:0;padding:16px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.58)}
.structure-tile span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
.structure-tile strong{display:block;margin-top:8px;font-size:16px;line-height:1.3;overflow-wrap:anywhere}
.structure-tile small{display:block;margin-top:6px;color:var(--muted);font-size:10px;line-height:1.45}
@media(max-width:980px){.structure-grid,.economic-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:620px){.health-stack{justify-content:flex-start}.structure-heading,.economic-heading{flex-direction:column}.structure-grid,.economic-grid{grid-template-columns:1fr}}
"""


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _price(value: float | None) -> str:
    return "—" if value is None else f"${value:,.2f}"


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:+,.2f}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _bps(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1f} bps"


def _label(value: Any) -> str:
    return _escape(str(value or "NONE").replace("_", " ").title())


def _list_text(items: list[Any], empty: str) -> str:
    if not items:
        return empty
    return ", ".join(str(item).replace("_", " ").title() for item in items[:3])


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
