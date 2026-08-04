from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import github_dashboard


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
        "BTC Economic Breakout Forecast",
    )
    document = document.replace(
        "Direction-first forecasting with adaptive price learning",
        "Cost-aware structural signals with locked economic validation",
    )
    document = document.replace(
        "Adaptive next-candle BTC direction and price forecast",
        "Causal BTC structure, cost-aware execution and economic validation",
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
    return 0


def _health_badges(latest: dict[str, Any]) -> str:
    pipeline_ok = latest.get("run_status") == "OK"
    health = latest.get("data_health")
    health = health if isinstance(health, dict) else {}
    data_ok = bool(health.get("candles_ok", False) and health.get("quote_ok", False))
    economic_ok = bool(latest.get("qualification_passed", False))
    return (
        '<div class="health-stack">'
        + _badge("Pipeline", "HEALTHY" if pipeline_ok else "FAIL-SAFE", pipeline_ok)
        + _badge("Data", "FRESH" if data_ok else "WARNING", data_ok)
        + _badge(
            "Economic model",
            "QUALIFIED" if economic_ok else "BLOCKED",
            economic_ok,
        )
        + "</div>"
    )


def _badge(label: str, value: str, ok: bool) -> str:
    return (
        f'<span class="health-badge {"ok" if ok else "warn"}">'
        f'<i></i><small>{_escape(label)}</small><strong>{_escape(value)}</strong>'
        "</span>"
    )


def _economic_panel(latest: dict[str, Any], history: list[Any]) -> str:
    trade_plan = latest.get("trade_plan")
    trade_plan = trade_plan if isinstance(trade_plan, dict) else {}
    contract = latest.get("next_candle_forecast")
    contract = contract if isinstance(contract, dict) else {}
    expected_gross = _number(trade_plan.get("predicted_gross_move_bps"))
    if expected_gross is None:
        expected_gross = abs(_number(latest.get("expected_return")) or 0.0) * 10_000.0
    stress_cost = _number(trade_plan.get("stress_execution_cost_bps"))
    if stress_cost is None:
        stress_cost = _number(latest.get("actual_cost_bps"))
    net_edge = _number(
        trade_plan.get(
            "predicted_stress_net_edge_bps",
            latest.get("expected_net_edge_bps"),
        )
    )
    required_edge = _number(trade_plan.get("minimum_required_net_edge_bps"))
    direction_weight = _number(contract.get("direction_blend_weight")) or 0.0
    return_weight = _number(contract.get("return_blend_weight")) or 0.0
    resolved = sum(
        1
        for item in history
        if isinstance(item, dict)
        and item.get("direction_result")
        in {"DIRECTION_CORRECT", "DIRECTION_WRONG"}
    )
    evidence = (
        "INSUFFICIENT LIVE EVIDENCE"
        if resolved < 100
        else "EARLY LIVE EVIDENCE"
        if resolved < 500
        else "ESTABLISHED LIVE SAMPLE"
    )
    action = str(latest.get("action") or "WAIT")
    return f'''
<section class="panel economic-panel">
  <div class="economic-heading">
    <div>
      <div class="structure-eyebrow">Net edge after execution costs</div>
      <h2>Economic decision</h2>
      <p class="sub">The direction forecast is never treated as a trade unless the locked direction/horizon policy clears calibrated probability, structure and stress-cost gates.</p>
    </div>
    <span class="economic-action">{_escape(action)}</span>
  </div>
  <div class="economic-grid">
    {_tile("Predicted gross move", _bps(expected_gross), "Model-aligned move before costs")}
    {_tile("Stress execution cost", _bps(stress_cost), "Fees, slippage and uncertainty buffer")}
    {_tile("Predicted net edge", _bps(net_edge), "Gross move minus stress execution cost")}
    {_tile("Required net edge", _bps(required_edge), "Locked policy threshold from development data")}
    {_tile("Direction online weight", _percent(direction_weight), "Zero unless online Brier and accuracy improve")}
    {_tile("Return online weight", _percent(return_weight), "Zero unless online return MAE improves")}
    {_tile("Resolved live forecasts", str(resolved), evidence)}
    {_tile("Qualification", "PASSED" if latest.get("qualification_passed") else "BLOCKED", "Locked chronological holdout verdict")}
  </div>
</section>'''


def _structure_panel(latest: dict[str, Any]) -> str:
    trade_plan = latest.get("trade_plan")
    trade_plan = trade_plan if isinstance(trade_plan, dict) else {}
    event_type = str(
        trade_plan.get("event_type") or latest.get("event_type") or "NONE"
    )
    event_score = _number(
        trade_plan.get("event_score", latest.get("trigger_score"))
    )
    breakout_source = str(
        trade_plan.get("breakout_source")
        or latest.get("breakout_source")
        or "NONE"
    )
    breakout_level = _number(
        trade_plan.get("breakout_level", latest.get("breakout_level"))
    )
    invalidation_level = _number(trade_plan.get("invalidation_level"))
    triangle_type = str(
        trade_plan.get("triangle_type")
        or latest.get("triangle_type")
        or "NONE"
    )
    selected_horizon = (
        latest.get("trade_selected_horizon")
        or trade_plan.get("maximum_holding_hours")
        or latest.get("selected_horizon")
    )
    action = str(latest.get("action") or "WAIT")
    state = (
        "Confirmed structural event"
        if event_type != "NONE"
        else "Waiting for a fresh structural crossing"
    )
    return f'''
<section class="panel structure-panel">
  <div class="structure-heading">
    <div>
      <div class="structure-eyebrow">Causal market structure</div>
      <h2>Breakout context</h2>
      <p class="sub">{_escape(state)}. A setup requires a real close-to-close crossing and a qualified 3h, 6h or 12h economic policy.</p>
    </div>
    <span class="structure-action">{_escape(action)}</span>
  </div>
  <div class="structure-grid">
    {_tile("Event", _label(event_type), "Primary setup classification")}
    {_tile("Source", _label(breakout_source), "Dynamic level or triangle boundary")}
    {_tile("Breakout level", _price(breakout_level), "Confirmed crossed structure")}
    {_tile("Invalidation", _price(invalidation_level), "Preferred structural stop reference")}
    {_tile("Triangle", _label(triangle_type), "Causal converging-boundary pattern")}
    {_tile("Event score", _percent(event_score), "Body, volume, close and structure quality")}
    {_tile("Trade horizon", f"{_escape(selected_horizon)}h" if selected_horizon else "—", "Selected only from qualified 3h, 6h and 12h policies")}
    {_tile("Regime", _label(latest.get("regime")), "Long-term structural environment")}
  </div>
</section>'''


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
.health-badge.warn i{background:var(--bad)}
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
    return number if number == number else None


def _price(value: float | None) -> str:
    return "—" if value is None else f"${value:,.2f}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _bps(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1f} bps"


def _label(value: Any) -> str:
    return _escape(str(value or "NONE").replace("_", " ").title())


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
