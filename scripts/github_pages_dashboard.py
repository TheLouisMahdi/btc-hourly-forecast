from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import github_dashboard


def main() -> int:
    status = github_dashboard.main()
    if status != 0:
        return status
    root = Path(__file__).resolve().parents[1]
    site_dir = root / "site"
    latest = _load_json(site_dir / "latest.json")
    index_path = site_dir / "index.html"
    document = index_path.read_text(encoding="utf-8")
    document = document.replace(
        "BTC Next-Candle Forecast",
        "BTC Structural Breakout Forecast",
    )
    document = document.replace(
        "Direction-first forecasting with adaptive price learning",
        "Dynamic support, resistance and triangle breakout intelligence",
    )
    document = document.replace(
        "Adaptive next-candle BTC direction and price forecast",
        "Causal BTC structure, breakout and next-candle forecast",
    )
    document = document.replace(
        "</style>",
        _structure_styles() + "\n</style>",
        1,
    )
    marker = '<section class="panel ledger">'
    panel = _structure_panel(latest)
    if marker in document:
        document = document.replace(
            marker,
            panel + "\n" + marker,
            1,
        )
    index_path.write_text(document, encoding="utf-8")
    return 0


def _structure_panel(latest: dict[str, Any]) -> str:
    trade_plan = latest.get("trade_plan")
    if not isinstance(trade_plan, dict):
        trade_plan = {}
    event_type = str(
        trade_plan.get("event_type")
        or latest.get("event_type")
        or "NONE"
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
    invalidation_level = _number(
        trade_plan.get("invalidation_level")
    )
    triangle_type = str(
        trade_plan.get("triangle_type")
        or latest.get("triangle_type")
        or "NONE"
    )
    selected_horizon = latest.get("selected_horizon")
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
      <p class="sub">{_escape(state)}. A setup requires a real close-to-close crossing of a confirmed dynamic or static level.</p>
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
    {_tile("Trade horizon", f"{_escape(selected_horizon)}h" if selected_horizon else "—", "Selected from 1h, 3h and 6h")}
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


def _structure_styles() -> str:
    return """
.structure-panel{margin-top:18px;background:linear-gradient(135deg,rgba(255,255,255,.84),rgba(222,236,231,.54))}
.structure-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:18px}
.structure-eyebrow{color:var(--sage2);font-size:10px;font-weight:850;letter-spacing:.12em;text-transform:uppercase;margin-bottom:7px}
.structure-action{display:inline-flex;padding:8px 12px;border-radius:999px;background:var(--mint);color:var(--sage2);font-size:11px;font-weight:850}
.structure-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.structure-tile{min-width:0;padding:16px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.58)}
.structure-tile span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
.structure-tile strong{display:block;margin-top:8px;font-size:16px;line-height:1.3;overflow-wrap:anywhere}
.structure-tile small{display:block;margin-top:6px;color:var(--muted);font-size:10px;line-height:1.45}
@media(max-width:980px){.structure-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:620px){.structure-heading{flex-direction:column}.structure-grid{grid-template-columns:1fr}}
"""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _label(value: Any) -> str:
    text = str(value or "NONE").replace("_", " ").title()
    return _escape(text)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
