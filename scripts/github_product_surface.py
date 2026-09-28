from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

MARKER = 'data-product-surface="v1"'


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    site = root / "site"
    index = site / "index.html"
    if not index.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")

    latest = _load(site / "latest.json", {})
    doc = index.read_text(encoding="utf-8")

    replacements = {
        "BTC Adaptive Model Trader": "BTC Signal Desk",
        "Adaptive BTC paper positions with a calibrated secondary next-close range":
            "BTC 1H market dashboard",
        "Adaptive BTC paper positions with an exact secondary next-close forecast":
            "Real-time BTC direction, range and position intelligence for the 1-hour market.",
        "Adaptive BTC paper positions with a secondary next-close forecast":
            "Real-time BTC direction, range and position intelligence for the 1-hour market.",
        "Adaptive next-candle BTC direction and price forecast":
            "BTC 1H market dashboard",
        "Next closed 1-hour candle": "BTC · 1H MARKET OUTLOOK",
        "Likely next-close range": "Projected 1H range",
        "Range-only forecast · no exact close is published":
            "Expected range for the next hourly close",
        "Target closes": "Next close",
        "Direction accuracy": "Signal accuracy",
        "Interval coverage": "Range hit rate",
        "Decision": "Position",
        "Trade gates remain independent from forecast scoring": "Current strategy stance",
        "Market regime": "Market structure",
        "Primary contract · aggressive entry, scaled capital risk": "POSITION PLAN",
        "Adaptive paper-trade lifecycle": "Position plan",
        "A valid structural event seeks a LONG or SHORT paper position. Qualification, economic edge and warnings change position size rather than silently vetoing the setup; hard data and structure failures still block entry.":
            "Entry, target and stop levels derived from the current 1H signal and risk profile.",
        "Risk allocation": "Allocated risk",
        "Dynamic paper-account risk assigned to this setup": "Capital allocated to this setup",
        "Risk score": "Setup score",
        "Qualification": "Signal quality",
        "Adaptive reward": "Reward target",
        "Starts at 5R and adapts only from resolved trade outcomes":
            "Target reward relative to defined risk",
        "After stress fees and slippage": "Estimated after fees and slippage",
        "Risk including stress execution cost": "Estimated including execution cost",
        "Expected value": "Expected P/L",
        "Probability-weighted paper-trade value": "Probability-weighted outcome estimate",
        "Suggested paper margin and leverage": "Position margin and leverage",
        "Holding contract": "Max holding time",
        "Expires:": "Exit by:",
        "Online samples": "Strategy samples",
        "Only resolved target/stop/time-exit positions":
            "Closed positions used for performance calibration",
        "Resolved positions": "Closed positions",
        "Learned performance": "Strategy P/L",
        "Cumulative realized paper PnL and average R":
            "Cumulative realized P/L and average R",
        "LONG / SHORT position ledger": "Position history",
        "Only actual suggested paper positions are listed. Open P&amp;L is marked to the latest closed candle; closed P&amp;L is immutable and realized.":
            "Open positions are marked to the latest closed candle. Closed results remain fixed.",
        "No LONG or SHORT position has been opened yet.": "No position has been opened yet.",
        "Forecast ledger": "Signal history",
        "Immutable results resolved only after each target candle closes":
            "Resolved hourly signals and their outcomes",
        "Source candle": "Time",
        "Source close": "BTC close",
        "Model range": "Projected range",
        "Direction result": "Signal result",
        "Interval result": "Range result",
        "Price path and next forecast": "BTC price & outlook",
        "Recent BTC closes with next-candle direction and calibrated range":
            "Recent BTC closes, signal outcomes and projected 1H range",
        "Forecast range": "Projected range",
        "<small>FORECAST CREATED</small>": "<small>SIGNAL TIME</small>",
        "<small>SOURCE CLOSED</small>": "<small>REFERENCE CLOSE</small>",
        "<small>TARGET OPEN</small>": "<small>NEXT CANDLE OPEN</small>",
        "<small>TARGET CLOSE</small>": "<small>NEXT CANDLE CLOSE</small>",
        "<small>TIME TO CLOSE</small>": "<small>TIME REMAINING</small>",
        "<small>CONTRACT</small>": "<small>STATUS</small>",
        "QUALIFIED": "CONFIRMED",
        "RISK-SCALED": "SCALED",
    }
    for old, new in replacements.items():
        doc = doc.replace(old, new)

    doc = re.sub(r"<title>.*?</title>", "<title>BTC Signal Desk</title>", doc, count=1)
    doc = re.sub(
        r'<p class="copy">The model expects the next close to finish <strong>(.*?)</strong>\. .*?</p>',
        r'<p class="copy">Next-hour bias: <strong>\1</strong>. Confidence and projected range update with each closed candle.</p>',
        doc,
        count=1,
        flags=re.DOTALL,
    )

    for label in ("Calibration", "Forecast source", "Evaluation"):
        doc = re.sub(
            rf'<div class="meta"><span>{label}</span>.*?</div>',
            "",
            doc,
            count=1,
            flags=re.DOTALL,
        )

    doc = re.sub(
        r'<section class="panel trade-lifecycle-panel">.*?</section>',
        _position_plan(latest),
        doc,
        count=1,
        flags=re.DOTALL,
    )

    doc = re.sub(
        r'<aside class="panel"><h2>Adaptive learning</h2>.*?</aside>',
        _market_snapshot(latest),
        doc,
        count=1,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'<section class="panel trade-assistant-panel">.*?</section>',
        _signal_confirmation(latest),
        doc,
        count=1,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'<section class="panel boundary-memory-panel">.*?</section>',
        "",
        doc,
        count=1,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'<section class="panel resilience-panel">.*?</section>',
        "",
        doc,
        count=1,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'<footer>.*?</footer>',
        '<footer><span>BTC Signal Desk · 1H Bitcoin market intelligence.</span>'
        '<span>Informational use only.</span></footer>',
        doc,
        count=1,
        flags=re.DOTALL,
    )

    doc = re.sub(
        r"(\d+) correct across (\d+) scored v2 forecasts",
        r"\1 correct across \2 resolved hourly signals",
        doc,
    )
    doc = re.sub(
        r"(\d+) closes inside (\d+) intervals",
        r"\1 closes inside \2 projected ranges",
        doc,
    )
    doc = doc.replace(" · Forecast ", " · Signal ")
    doc = doc.replace(" calibrated range", " confidence range")
    doc = re.sub(
        r'<p class="sub"[^>]*>.*?</p>',
        "",
        doc,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'<p class="copy"[^>]*>.*?</p>',
        "",
        doc,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'<div class="label" style="margin-top:6px">.*?</div>',
        "",
        doc,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'(<article class="metric">.*?<strong>.*?</strong>)<small>.*?</small>(</article>)',
        r'\1\2',
        doc,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'(<div class="trade-lifecycle-tile">.*?<strong>.*?</strong>)<small>.*?</small>(</div>)',
        r'\1\2',
        doc,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'<footer>.*?</footer>',
        "",
        doc,
        count=1,
        flags=re.DOTALL,
    )
    doc = re.sub(
        r'(<small>STATUS</small><strong>).*?(</strong>)',
        r'\1ACTIVE\2',
        doc,
        count=1,
        flags=re.DOTALL,
    )

    if MARKER not in doc:
        doc = doc.replace("<body", f'<body {MARKER}', 1)
    index.write_text(doc, encoding="utf-8")
    return 0


def _position_plan(latest: dict[str, Any]) -> str:
    plan = latest.get("trade_plan")
    plan = plan if isinstance(plan, dict) else {}
    active = latest.get("active_trade")
    if isinstance(active, dict) and isinstance(active.get("trade_plan"), dict):
        plan = active["trade_plan"]

    direction = str(latest.get("action") or plan.get("direction") or "WAIT").upper()
    entry = _price(plan.get("entry_reference"))
    target = _price(plan.get("target_price"))
    stop = _price(plan.get("stop_price"))
    risk_budget = _money(plan.get("risk_budget_usd"))
    risk_fraction = _percent(plan.get("risk_fraction"))
    setup_score = _percent(plan.get("risk_score"))
    leverage = _multiple(plan.get("suggested_leverage"))
    margin = _money(plan.get("margin_required_usd"))
    expected = _money(plan.get("expected_value_usd"))
    reward = _number(plan.get("risk_reward"))
    holding = plan.get("maximum_holding_hours")
    status = str(plan.get("status") or latest.get("action") or "WAIT").replace("_", " ").upper()

    def tile(label: str, value: str, note: str) -> str:
        return (
            '<div class="trade-lifecycle-tile">'
            f'<span>{html.escape(label)}</span>'
            f'<strong>{html.escape(value)}</strong>'
            f'<small>{html.escape(note)}</small></div>'
        )

    grid = (
        tile("Status", f"{direction} · {status}", "Current 1H position stance")
        + tile("Allocated risk", f"{risk_budget} · {risk_fraction}", "Capital allocated to this setup")
        + tile("Setup score", setup_score, "Composite quality score for the current setup")
        + tile("Risk / reward", "—" if reward is None else f"{reward:.2f}R", "Target reward relative to defined risk")
        + tile("Expected P/L", expected, "Probability-weighted outcome estimate")
        + tile("Margin / leverage", f"{margin} · {leverage}", "Position margin and leverage")
        + tile("Max holding time", "—" if holding in (None, "") else f"{holding}h", "Maximum planned holding window")
    )
    return f'''<section class="panel trade-lifecycle-panel">
  <div class="trade-lifecycle-heading">
    <div>
      <div class="structure-eyebrow">POSITION PLAN</div>
      <h2>Position plan</h2>
      <p class="sub">Entry, target and stop levels for the current BTC 1H setup.</p>
    </div>
    <span class="trade-lifecycle-state neutral">{html.escape(status)}</span>
  </div>
  <div class="trade-route">
    <div><small>ENTRY</small><strong>{entry}</strong></div>
    <span>→</span>
    <div class="target"><small>TARGET</small><strong>{target}</strong></div>
    <span>or</span>
    <div class="stop"><small>STOP</small><strong>{stop}</strong></div>
  </div>
  <div class="trade-lifecycle-grid">{grid}</div>
</section>'''


def _market_snapshot(latest: dict[str, Any]) -> str:
    contract = latest.get("next_candle_forecast")
    contract = contract if isinstance(contract, dict) else {}
    direction = str(
        contract.get("direction")
        or latest.get("next_candle_direction")
        or "—"
    ).upper()
    confidence = _percent(
        contract.get("direction_probability", latest.get("next_candle_confidence"))
    )
    rows = (
        ("BTC close", _price(latest.get("market_price", latest.get("price")))),
        ("1H bias", direction),
        ("Confidence", confidence),
        ("Structure", _label(latest.get("regime"))),
        ("Position", _label(latest.get("action"))),
    )
    items = "".join(
        '<div class="learn"><div class="learn-head">'
        f'<span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>'
        "</div></div>"
        for label, value in rows
    )
    return (
        '<aside class="panel market-snapshot-panel">'
        '<h2>Market snapshot</h2>'
        '<p class="sub">The essentials for the current BTC 1H setup.</p>'
        f'<div class="learning">{items}</div></aside>'
    )


def _signal_confirmation(latest: dict[str, Any]) -> str:
    assistant = latest.get("trade_assistant")
    assistant = assistant if isinstance(assistant, dict) else {}
    active = latest.get("active_trade")
    selected = bool(assistant.get("selected", False))
    qualified = bool(assistant.get("qualified", False))

    if isinstance(active, dict):
        state = "POSITION OPEN"
        note = "Managing the active position with its original target, stop and exit window."
    elif qualified and selected:
        state = "CONFIRMED"
        note = "Current setup has additional confirmation from recent pattern behavior."
    elif assistant.get("status") == "UNAVAILABLE":
        state = "SIGNAL ONLY"
        note = "Position sizing follows the core signal while confirmation data is limited."
    else:
        state = "WATCHING"
        note = "Waiting for stronger setup confirmation."

    tiles = (
        _tile("Status", state, note)
        + _tile(
            "Setup quality",
            _percent(assistant.get("p_take")),
            "Estimated probability the setup remains actionable.",
        )
        + _tile(
            "False-signal risk",
            _percent(assistant.get("p_false")),
            "Estimated probability the setup loses confirmation.",
        )
    )
    return (
        '<section class="panel trade-assistant-panel">'
        '<div class="assistant-heading"><div>'
        '<div class="structure-eyebrow">POSITION READINESS</div>'
        '<h2>Signal confirmation</h2>'
        '<p class="sub">A compact second check for setup quality before risk is allocated.</p>'
        '</div>'
        f'<span class="assistant-state neutral">{html.escape(state)}</span></div>'
        f'<div class="assistant-grid">{tiles}</div></section>'
    )


def _tile(label: str, value: str, note: str) -> str:
    return (
        '<div class="assistant-tile">'
        f'<small>{html.escape(label)}</small>'
        f'<strong>{html.escape(value)}</strong>'
        '</div>'
    )


def _load(path: Path, default: Any) -> Any:
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


def _percent(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number * 100:.1f}%"


def _price(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else "$" + f"{number:,.2f}"


def _money(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else "$" + f"{number:,.2f}"


def _multiple(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.0f}×"


def _label(value: Any) -> str:
    text = str(value or "—").replace("_", " ").strip()
    return text.title() if text != "—" else text


if __name__ == "__main__":
    raise SystemExit(main())
