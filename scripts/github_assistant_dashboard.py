from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    site_dir = root / "site"
    index_path = site_dir / "index.html"
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")
    latest = _load(site_dir / "latest.json", {})
    document = index_path.read_text(encoding="utf-8")

    document = document.replace(
        "BTC Adaptive Target–Stop Trader",
        "BTC Trade Assistant",
    )
    document = document.replace(
        "Aggressive structural entries with risk-scaled sizing and adaptive exits",
        "Direction-first forecasting with precision-gated paper positions",
    )
    document = document.replace(
        "Primary contract · aggressive entry, scaled capital risk",
        "Secondary contract · precision-gated paper position",
    )
    document = document.replace(
        "A valid structural event seeks a LONG or SHORT paper position. Qualification, economic edge and warnings change position size rather than silently vetoing the setup; hard data and structure failures still block entry.",
        "The one-hour forecast remains primary. A LONG or SHORT paper position is secondary and opens only when its locked precision meta gate accepts the structural setup.",
    )
    document = document.replace(
        "Qualification adjusts size; it does not veto a valid structure",
        "Position entry requires qualified precision-meta evidence",
    )
    document = document.replace(
        "Starts at 5R and adapts only from resolved trade outcomes",
        "New qualified positions use horizon-aligned MFE/MAE exits; existing open contracts stay immutable",
    )

    if "trade-assistant-panel" not in document:
        panel = _panel(latest)
        anchor = re.search(
            r'<section class="panel trade-lifecycle-panel">',
            document,
        )
        if anchor is not None:
            document = (
                document[: anchor.start()]
                + panel
                + "\n"
                + document[anchor.start() :]
            )
        else:
            ledger = re.search(
                r'<section class="[^"]*\bledger\b[^"]*">',
                document,
            )
            if ledger is not None:
                document = (
                    document[: ledger.start()]
                    + panel
                    + "\n"
                    + document[ledger.start() :]
                )
            elif "</main>" in document:
                document = document.replace(
                    "</main>", panel + "\n</main>", 1
                )
            else:
                raise RuntimeError("No trade-assistant insertion anchor")

    if "--assistant-good" not in document:
        document = document.replace(
            "</style>",
            _styles() + "\n</style>",
            1,
        )
    index_path.write_text(document, encoding="utf-8")
    return 0


def _panel(latest: dict[str, Any]) -> str:
    assistant = latest.get("trade_assistant")
    assistant = assistant if isinstance(assistant, dict) else {}
    pattern = latest.get("candle_pattern_memory")
    pattern = pattern if isinstance(pattern, dict) else {}
    adjustment = (
        latest.get("next_candle_forecast", {})
        if isinstance(latest.get("next_candle_forecast"), dict)
        else {}
    ).get("pattern_memory_adjustment")
    adjustment = adjustment if isinstance(adjustment, dict) else {}
    active = latest.get("active_trade")
    active = active if isinstance(active, dict) else None

    status = str(assistant.get("status") or "WAITING_FOR_META_EVIDENCE")
    selected = bool(assistant.get("selected", False))
    qualified = bool(assistant.get("qualified", False))
    p_take = _number(assistant.get("p_take"))
    p_false = _number(assistant.get("p_false"))
    reason = str(assistant.get("reason") or "NO_ACTIONABLE_POSITION_CANDIDATE")
    exit_profile = assistant.get("exit_profile")
    exit_profile = exit_profile if isinstance(exit_profile, dict) else {}
    horizon = assistant.get("horizon") or exit_profile.get("horizon_hours")

    live = pattern.get("live")
    live = live if isinstance(live, dict) else {}
    static = pattern.get("static")
    static = static if isinstance(static, dict) else {}
    memory_shrink = _number(adjustment.get("confidence_shrink_fraction")) or 0.0

    if active:
        position_state = "PRESERVED OPEN CONTRACT"
        position_class = "preserved"
        position_note = (
            "The existing open position keeps its original target, stop and expiry. "
            "The new precision architecture applies only to future entries."
        )
    elif qualified and selected:
        position_state = "META QUALIFIED"
        position_class = "qualified"
        position_note = (
            "The current structural candidate passed the locked precision gate."
        )
    elif status == "UNAVAILABLE" or not qualified:
        position_state = "EXPERIMENTAL / BLOCKED"
        position_class = "experimental"
        position_note = (
            "New positions stay blocked until a challenger produces a qualified "
            "precision meta head on locked chronological holdout data."
        )
    else:
        position_state = "SCANNING"
        position_class = "neutral"
        position_note = "No meta-qualified position candidate is active."

    return f'''
<section class="panel trade-assistant-panel">
  <div class="assistant-heading">
    <div>
      <div class="structure-eyebrow">Primary forecast · secondary position contract</div>
      <h2>Trade assistant</h2>
      <p class="sub">The next closed 1-hour candle remains the primary forecast. Position entries are precision-gated, fake-breakout-aware and paper-only until their locked holdout evidence qualifies.</p>
    </div>
    <span class="assistant-state {position_class}">{_escape(position_state)}</span>
  </div>
  <div class="assistant-grid">
    {_tile("1h forecast role", "PRIMARY", f"Pattern-memory confidence adjustment: {_percent(memory_shrink)}")}
    {_tile("Position role", "SECONDARY", position_note)}
    {_tile("Meta take probability", _percent(p_take), f"Status: {status.replace('_', ' ')}")}
    {_tile("Fake-breakout probability", _percent(p_false), reason.replace('_', ' '))}
    {_tile("Selected horizon", f"{_escape(horizon)}h" if horizon else "—", "Future exits align to the selected training horizon")}
    {_tile("Static fake memory", _memory_value(static), "Bloom membership is accepted only with exact repeated-count statistics")}
    {_tile("Live candle memory", _memory_value(live), "Resolved wrong next-candle contexts reduce future confidence without flipping direction")}
    {_tile("Position gate", "PASS" if qualified and selected else "HOLD", "Precision and positive net expectancy are required before a new paper entry")}
  </div>
</section>'''


def _memory_value(item: dict[str, Any]) -> str:
    if not item:
        return "—"
    count = item.get("count")
    bad_rate = _number(
        item.get("bad_rate", item.get("wrong_rate"))
    )
    hit = bool(item.get("bloom_hit", item.get("bad_pattern", False)))
    if count is None:
        return "ACTIVE" if hit else "CLEAR"
    suffix = " bad" if hit else " seen"
    return f"{_escape(count)} · {_percent(bad_rate)}{suffix}"


def _tile(label: str, value: str, note: str) -> str:
    return (
        '<div class="assistant-tile">'
        f'<small>{_escape(label)}</small>'
        f'<strong>{value}</strong>'
        f'<p>{_escape(note)}</p>'
        "</div>"
    )


def _styles() -> str:
    return r'''
:root{--assistant-good:var(--ok,#4d8b76);--assistant-warn:var(--wait,#9a865d);--assistant-bad:var(--bad,#bd726e)}
.trade-assistant-panel{margin-top:18px}.assistant-heading{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}.assistant-heading h2{margin:5px 0 0}.assistant-state{display:inline-flex;align-items:center;justify-content:center;padding:9px 12px;border-radius:999px;font-size:10px;font-weight:850;letter-spacing:.035em;white-space:nowrap;border:1px solid var(--line)}.assistant-state.qualified{color:var(--assistant-good);background:color-mix(in srgb,var(--assistant-good) 10%,transparent)}.assistant-state.experimental{color:var(--assistant-bad);background:color-mix(in srgb,var(--assistant-bad) 9%,transparent)}.assistant-state.preserved{color:var(--assistant-warn);background:color-mix(in srgb,var(--assistant-warn) 10%,transparent)}.assistant-state.neutral{color:var(--muted)}.assistant-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:18px}.assistant-tile{min-width:0;padding:15px;border:1px solid var(--line);border-radius:18px;background:color-mix(in srgb,var(--paper) 84%,transparent)}.assistant-tile small{display:block;color:var(--muted);font-size:10px}.assistant-tile strong{display:block;margin-top:7px;font-size:15px;overflow-wrap:anywhere}.assistant-tile p{margin:6px 0 0;color:var(--muted);font-size:10px;line-height:1.45}@media(max-width:900px){.assistant-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:620px){.assistant-heading{flex-direction:column}.assistant-grid{grid-template-columns:1fr}.assistant-state{white-space:normal;text-align:center}}
'''


def _load(path: Path, fallback: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except Exception:
        return fallback


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
