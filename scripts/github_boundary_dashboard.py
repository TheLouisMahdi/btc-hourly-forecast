from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    site_dir = root / "site"
    index_path = site_dir / "index.html"
    latest_path = site_dir / "latest.json"
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")
    latest = _load(latest_path)
    document = index_path.read_text(encoding="utf-8")
    document = document.replace(
        "</style>",
        _styles() + "\n</style>",
        1,
    )
    marker = '<section class="panel economic-panel">'
    panel = _panel(latest)
    if marker in document:
        document = document.replace(marker, panel + "\n" + marker, 1)
    else:
        fallback = '<section class="panel ledger">'
        document = document.replace(fallback, panel + "\n" + fallback, 1)
    index_path.write_text(document, encoding="utf-8")
    return 0


def _panel(latest: dict[str, Any]) -> str:
    memory = latest.get("boundary_memory")
    memory = memory if isinstance(memory, dict) else {}
    side = str(memory.get("boundary_side") or "NONE")
    status = str(memory.get("status") or "UNAVAILABLE")
    selected_horizon = memory.get("selected_horizon")
    horizons = memory.get("horizons")
    horizons = horizons if isinstance(horizons, dict) else {}
    selected = horizons.get(str(selected_horizon), {}) if selected_horizon else {}
    selected = selected if isinstance(selected, dict) else {}
    p_break = _number(selected.get("p_break", memory.get("p_break")))
    p_bad = _number(
        selected.get("p_unprofitable", memory.get("p_unprofitable"))
    )
    front = bool(selected.get("front_memory_hit", False))
    backup = bool(selected.get("backup_memory_hit", False))
    veto = bool(
        selected.get(
            "negative_memory_veto",
            memory.get("negative_memory_veto", False),
        )
    )
    qualified = bool(selected.get("qualified", memory.get("qualified", False)))
    loaded = bool(latest.get("negative_memory_loaded", False))
    state_class, verdict = _state(loaded, qualified, veto, status)
    note = {
        "safe": "The selected side and horizon passed the locked memory holdout gate.",
        "veto": "A recurring or learned hard-negative pattern vetoed this candidate.",
        "shadow": "Memory is loaded but this context is not qualified; output remains diagnostic.",
        "unavailable": "No compatible negative-memory artifact is active for this record.",
    }[state_class]
    return f'''
<section class="panel boundary-memory-panel">
  <div class="boundary-memory-heading">
    <div>
      <div class="structure-eyebrow">Sandwiched negative memory</div>
      <h2>Support / resistance risk memory</h2>
      <p class="sub">{_escape(note)} Front and backup Bloom filters identify recurring fingerprints; the learned middle estimates boundary-break and no-profit risk.</p>
    </div>
    <span class="boundary-memory-state {state_class}">{_escape(verdict)}</span>
  </div>
  <div class="boundary-memory-grid">
    {_tile("Context", _label(side), status)}
    {_tile("Boundary level", _price(memory.get("boundary_level")), "Structural level used by the side-specific head")}
    {_tile("Distance", _atr(memory.get("boundary_distance_atr")), "Current distance from support or resistance")}
    {_tile("Memory horizon", f"{selected_horizon}h" if selected_horizon else "—", "Chronological side-specific head")}
    {_tile("Break probability", _percent(p_break), "Probability the active level breaks in the modeled direction")}
    {_tile("No-profit risk", _percent(p_bad), "Probability the setup fails to produce stress-net profit")}
    {_tile("Front Bloom", "HIT" if front else "CLEAR", "Recurring historically bad fingerprint")}
    {_tile("Backup Bloom", "HIT" if backup else "CLEAR", "Hard negative missed by the learned middle")}
  </div>
</section>'''


def _state(
    loaded: bool,
    qualified: bool,
    veto: bool,
    status: str,
) -> tuple[str, str]:
    if not loaded or status.upper() == "UNAVAILABLE":
        return "unavailable", "UNAVAILABLE"
    if veto:
        return "veto", "VETO"
    if qualified:
        return "safe", "PASS"
    return "shadow", "SHADOW"


def _tile(title: str, value: str, note: str) -> str:
    return f'''
<div class="boundary-memory-tile">
  <span>{_escape(title)}</span>
  <strong>{_escape(value)}</strong>
  <small>{_escape(note)}</small>
</div>'''


def _styles() -> str:
    return r'''
.boundary-memory-panel{position:relative;margin-top:18px;overflow:hidden;background:linear-gradient(135deg,rgba(255,255,255,.9),rgba(236,234,245,.58))}
.boundary-memory-panel:before{content:"";position:absolute;inset:-40% auto auto -15%;width:320px;height:320px;border-radius:50%;background:radial-gradient(circle,rgba(111,155,145,.12),transparent 68%);pointer-events:none}
.boundary-memory-heading{position:relative;display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:20px}.boundary-memory-heading h2{margin:5px 0 8px;font-size:clamp(1.35rem,3vw,2rem)}
.boundary-memory-state{display:inline-flex;align-items:center;justify-content:center;min-width:112px;padding:10px 14px;border-radius:999px;font-size:.78rem;font-weight:850;letter-spacing:.08em;border:1px solid var(--line)}
.boundary-memory-state.safe{background:rgba(77,139,118,.12);color:var(--ok)}.boundary-memory-state.veto{background:rgba(189,114,110,.12);color:var(--bad)}.boundary-memory-state.shadow{background:rgba(154,134,93,.12);color:var(--wait)}.boundary-memory-state.unavailable{background:rgba(116,131,127,.09);color:var(--muted)}
.boundary-memory-grid{position:relative;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.boundary-memory-tile{padding:16px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.58);min-height:118px;display:flex;flex-direction:column;gap:8px}.boundary-memory-tile span{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}.boundary-memory-tile strong{font-size:1rem;overflow-wrap:anywhere}.boundary-memory-tile small{color:var(--muted);font-size:.68rem;line-height:1.45}
@media(max-width:850px){.boundary-memory-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){.boundary-memory-heading{flex-direction:column}.boundary-memory-grid{grid-template-columns:1fr}.boundary-memory-state{align-self:flex-start}}
'''


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


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
    return "—" if number is None else f"${number:,.2f}"


def _atr(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.2f} ATR"


def _label(value: Any) -> str:
    return str(value or "NONE").replace("_", " ").title()


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
