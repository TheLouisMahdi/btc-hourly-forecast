from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    index_path = root / "site" / "index.html"
    latest = _load(root / ".github_state" / "latest.json", {})
    contract = latest.get("next_candle_forecast")
    if not index_path.exists() or not isinstance(contract, dict):
        return 0
    document = index_path.read_text(encoding="utf-8")
    document = document.replace(
        "Next closed 1-hour candle · structural context",
        "Secondary exact next-close forecast",
    )
    document = document.replace(
        "Next closed 1-hour candle",
        "Secondary exact next-close forecast",
    )
    document = document.replace(
        "Model-estimated next close",
        "Estimate for the exact target candle close",
    )
    document = document.replace("</style>", _styles() + "\n</style>", 1)
    timing = _timing_strip(contract)
    if timing and "exact-candle-timing" not in document:
        document = document.replace("</aside>", timing + "\n</aside>", 1)
    index_path.write_text(document, encoding="utf-8")
    return 0


def _timing_strip(contract: dict[str, Any]) -> str:
    created = contract.get("forecast_created_at")
    source_close = contract.get("source_close_time")
    target_open = contract.get("target_open_time")
    target_close = contract.get("target_close_time")
    if not all((created, source_close, target_open, target_close)):
        return ""
    horizon = _duration(contract.get("forecast_horizon_seconds"))
    status = str(contract.get("timing_status") or "EXACT_NEXT_CLOSED_CANDLE")
    return f'''
<div class="exact-candle-timing">
  <div><small>FORECAST CREATED</small><strong>{_escape(_time(created))}</strong></div>
  <div><small>SOURCE CLOSED</small><strong>{_escape(_time(source_close))}</strong></div>
  <div><small>TARGET OPEN</small><strong>{_escape(_time(target_open))}</strong></div>
  <div><small>TARGET CLOSE</small><strong>{_escape(_time(target_close))}</strong></div>
  <div><small>TIME TO CLOSE</small><strong>{_escape(horizon)}</strong></div>
  <div><small>CONTRACT</small><strong>{_escape(status.replace("_", " "))}</strong></div>
</div>'''


def _styles() -> str:
    return r'''
.exact-candle-timing{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.exact-candle-timing div{min-width:0;padding:8px 9px;border-radius:11px;background:rgba(255,255,255,.56);border:1px solid var(--line)}
.exact-candle-timing small{display:block;color:var(--muted);font-size:7px;letter-spacing:.07em}.exact-candle-timing strong{display:block;margin-top:3px;font-size:9px;line-height:1.35;overflow-wrap:anywhere}
@media(max-width:720px){.exact-candle-timing{grid-template-columns:repeat(2,minmax(0,1fr))}}
'''


def _duration(value: Any) -> str:
    try:
        seconds = max(0, int(float(value)))
    except (TypeError, ValueError):
        return "—"
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def _time(value: Any) -> str:
    try:
        timestamp = pd.Timestamp(value)
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        return timestamp.strftime("%H:%M:%S UTC")
    except Exception:
        return str(value)


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
