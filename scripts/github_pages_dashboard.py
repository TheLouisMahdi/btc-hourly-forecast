from __future__ import annotations

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
    index_path = site_dir / "index.html"
    document = index_path.read_text(encoding="utf-8")

    document = document.replace(
        "BTC Next-Candle Forecast",
        "BTC Adaptive Model Trader",
    )
    document = document.replace(
        "Direction-first forecasting with adaptive price learning",
        "Model-first paper trading with disciplined risk control",
    )
    document = document.replace(
        "Adaptive next-candle BTC direction and price forecast",
        "Adaptive BTC paper positions with a secondary next-close forecast",
    )
    document = document.replace(
        "</style>",
        _extra_styles() + "\n</style>",
        1,
    )
    document = re.sub(
        r'<div class="health"><i></i>.*?</div>',
        _health_summary(latest),
        document,
        count=1,
    )
    index_path.write_text(document, encoding="utf-8")

    for component in (
        github_boundary_dashboard,
        github_trade_dashboard,
        github_timing_dashboard,
    ):
        status = component.main()
        if status != 0:
            return status
    return 0


def _health_summary(latest: dict[str, Any]) -> str:
    pipeline_ok = latest.get("run_status") == "OK"
    health = latest.get("data_health")
    health = health if isinstance(health, dict) else {}
    data_ok = bool(
        health.get("candles_ok", False)
        and health.get("quote_ok", False)
        and not health.get("provider_mismatch", False)
    )
    healthy = pipeline_ok and data_ok
    status = "SYSTEM HEALTHY" if healthy else "SYSTEM CHECK"
    action = str(latest.get("action") or "WAIT").replace("_", " ")
    return (
        f'<div class="health-summary {"ok" if healthy else "warn"}">'
        '<i></i>'
        f'<strong>{status}</strong>'
        '<span class="health-separator">·</span>'
        f'<span>{action}</span>'
        '<span class="health-separator">·</span>'
        '<span>PAPER</span>'
        "</div>"
    )


def _extra_styles() -> str:
    return """
.health-summary{display:flex;align-items:center;gap:7px;padding:9px 12px;border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.72);font-size:10px;letter-spacing:.03em;white-space:nowrap}
.health-summary i{width:7px;height:7px;border-radius:50%;background:var(--ok)}
.health-summary.warn i{background:var(--wait)}
.health-summary strong{font-size:10px}
.health-summary span{color:var(--muted)}
.health-separator{opacity:.55}
@media(max-width:620px){.health-summary{font-size:9px;padding:8px 10px}.health-summary strong{font-size:9px}}
"""


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


if __name__ == "__main__":
    raise SystemExit(main())
