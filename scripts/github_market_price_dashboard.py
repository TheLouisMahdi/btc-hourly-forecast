from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

MARKER = 'data-market-price="latest-close-v1"'


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    state_dir = root / ".github_state"
    site_dir = root / "site"
    index_path = site_dir / "index.html"
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")

    document = index_path.read_text(encoding="utf-8")
    if MARKER in document:
        return 0

    latest = _load_json(state_dir / "latest.json", {})
    candles = _load_json(state_dir / "chart_candles.json", [])
    value, timestamp = _latest_market_price(candles, latest)

    document = document.replace("</style>", _styles() + "\n</style>", 1)
    card = _card(value, timestamp)
    marker = '<aside class="forecast-card">'
    if marker in document:
        document = document.replace(marker, marker + "\n" + card, 1)
    elif "</nav>" in document:
        document = document.replace("</nav>", card + "\n</nav>", 1)
    else:
        raise RuntimeError("No dashboard anchor is available for market price")

    index_path.write_text(document, encoding="utf-8")
    return 0


def _latest_market_price(
    candles: Any,
    latest: dict[str, Any],
) -> tuple[float | None, str | None]:
    if isinstance(candles, list):
        for item in reversed(candles):
            if not isinstance(item, dict):
                continue
            value = _finite(item.get("close"))
            if value is not None:
                return value, _text(item.get("open_time"))

    value = _finite(latest.get("market_price"))
    if value is not None:
        return value, _text(latest.get("market_price_time"))

    value = _finite(latest.get("price"))
    if value is not None:
        return value, _text(latest.get("candle_time"))

    contract = latest.get("next_candle_forecast")
    if isinstance(contract, dict):
        value = _finite(contract.get("reference_close"))
        if value is not None:
            return value, _text(contract.get("source_open_time"))
    return None, None


def _card(value: float | None, timestamp: str | None) -> str:
    display = "—" if value is None else f"${value:,.2f}"
    time_text = _compact_time(timestamp)
    return (
        f'<div class="market-price-card" {MARKER}>'
        '<span>Latest closed BTC</span>'
        f'<strong>{html.escape(display)}</strong>'
        f'<small>{html.escape(time_text)}</small>'
        "</div>"
    )


def _styles() -> str:
    return """
.market-price-card{margin-bottom:16px;padding:14px 16px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(135deg,var(--mint),rgba(255,255,255,.62))}
.market-price-card span{display:block;color:var(--muted);font-size:10px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
.market-price-card strong{display:block;margin-top:5px;font-size:28px;line-height:1.1;letter-spacing:-.04em}
.market-price-card small{display:block;margin-top:5px;color:var(--muted);font-size:10px}
:root[data-theme="dark"] .market-price-card{background:linear-gradient(135deg,rgba(69,116,105,.28),rgba(24,39,36,.72))}
"""


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _compact_time(value: str | None) -> str:
    if not value:
        return "Latest available closed candle"
    try:
        from datetime import datetime

        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return timestamp.strftime("%b %d · %H:%M UTC")
    except Exception:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
