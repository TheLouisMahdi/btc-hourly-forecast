from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

MARKER = 'data-resilience-dashboard="v1"'


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    site_dir = root / "site"
    index_path = site_dir / "index.html"
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")

    document = index_path.read_text(encoding="utf-8")
    latest = _load_json(site_dir / "latest.json", {})
    history = _load_json(site_dir / "history.json", [])
    history = history if isinstance(history, list) else []

    chart_pattern = re.compile(
        r'<svg class="chart".*?</svg>\s*<div class="legend">.*?</div>',
        flags=re.DOTALL,
    )
    document = chart_pattern.sub(_time_aware_chart(history), document, count=1)
    panel = _resilience_panel(latest, history)
    ledger = '<section class="panel ledger">'
    if ledger in document and "Data continuity & learning safety" not in document:
        document = document.replace(ledger, panel + "\n" + ledger, 1)
    document = document.replace("</style>", _styles() + "\n</style>", 1)
    if MARKER not in document:
        document = re.sub(
            r"<body([^>]*)>",
            lambda match: f'<body{match.group(1)} {MARKER}>',
            document,
            count=1,
        )
    index_path.write_text(document, encoding="utf-8")
    return 0


def _time_aware_chart(history: list[dict[str, Any]]) -> str:
    points: list[dict[str, Any]] = []
    for item in history[-168:]:
        price = _number(item.get("price"))
        timestamp = _timestamp(
            item.get("candle_time") or item.get("run_finished_at")
        )
        if price is None or timestamp is None:
            continue
        points.append({"time": timestamp, "price": price, "item": item})
    points.sort(key=lambda point: point["time"])
    if len(points) < 2:
        return '<div class="empty">More forecast history is required.</div>'

    width, height, pad_x, pad_y = 1120, 350, 52, 32
    latest_contract = points[-1]["item"].get("next_candle_forecast")
    latest_contract = (
        latest_contract if isinstance(latest_contract, dict) else {}
    )
    scale = [point["price"] for point in points]
    for key in ("likely_close_low", "likely_close_high"):
        value = _number(latest_contract.get(key))
        if value is not None:
            scale.append(value)
    low, high = min(scale), max(scale)
    margin = max((high - low) * 0.15, 1.0)
    low -= margin
    high += margin
    start = points[0]["time"]
    end = max(points[-1]["time"], start + pd.Timedelta(hours=1))
    duration = max((end - start).total_seconds(), 3600.0)

    def xy(timestamp: pd.Timestamp, value: float) -> tuple[float, float]:
        fraction = (timestamp - start).total_seconds() / duration
        x = pad_x + fraction * (width - 2 * pad_x - 64)
        y = height - pad_y - (value - low) * (height - 2 * pad_y) / max(
            high - low, 1e-9
        )
        return x, y

    segments: list[list[dict[str, Any]]] = [[points[0]]]
    gaps: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for previous, current in zip(points, points[1:]):
        hours = (current["time"] - previous["time"]).total_seconds() / 3600.0
        if hours > 1.5:
            gaps.append((previous, current, max(int(round(hours)) - 1, 1)))
            segments.append([current])
        else:
            segments[-1].append(current)

    paths: list[str] = []
    for segment in segments:
        if len(segment) < 2:
            continue
        commands = []
        for index, point in enumerate(segment):
            x, y = xy(point["time"], point["price"])
            commands.append(f'{"M" if index == 0 else "L"} {x:.2f} {y:.2f}')
        paths.append(f'<path d="{" ".join(commands)}" class="line"/>')

    grid: list[str] = []
    labels: list[str] = []
    for index in range(4):
        ratio = index / 3
        y = pad_y + ratio * (height - 2 * pad_y)
        label = high - ratio * (high - low)
        grid.append(
            f'<line x1="{pad_x}" y1="{y:.2f}" x2="{width-pad_x}" '
            f'y2="{y:.2f}" class="grid-line"/>'
        )
        labels.append(
            f'<text x="6" y="{y+4:.2f}" class="axis">${label:,.0f}</text>'
        )

    gap_marks: list[str] = []
    for previous, current, missing in gaps:
        middle = previous["time"] + (current["time"] - previous["time"]) / 2
        x, _ = xy(middle, (low + high) / 2)
        gap_marks.append(
            f'<line x1="{x:.2f}" y1="{pad_y}" x2="{x:.2f}" '
            f'y2="{height-pad_y}" class="gap-line"/>'
            f'<text x="{x+5:.2f}" y="{pad_y+13}" class="gap-label">'
            f'{missing}h not published</text>'
        )

    markers: list[str] = []
    for point in points:
        contract = point["item"].get("next_candle_forecast")
        contract = contract if isinstance(contract, dict) else {}
        direction = str(contract.get("direction") or "").upper()
        if direction not in {"UP", "DOWN"}:
            continue
        result = str(point["item"].get("direction_result") or "PENDING")
        marker_class = {
            "DIRECTION_CORRECT": "correct",
            "DIRECTION_WRONG": "wrong",
            "PENDING": "pending-dot",
        }.get(result, "pending-dot")
        x, y = xy(point["time"], point["price"])
        symbol = "↑" if direction == "UP" else "↓"
        markers.append(
            f'<g transform="translate({x:.2f},{y-12:.2f})"><circle r="11" '
            f'class="dot {marker_class}"/><text y="4" text-anchor="middle" '
            f'class="symbol">{symbol}</text></g>'
        )

    band = ""
    band_low = _number(latest_contract.get("likely_close_low"))
    band_high = _number(latest_contract.get("likely_close_high"))
    if band_low is not None and band_high is not None:
        last_x, _ = xy(points[-1]["time"], points[-1]["price"])
        top = xy(points[-1]["time"], max(band_low, band_high))[1]
        bottom = xy(points[-1]["time"], min(band_low, band_high))[1]
        band = (
            f'<rect x="{last_x:.2f}" y="{top:.2f}" '
            f'width="{max(4.0, width-pad_x-last_x):.2f}" '
            f'height="{max(2.0, bottom-top):.2f}" class="band"/>'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Time-aware BTC forecast history; lines break where hourly '
        'workflow records were not published">'
        + "".join(grid)
        + "".join(labels)
        + band
        + "".join(paths)
        + "".join(gap_marks)
        + "".join(markers)
        + "</svg>"
        + '<div class="legend">'
        + '<span><i style="background:var(--ok)"></i>Correct direction</span>'
        + '<span><i style="background:var(--bad)"></i>Wrong direction</span>'
        + '<span><i style="background:var(--wait)"></i>Pending</span>'
        + '<span><i class="gap-key"></i>Missing workflow record; not interpolated</span>'
        + '<span><i style="background:rgba(143,138,184,.35);border-radius:3px;width:18px"></i>Calibrated range</span>'
        + "</div>"
    )


def _resilience_panel(
    latest: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    market = latest.get("market_refresh")
    market = market if isinstance(market, dict) else {}
    freshness = market.get("freshness")
    freshness = freshness if isinstance(freshness, dict) else {}
    continuity = market.get("continuity")
    continuity = continuity if isinstance(continuity, dict) else {}
    policy = latest.get("retrain_policy")
    policy = policy if isinstance(policy, dict) else {}
    price_model = latest.get("price_forecast_model")
    price_model = price_model if isinstance(price_model, dict) else {}

    missing_records, largest_gap = _history_gap_summary(history)
    lag = _number(freshness.get("lag_hours"))
    fresh = freshness.get("fresh") is True
    attempts = market.get("attempts")
    attempts = attempts if isinstance(attempts, list) else []
    selected_window = _number(market.get("selected_lookback_days"))
    contiguous_rows = _number(continuity.get("selected_rows"))
    policy_status = str(policy.get("status") or "NOT EVALUATED")
    policy_reason = str(policy.get("reason") or "—")
    online_samples = _number(price_model.get("samples_seen"))
    last_learned = _compact_time(price_model.get("last_trained_open_time"))

    data_status = "FRESH" if fresh else "PROTECTED / STALE"
    data_note = (
        "The latest closed candle passed the freshness gate."
        if fresh
        else "Forecasting and online learning are blocked when recent candles are stale."
    )
    recovery_note = (
        f"Same-provider recovery selected {selected_window:g} days after {len(attempts)} attempt(s)."
        if selected_window is not None
        else "No successful recovery window is recorded."
    )
    return f'''
<section class="panel resilience-panel">
  <div class="resilience-heading">
    <div>
      <div class="structure-eyebrow">No interpolation · no stale-data learning</div>
      <h2>Data continuity &amp; learning safety</h2>
      <p class="sub">Missed GitHub schedules are recovered from exchange history. Stale or discontinuous market tails are rejected before model updates.</p>
    </div>
    <span class="resilience-state {"ok" if fresh else "warn"}">{_escape(data_status)}</span>
  </div>
  <div class="resilience-grid">
    {_tile("Latest candle lag", _hours(lag), data_note)}
    {_tile("Continuous market rows", _integer(contiguous_rows), "Newest uninterrupted provider segment used by indicators and learning")}
    {_tile("Fetch recovery", _days(selected_window), recovery_note)}
    {_tile("Dashboard gaps", str(missing_records), f"Largest unpublished workflow gap: {largest_gap}h; chart lines are intentionally broken")}
    {_tile("Online learner", _integer(online_samples), f"Last learned closed candle: {last_learned}; missed runs are caught up chronologically")}
    {_tile("Heavy retraining", _label(policy_status), _label(policy_reason))}
  </div>
</section>'''


def _history_gap_summary(history: list[dict[str, Any]]) -> tuple[int, int]:
    times = []
    for item in history[-168:]:
        timestamp = _timestamp(
            item.get("candle_time") or item.get("run_finished_at")
        )
        if timestamp is not None:
            times.append(timestamp)
    times = sorted(set(times))
    missing = 0
    largest = 0
    for previous, current in zip(times, times[1:]):
        hours = int(round((current - previous).total_seconds() / 3600.0))
        gap = max(hours - 1, 0)
        missing += gap
        largest = max(largest, gap)
    return missing, largest


def _tile(title: str, value: str, note: str) -> str:
    return f'''
<div class="resilience-tile">
  <span>{_escape(title)}</span>
  <strong>{_escape(value)}</strong>
  <small>{_escape(note)}</small>
</div>'''


def _styles() -> str:
    return """
.gap-line{stroke:var(--wait);stroke-width:1.5;stroke-dasharray:5 5;opacity:.75}
.gap-label{fill:var(--wait);font-size:9px;font-weight:750}
.gap-key{width:18px!important;height:2px!important;border-radius:0!important;background:repeating-linear-gradient(90deg,var(--wait) 0 5px,transparent 5px 9px)}
.resilience-panel{margin-top:18px;background:linear-gradient(135deg,rgba(255,255,255,.88),rgba(222,236,231,.52),rgba(236,234,245,.42))}
.resilience-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;margin-bottom:18px}
.resilience-state{display:inline-flex;padding:8px 12px;border-radius:999px;background:var(--mint);color:var(--sage2);font-size:11px;font-weight:850}
.resilience-state.warn{background:var(--peach2);color:#9c625f}
.resilience-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.resilience-tile{min-width:0;padding:16px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.58)}
.resilience-tile span{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.06em}
.resilience-tile strong{display:block;margin-top:8px;font-size:16px;line-height:1.3;overflow-wrap:anywhere}
.resilience-tile small{display:block;margin-top:6px;color:var(--muted);font-size:10px;line-height:1.45}
:root[data-theme="dark"] .resilience-panel{background:linear-gradient(135deg,rgba(22,37,34,.91),rgba(47,41,63,.54))}
:root[data-theme="dark"] .resilience-tile{background:var(--surface-soft);border-color:var(--line)}
@media(max-width:980px){.resilience-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:620px){.resilience-heading{flex-direction:column}.resilience-grid{grid-template-columns:1fr}}
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
    return number if pd.notna(number) else None


def _timestamp(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        timestamp = pd.Timestamp(value)
        return (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
    except Exception:
        return None


def _compact_time(value: Any) -> str:
    timestamp = _timestamp(value)
    return "—" if timestamp is None else timestamp.strftime("%b %d · %H:%M UTC")


def _hours(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}h"


def _days(value: float | None) -> str:
    return "—" if value is None else f"{value:g} days"


def _integer(value: float | None) -> str:
    return "—" if value is None else f"{int(value):,}"


def _label(value: Any) -> str:
    return str(value or "—").replace("_", " ").title()


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
