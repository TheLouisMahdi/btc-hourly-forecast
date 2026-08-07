from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

MARKER = 'data-elegant-chart="v2"'
MAX_POINTS = 96
MAX_OUTCOME_MARKERS = 6


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

    chart = _chart(history, latest)
    pattern = re.compile(
        r'<svg class="chart".*?</svg>\s*<div class="legend">.*?</div>',
        flags=re.DOTALL,
    )
    document, replacements = pattern.subn(chart, document, count=1)
    if replacements != 1:
        raise RuntimeError("Price chart contract could not replace the legacy chart")

    document = document.replace(
        "Closed-candle prices, direction outcomes and the calibrated closing range",
        "Recent BTC closes with next-candle direction and calibrated range",
    )
    document = document.replace("</style>", _styles() + "\n</style>", 1)
    index_path.write_text(document, encoding="utf-8")
    return 0


def _chart(history: list[dict[str, Any]], latest: dict[str, Any]) -> str:
    points = _points(history)
    if len(points) < 2:
        return (
            f'<div class="price-chart-empty" {MARKER}>'
            "More closed-candle history is required to draw the price path."
            "</div>"
        )

    width, height = 980.0, 420.0
    pad_left, pad_right, pad_top, pad_bottom = 58.0, 78.0, 26.0, 52.0
    plot_bottom = height - pad_bottom

    contract = latest.get("next_candle_forecast")
    if not isinstance(contract, dict):
        contract = points[-1]["item"].get("next_candle_forecast")
    contract = contract if isinstance(contract, dict) else {}

    prices = [point["price"] for point in points]
    range_low = _number(contract.get("likely_close_low"))
    range_high = _number(contract.get("likely_close_high"))
    if range_low is not None:
        prices.append(range_low)
    if range_high is not None:
        prices.append(range_high)
    low, high = min(prices), max(prices)
    margin = max((high - low) * 0.10, max(abs(high), 1.0) * 0.0015)
    low -= margin
    high += margin

    start = points[0]["time"]
    last_time = points[-1]["time"]
    target_time = _timestamp(contract.get("target_close_time"))
    if target_time is None or target_time <= last_time:
        target_time = last_time + pd.Timedelta(hours=1)
    end = max(target_time, last_time + pd.Timedelta(hours=1))
    duration = max((end - start).total_seconds(), 3600.0)

    def xy(timestamp: pd.Timestamp, value: float) -> tuple[float, float]:
        x_fraction = (timestamp - start).total_seconds() / duration
        x = pad_left + x_fraction * (width - pad_left - pad_right)
        y = plot_bottom - (value - low) * (plot_bottom - pad_top) / max(high - low, 1e-9)
        return x, y

    segments, has_gaps = _segments(points)
    paths: list[str] = []
    areas: list[str] = []
    for segment in segments:
        if len(segment) < 2:
            continue
        coordinates = [xy(point["time"], point["price"]) for point in segment]
        commands = " ".join(
            f'{"M" if index == 0 else "L"} {x:.2f} {y:.2f}'
            for index, (x, y) in enumerate(coordinates)
        )
        paths.append(f'<path d="{commands}" class="price-line"/>')
        first_x = coordinates[0][0]
        last_x = coordinates[-1][0]
        area_commands = (
            commands
            + f" L {last_x:.2f} {plot_bottom:.2f}"
            + f" L {first_x:.2f} {plot_bottom:.2f} Z"
        )
        areas.append(f'<path d="{area_commands}" class="price-area"/>')

    horizontal_grid: list[str] = []
    y_labels: list[str] = []
    for index in range(4):
        ratio = index / 3
        y = pad_top + ratio * (plot_bottom - pad_top)
        value = high - ratio * (high - low)
        horizontal_grid.append(
            f'<line x1="{pad_left:.2f}" y1="{y:.2f}" '
            f'x2="{width-pad_right:.2f}" y2="{y:.2f}" class="chart-grid"/>'
        )
        y_labels.append(
            f'<text x="6" y="{y+4:.2f}" class="chart-y-label">{_compact_price(value)}</text>'
        )

    x_labels: list[str] = []
    for index in range(5):
        ratio = index / 4
        timestamp = start + (last_time - start) * ratio
        x, _ = xy(timestamp, low)
        anchor = "start" if index == 0 else "end" if index == 4 else "middle"
        mobile_hide = " x-mobile-hide" if index in {1, 3} else ""
        x_labels.append(
            f'<text x="{x:.2f}" y="{height-16:.2f}" text-anchor="{anchor}" '
            f'class="chart-x-label{mobile_hide}">{_time_label(timestamp, index == 4)}</text>'
        )

    outcome_candidates = [
        point
        for point in points
        if str(point["item"].get("direction_result") or "")
        in {"DIRECTION_CORRECT", "DIRECTION_WRONG"}
    ][-MAX_OUTCOME_MARKERS:]
    outcomes: list[str] = []
    for point in outcome_candidates:
        x, y = xy(point["time"], point["price"])
        result = str(point["item"].get("direction_result"))
        css_class = "correct" if result == "DIRECTION_CORRECT" else "wrong"
        direction = str(
            (point["item"].get("next_candle_forecast") or {}).get("direction")
            if isinstance(point["item"].get("next_candle_forecast"), dict)
            else ""
        ).upper()
        title = html.escape(
            f"{_time_label(point['time'], False)} · Forecast {direction or '—'} · "
            f"{'Correct' if css_class == 'correct' else 'Wrong'} · ${point['price']:,.2f}"
        )
        outcomes.append(
            f'<g class="outcome-marker {css_class}"><title>{title}</title>'
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.0"/></g>'
        )

    last_x, last_y = xy(points[-1]["time"], points[-1]["price"])
    direction = str(contract.get("direction") or "").upper()
    direction_symbol = "↑" if direction == "UP" else "↓" if direction == "DOWN" else ""
    latest_marker = (
        f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="10" class="latest-ring"/>'
        f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="4.5" class="latest-dot"/>'
    )
    if direction_symbol:
        latest_marker += (
            f'<text x="{last_x+13:.2f}" y="{last_y-10:.2f}" '
            f'class="direction-arrow">{direction_symbol}</text>'
        )

    band = ""
    if range_low is not None and range_high is not None and range_low > 0 and range_high > 0:
        band_low, band_high = sorted((range_low, range_high))
        top_y = xy(last_time, band_high)[1]
        bottom_y = xy(last_time, band_low)[1]
        target_x = xy(target_time, points[-1]["price"])[0]
        band_x = min(last_x + 8.0, target_x)
        band_width = max(target_x - band_x, 18.0)
        band_height = max(bottom_y - top_y, 8.0)
        label_x = band_x + band_width / 2
        label_y = max(top_y - 9.0, pad_top + 11.0)
        range_text = f"${band_low:,.0f} – ${band_high:,.0f}"
        band = (
            f'<rect x="{band_x:.2f}" y="{top_y:.2f}" width="{band_width:.2f}" '
            f'height="{band_height:.2f}" rx="9" class="forecast-band"/>'
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" '
            f'class="forecast-band-label">Likely range</text>'
            f'<text x="{label_x:.2f}" y="{min(bottom_y+17.0, plot_bottom-3.0):.2f}" '
            f'text-anchor="middle" class="forecast-price-label">{html.escape(range_text)}</text>'
        )

    gap_legend = '<span class="legend-gap">⋯ Data gap</span>' if has_gaps else ""
    return f'''
<div class="price-chart-shell" {MARKER}>
  <svg class="chart price-chart-v2" viewBox="0 0 {int(width)} {int(height)}" role="img" aria-label="BTC closed-candle price path with breaks for missing records and a calibrated next-candle range">
    <defs>
      <linearGradient id="priceAreaGradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" class="price-area-stop top"/>
        <stop offset="100%" class="price-area-stop bottom"/>
      </linearGradient>
    </defs>
    {''.join(horizontal_grid)}
    {''.join(y_labels)}
    {''.join(x_labels)}
    {''.join(areas)}
    {band}
    {''.join(paths)}
    {''.join(outcomes)}
    {latest_marker}
  </svg>
  <div class="legend chart-legend">
    <span><i class="legend-dot correct"></i>Correct</span>
    <span><i class="legend-dot wrong"></i>Wrong</span>
    <span><i class="legend-range"></i>Forecast range</span>
    {gap_legend}
  </div>
</div>'''


def _points(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[pd.Timestamp, dict[str, Any]] = {}
    for item in history[-MAX_POINTS * 2 :]:
        if not isinstance(item, dict):
            continue
        price = _number(item.get("price"))
        timestamp = _timestamp(item.get("candle_time") or item.get("run_finished_at"))
        if price is None or timestamp is None:
            continue
        keyed[timestamp] = {"time": timestamp, "price": price, "item": item}
    return [keyed[key] for key in sorted(keyed)][-MAX_POINTS:]


def _segments(points: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], bool]:
    segments: list[list[dict[str, Any]]] = [[points[0]]]
    has_gaps = False
    for previous, current in zip(points, points[1:]):
        hours = (current["time"] - previous["time"]).total_seconds() / 3600.0
        if hours > 1.5:
            has_gaps = True
            segments.append([current])
        else:
            segments[-1].append(current)
    return segments, has_gaps


def _styles() -> str:
    return r'''
:root{
  --chart-price:#5f8f86;
  --chart-fill-top:rgba(95,143,134,.13);
  --chart-fill-bottom:rgba(95,143,134,0);
  --chart-grid:rgba(41,56,52,.075);
  --chart-label:rgba(54,70,66,.58);
  --chart-correct:#5f9e87;
  --chart-wrong:#c57d78;
  --chart-range:rgba(130,120,170,.15);
  --chart-range-border:rgba(130,120,170,.32);
  --chart-range-text:#716a91;
}
:root[data-theme="dark"]{
  --chart-price:#8fc9bb;
  --chart-fill-top:rgba(143,201,187,.17);
  --chart-fill-bottom:rgba(143,201,187,0);
  --chart-grid:rgba(230,240,236,.085);
  --chart-label:rgba(226,239,234,.58);
  --chart-correct:#86cbb4;
  --chart-wrong:#e59a94;
  --chart-range:rgba(158,145,205,.18);
  --chart-range-border:rgba(177,164,220,.38);
  --chart-range-text:#c5bbe8;
}
.price-chart-shell{margin-top:10px}
.price-chart-v2{display:block;width:100%;height:auto;overflow:visible}
.chart-grid{stroke:var(--chart-grid);stroke-width:1}
.chart-y-label,.chart-x-label{fill:var(--chart-label);font-size:10px;font-weight:650;letter-spacing:.01em}
.price-line{fill:none;stroke:var(--chart-price);stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round;vector-effect:non-scaling-stroke}
.price-area{fill:url(#priceAreaGradient);stroke:none}
.price-area-stop.top{stop-color:var(--chart-price);stop-opacity:.15}
.price-area-stop.bottom{stop-color:var(--chart-price);stop-opacity:0}
.outcome-marker circle{stroke:var(--paper);stroke-width:2;vector-effect:non-scaling-stroke}.outcome-marker.correct circle{fill:var(--chart-correct)}.outcome-marker.wrong circle{fill:var(--chart-wrong)}
.latest-ring{fill:none;stroke:var(--chart-price);stroke-width:1.5;opacity:.28;vector-effect:non-scaling-stroke}.latest-dot{fill:var(--chart-price);stroke:var(--paper);stroke-width:2;vector-effect:non-scaling-stroke}.direction-arrow{fill:var(--chart-price);font-size:17px;font-weight:900}
.forecast-band{fill:var(--chart-range);stroke:var(--chart-range-border);stroke-width:1.2;vector-effect:non-scaling-stroke}.forecast-band-label{fill:var(--chart-range-text);font-size:9px;font-weight:800;letter-spacing:.04em}.forecast-price-label{fill:var(--chart-range-text);font-size:9px;font-weight:700}
.chart-legend{display:flex;align-items:center;flex-wrap:wrap;gap:10px 18px;margin-top:8px;color:var(--muted);font-size:12px}.chart-legend span{display:inline-flex;align-items:center;gap:7px}.legend-dot{width:8px!important;height:8px!important;border-radius:50%!important}.legend-dot.correct{background:var(--chart-correct)!important}.legend-dot.wrong{background:var(--chart-wrong)!important}.legend-range{width:18px!important;height:8px!important;border-radius:4px!important;background:var(--chart-range)!important;border:1px solid var(--chart-range-border)}.legend-gap{letter-spacing:.01em}
.price-chart-empty{padding:54px 16px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:18px}
/* Give the compact controls a little breathing room without enlarging the header. */
.health-badge{padding:10px 12px!important}
.theme-toggle{padding:9px 13px 9px 10px!important;min-height:46px!important}
@media(max-width:620px){
  .price-chart-shell{margin-top:6px}
  .chart-x-label.x-mobile-hide{display:none}
  .chart-y-label,.chart-x-label{font-size:9px}
  .forecast-price-label{display:none}
  .forecast-band-label{font-size:8px}
  .chart-legend{gap:8px 12px;font-size:11px;margin-top:4px}
  .health-badge{padding:10px 12px!important}
  .theme-toggle{padding:9px 12px!important}
}
@media(prefers-reduced-motion:reduce){.latest-ring{animation:none}}
'''


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
        return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    except Exception:
        return None


def _compact_price(value: float) -> str:
    if abs(value) >= 1000:
        return f"${value / 1000:.1f}k"
    return f"${value:,.0f}"


def _time_label(value: pd.Timestamp, now_label: bool) -> str:
    if now_label:
        return "Now"
    return value.strftime("%b %d · %H:%M")


if __name__ == "__main__":
    raise SystemExit(main())
