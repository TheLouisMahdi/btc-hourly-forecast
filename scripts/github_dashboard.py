from __future__ import annotations

import argparse
import html
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from github_common import json_safe, write_json

MAX_HISTORY = 24 * 30
FINAL_DIRECTION_RESULTS = {
    "DIRECTION_CORRECT",
    "DIRECTION_WRONG",
}
SETTLEMENT_DELAY_SECONDS = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve next-candle forecasts and render the dashboard"
    )
    parser.add_argument("--state-dir", default=".github_state")
    parser.add_argument("--runtime-dir", default=".github_runtime/hourly")
    parser.add_argument("--site-dir", default="site")
    return parser.parse_args()


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_candles(
    database_path: Path,
    provider: str | None,
) -> pd.DataFrame:
    if not database_path.exists():
        return pd.DataFrame()
    clauses = ["closed = 1"]
    params: list[Any] = []
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    sql = (
        "SELECT open_time, open, high, low, close FROM candles WHERE "
        + " AND ".join(clauses)
        + " ORDER BY open_time"
    )
    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    if not frame.empty:
        frame["open_time"] = pd.to_datetime(
            frame["open_time"],
            utc=True,
        )
    return frame


def pending(
    item: dict[str, Any],
    result: str = "PENDING",
) -> dict[str, Any]:
    output = dict(item)
    output["prediction_result"] = result
    output.setdefault("direction_result", result)
    output.setdefault("interval_result", "PENDING")
    output.setdefault("actual_close", None)
    output.setdefault("actual_close_return", None)
    output.setdefault("actual_direction", None)
    output.setdefault("resolved_at", None)
    return output


def resolve_outcomes(
    history: list[dict[str, Any]],
    candles: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    current_time = _utc(now or pd.Timestamp.now(tz="UTC"))
    candle_map = _candle_map(candles)
    resolved: list[dict[str, Any]] = []

    for source in history:
        item = dict(source)
        if (
            item.get("direction_result") in FINAL_DIRECTION_RESULTS
            and item.get("resolved_at")
        ):
            item["prediction_result"] = item["direction_result"]
            resolved.append(item)
            continue
        if item.get("run_status") != "OK":
            resolved.append(pending(item, "NOT_SCORED"))
            continue

        contract = item.get("next_candle_forecast")
        if not isinstance(contract, dict):
            resolved.append(pending(item, "LEGACY_NOT_SCORED"))
            continue

        try:
            target_open_time = _utc(contract["target_open_time"])
            target_close_time = _utc(contract["target_close_time"])
            reference_close = float(contract["reference_close"])
            predicted_low = float(contract["likely_close_low"])
            predicted_high = float(contract["likely_close_high"])
        except (KeyError, TypeError, ValueError):
            resolved.append(pending(item, "NOT_SCORED"))
            continue

        item["target_candle_open_time"] = target_open_time.isoformat()
        item["target_candle_close_time"] = target_close_time.isoformat()
        evaluation_time = target_close_time + pd.Timedelta(
            seconds=SETTLEMENT_DELAY_SECONDS
        )
        item["evaluation_available_at"] = evaluation_time.isoformat()
        if current_time < evaluation_time:
            item["seconds_until_evaluation"] = max(
                0,
                int((evaluation_time - current_time).total_seconds()),
            )
            resolved.append(pending(item))
            continue

        target_candle = candle_map.get(target_open_time)
        if target_candle is None:
            resolved.append(pending(item))
            continue
        if reference_close <= 0 or predicted_low <= 0 or predicted_high <= 0:
            resolved.append(pending(item, "NOT_SCORED"))
            continue

        actual_close = float(target_candle["close"])
        actual_close_return = actual_close / reference_close - 1.0
        actual_direction = (
            "UP"
            if actual_close_return > 0
            else "DOWN"
            if actual_close_return < 0
            else "FLAT"
        )
        forecast_direction = str(
            contract.get("direction") or ""
        ).upper()
        direction_result = (
            "DIRECTION_CORRECT"
            if forecast_direction in {"UP", "DOWN"}
            and actual_direction == forecast_direction
            else "DIRECTION_WRONG"
        )
        in_range = min(predicted_low, predicted_high) <= actual_close <= max(
            predicted_low,
            predicted_high,
        )
        interval_result = "IN_RANGE" if in_range else "OUT_OF_RANGE"
        item.update(
            {
                "prediction_result": direction_result,
                "direction_result": direction_result,
                "interval_result": interval_result,
                "actual_close": actual_close,
                "actual_price": actual_close,
                "actual_close_return": actual_close_return,
                "actual_return": actual_close_return,
                "actual_direction": actual_direction,
                "actual_candle_open": float(target_candle["open"]),
                "actual_candle_high": float(target_candle["high"]),
                "actual_candle_low": float(target_candle["low"]),
                "resolved_at": current_time.isoformat(),
                "seconds_until_evaluation": 0,
            }
        )
        resolved.append(item)
    return resolved


def direction_accuracy(
    history: list[dict[str, Any]],
) -> tuple[int, int, float | None]:
    scored = [
        item
        for item in history
        if item.get("direction_result") in FINAL_DIRECTION_RESULTS
    ]
    correct = sum(
        item.get("direction_result") == "DIRECTION_CORRECT"
        for item in scored
    )
    return (
        correct,
        len(scored),
        correct / len(scored) if scored else None,
    )


def interval_coverage(
    history: list[dict[str, Any]],
) -> tuple[int, int, float | None]:
    scored = [
        item
        for item in history
        if item.get("interval_result") in {"IN_RANGE", "OUT_OF_RANGE"}
    ]
    inside = sum(
        item.get("interval_result") == "IN_RANGE"
        for item in scored
    )
    return (
        inside,
        len(scored),
        inside / len(scored) if scored else None,
    )


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    return f"{numeric:,.{digits}f}"


def percent(value: Any, digits: int = 1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(numeric):
        return "—"
    return f"{numeric * 100:.{digits}f}%"


def value(item: Any) -> str:
    return "—" if item in (None, "") else str(item)


def price(value_: Any) -> str:
    formatted = number(value_, 2)
    return "—" if formatted == "—" else f"${formatted}"


def compact_time(value_: Any) -> str:
    try:
        timestamp = _utc(value_)
    except Exception:
        return value(value_)
    return timestamp.strftime("%b %d · %H:%M UTC")


def chart(history: list[dict[str, Any]]) -> str:
    points = [
        item
        for item in history[-168:]
        if _finite(item.get("price")) is not None
    ]
    if len(points) < 2:
        return (
            '<div class="empty-chart">'
            "The chart will appear after more forecasts are recorded."
            "</div>"
        )

    width = 1120
    height = 360
    pad_x = 42
    pad_y = 30
    prices = [float(item["price"]) for item in points]
    latest_contract = points[-1].get("next_candle_forecast")
    scale_values = list(prices)
    if isinstance(latest_contract, dict):
        for key in ("likely_close_low", "likely_close_high", "median_close"):
            candidate = _finite(latest_contract.get(key))
            if candidate is not None:
                scale_values.append(candidate)
    low = min(scale_values)
    high = max(scale_values)
    margin = max((high - low) * 0.15, 1.0)
    low -= margin
    high += margin

    def xy(index: float, price_value: float) -> tuple[float, float]:
        x = (
            pad_x
            + index
            * (width - 2 * pad_x)
            / max(1, len(points))
        )
        y = (
            height
            - pad_y
            - (price_value - low)
            * (height - 2 * pad_y)
            / max(high - low, 1e-9)
        )
        return x, y

    path = " ".join(
        ("M" if index == 0 else "L")
        + f" {xy(index, item_price)[0]:.2f} {xy(index, item_price)[1]:.2f}"
        for index, item_price in enumerate(prices)
    )
    area = (
        path
        + f" L {xy(len(prices) - 1, low)[0]:.2f} {height-pad_y:.2f}"
        + f" L {xy(0, low)[0]:.2f} {height-pad_y:.2f} Z"
    )

    grid: list[str] = []
    labels: list[str] = []
    for line in range(4):
        ratio = line / 3
        y = pad_y + ratio * (height - 2 * pad_y)
        label_price = high - ratio * (high - low)
        grid.append(
            f'<line x1="{pad_x}" y1="{y:.2f}" '
            f'x2="{width-pad_x}" y2="{y:.2f}" class="chart-grid" />'
        )
        labels.append(
            f'<text x="6" y="{y+4:.2f}" class="chart-label">'
            f'${label_price:,.0f}</text>'
        )

    markers: list[str] = []
    for index, item in enumerate(points):
        result = str(item.get("direction_result") or "PENDING")
        direction = str(item.get("forecast_direction") or "")
        if direction not in {"UP", "DOWN"}:
            continue
        x, y = xy(index, float(item["price"]))
        marker_class = {
            "DIRECTION_CORRECT": "marker-correct",
            "DIRECTION_WRONG": "marker-wrong",
            "PENDING": "marker-pending",
        }.get(result, "marker-muted")
        symbol = "↑" if direction == "UP" else "↓"
        markers.append(
            f'<g transform="translate({x:.2f},{y-12:.2f})">'
            f'<circle r="11" class="marker-dot {marker_class}" />'
            f'<text y="4" text-anchor="middle" class="marker-symbol">'
            f'{symbol}</text></g>'
        )

    forecast_band = ""
    if isinstance(latest_contract, dict):
        band_low = _finite(latest_contract.get("likely_close_low"))
        band_high = _finite(latest_contract.get("likely_close_high"))
        median = _finite(latest_contract.get("median_close"))
        if band_low is not None and band_high is not None and median is not None:
            x0 = xy(len(points) - 1, prices[-1])[0]
            x1 = width - pad_x
            top = xy(len(points), band_high)[1]
            bottom = xy(len(points), band_low)[1]
            median_y = xy(len(points), median)[1]
            forecast_band = (
                f'<rect x="{x0:.2f}" y="{top:.2f}" '
                f'width="{max(0.0, x1-x0):.2f}" '
                f'height="{max(2.0, bottom-top):.2f}" '
                'class="forecast-band" />'
                f'<line x1="{x0:.2f}" y1="{median_y:.2f}" '
                f'x2="{x1:.2f}" y2="{median_y:.2f}" '
                'class="forecast-median" />'
            )

    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="BTC close history and next forecast range">'
        + "".join(grid)
        + "".join(labels)
        + f'<path d="{area}" class="chart-area" />'
        + f'<path d="{path}" class="chart-line" />'
        + forecast_band
        + "".join(markers)
        + "</svg>"
        + '<div class="chart-legend">'
        + '<span><i class="legend-dot correct-dot"></i>Correct direction</span>'
        + '<span><i class="legend-dot wrong-dot"></i>Wrong direction</span>'
        + '<span><i class="legend-dot pending-dot"></i>Pending</span>'
        + '<span><i class="legend-band"></i>Next close range</span>'
        + "</div>"
    )


def history_rows(history: list[dict[str, Any]]) -> str:
    output: list[str] = []
    classes = {
        "DIRECTION_CORRECT": "result-correct",
        "DIRECTION_WRONG": "result-wrong",
        "PENDING": "result-pending",
        "LEGACY_NOT_SCORED": "result-muted",
        "NOT_SCORED": "result-muted",
    }
    for item in reversed(history[-36:]):
        contract = item.get("next_candle_forecast")
        contract = contract if isinstance(contract, dict) else {}
        direction = value(contract.get("direction") or item.get("forecast_direction"))
        probability_up = contract.get("probability_up")
        direction_probability = (
            probability_up
            if direction == "UP"
            else 1.0 - float(probability_up)
            if _finite(probability_up) is not None
            else None
        )
        range_text = (
            f"{price(contract.get('likely_close_low'))} – "
            f"{price(contract.get('likely_close_high'))}"
            if contract
            else "—"
        )
        direction_result = str(
            item.get("direction_result") or "PENDING"
        )
        interval_result = value(item.get("interval_result"))
        output.append(
            "<tr>"
            f"<td>{esc(compact_time(item.get('candle_time') or item.get('run_finished_at')))}</td>"
            f"<td>{esc(price(item.get('price')))}</td>"
            f'<td><span class="direction-mini {"up" if direction == "UP" else "down"}">{esc(direction)}</span></td>'
            f"<td>{esc(percent(direction_probability))}</td>"
            f"<td>{esc(price(contract.get('median_close')))}</td>"
            f"<td>{esc(range_text)}</td>"
            f"<td>{esc(price(item.get('actual_close')))}</td>"
            f'<td><span class="table-result {classes.get(direction_result, "result-muted")}">{esc(direction_result.replace("DIRECTION_", ""))}</span></td>'
            f"<td>{esc(interval_result.replace('_', ' '))}</td>"
            "</tr>"
        )
    if output:
        return "".join(output)
    return (
        '<tr><td colspan="9" class="empty">'
        "No forecasts have been recorded yet."
        "</td></tr>"
    )


def blocker_chips(blockers: Any) -> str:
    if not isinstance(blockers, list) or not blockers:
        return '<span class="chip chip-good">No active blockers</span>'
    return "".join(
        f'<span class="chip">{esc(str(blocker).replace("_", " ").title())}</span>'
        for blocker in blockers
    )


def render(
    latest: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    contract = latest.get("next_candle_forecast")
    contract = contract if isinstance(contract, dict) else {}
    model = latest.get("price_forecast_model")
    model = model if isinstance(model, dict) else {}
    model_metrics = model.get("metrics")
    model_metrics = model_metrics if isinstance(model_metrics, dict) else {}

    direction = value(contract.get("direction") or latest.get("forecast_direction"))
    direction_class = "up" if direction == "UP" else "down"
    probability_up = _finite(contract.get("probability_up"))
    direction_probability = (
        probability_up
        if direction == "UP"
        else 1.0 - probability_up
        if probability_up is not None
        else None
    )
    confidence = percent(direction_probability)
    signal_strength = value(contract.get("signal_strength"))
    source = value(contract.get("forecast_source"))
    source_label = source.replace("_", " ").title()
    action = value(
        latest.get("action")
        or latest.get("status")
        or latest.get("run_status")
    )
    run_ok = latest.get("run_status") == "OK"

    direction_correct, direction_total, direction_rate = direction_accuracy(
        history
    )
    interval_inside, interval_total, coverage_rate = interval_coverage(history)
    direction_rate_text = (
        "—" if direction_rate is None else f"{direction_rate * 100:.1f}%"
    )
    coverage_text = (
        "—" if coverage_rate is None else f"{coverage_rate * 100:.1f}%"
    )

    direction_weight = _finite(contract.get("direction_blend_weight")) or 0.0
    return_weight = _finite(contract.get("return_blend_weight")) or 0.0
    batch_share = max(0.0, 1.0 - max(direction_weight, return_weight))
    online_share = max(direction_weight, return_weight)
    evaluation_text = (
        compact_time(latest.get("evaluation_available_at"))
        if latest.get("prediction_result") == "PENDING"
        else compact_time(latest.get("resolved_at"))
    )

    styles = """
:root {
  color-scheme: light;
  --bg: #f4f7f5;
  --surface: rgba(255, 255, 255, 0.76);
  --surface-solid: #ffffff;
  --ink: #263532;
  --muted: #72817d;
  --line: rgba(74, 101, 95, 0.14);
  --sage: #6f9b91;
  --sage-deep: #47746b;
  --mint: #dcebe6;
  --lavender: #8f8ab8;
  --lavender-soft: #ebe9f5;
  --peach: #c99078;
  --peach-soft: #f5e6de;
  --blue-soft: #e2edf2;
  --correct: #4d8b76;
  --wrong: #bd726e;
  --pending: #9b865c;
  --shadow: 0 24px 70px rgba(57, 79, 73, 0.10);
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  min-height: 100vh;
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  background:
    radial-gradient(circle at 8% 8%, rgba(220,235,230,.9), transparent 32%),
    radial-gradient(circle at 92% 5%, rgba(235,233,245,.86), transparent 30%),
    linear-gradient(180deg, #f8faf9 0%, var(--bg) 100%);
}
a { color: inherit; }
.shell { width: min(1240px, 92%); margin: 0 auto; padding: 28px 0 44px; }
.topbar { display: flex; justify-content: space-between; align-items: center; gap: 18px; margin-bottom: 26px; }
.brand { display: flex; align-items: center; gap: 13px; }
.brand-mark { width: 44px; height: 44px; border-radius: 15px; display: grid; place-items: center; background: linear-gradient(135deg, var(--mint), var(--lavender-soft)); box-shadow: inset 0 0 0 1px rgba(71,116,107,.1); font-weight: 850; color: var(--sage-deep); }
.brand h1 { margin: 0; font-size: 18px; letter-spacing: -.02em; }
.brand p { margin: 3px 0 0; color: var(--muted); font-size: 12px; }
.run-pill { display: inline-flex; align-items: center; gap: 8px; padding: 9px 13px; border-radius: 999px; background: rgba(255,255,255,.72); border: 1px solid var(--line); font-size: 12px; font-weight: 750; }
.run-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--correct); box-shadow: 0 0 0 5px rgba(77,139,118,.12); }
.run-pill.failed .run-dot { background: var(--wrong); box-shadow: 0 0 0 5px rgba(189,114,110,.12); }
.hero { position: relative; overflow: hidden; display: grid; grid-template-columns: 1.18fr .82fr; gap: 22px; padding: clamp(24px, 4vw, 48px); border: 1px solid rgba(255,255,255,.85); border-radius: 34px; background: linear-gradient(135deg, rgba(255,255,255,.94), rgba(247,250,249,.72)); box-shadow: var(--shadow); }
.hero::after { content: ""; position: absolute; width: 360px; height: 360px; right: -140px; top: -180px; border-radius: 50%; background: linear-gradient(135deg, rgba(220,235,230,.7), rgba(235,233,245,.68)); filter: blur(2px); }
.hero-main, .hero-side { position: relative; z-index: 1; }
.eyebrow { display: inline-flex; align-items: center; gap: 8px; color: var(--sage-deep); font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.eyebrow::before { content: ""; width: 22px; height: 2px; border-radius: 2px; background: var(--sage); }
.direction-row { display: flex; align-items: end; flex-wrap: wrap; gap: 14px; margin: 18px 0 8px; }
.direction { font-size: clamp(58px, 10vw, 106px); line-height: .88; letter-spacing: -.075em; font-weight: 850; }
.direction.up { color: var(--sage-deep); }
.direction.down { color: #a96663; }
.confidence-badge { margin-bottom: 8px; padding: 9px 12px; border-radius: 14px; background: var(--mint); color: var(--sage-deep); font-size: 13px; font-weight: 800; }
.hero-copy { max-width: 620px; color: var(--muted); line-height: 1.7; font-size: 14px; }
.hero-copy strong { color: var(--ink); }
.probability { margin-top: 26px; }
.probability-head { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; margin-bottom: 9px; }
.probability-track { height: 12px; border-radius: 999px; background: var(--peach-soft); overflow: hidden; box-shadow: inset 0 0 0 1px rgba(127,93,83,.08); }
.probability-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--sage), #82aaa1); }
.hero-side { display: flex; flex-direction: column; justify-content: space-between; gap: 18px; padding: 22px; border-radius: 25px; background: rgba(246,249,248,.75); border: 1px solid var(--line); }
.range-label { color: var(--muted); font-size: 12px; }
.range-value { margin-top: 7px; font-size: clamp(24px, 4vw, 38px); font-weight: 820; letter-spacing: -.04em; }
.range-caption { margin-top: 8px; color: var(--muted); font-size: 12px; line-height: 1.55; }
.range-scale { position: relative; height: 10px; margin: 22px 0 10px; border-radius: 999px; background: linear-gradient(90deg, var(--peach-soft), var(--lavender-soft), var(--mint)); }
.range-scale::after { content: ""; position: absolute; left: 50%; top: 50%; width: 16px; height: 16px; border-radius: 50%; background: var(--surface-solid); border: 4px solid var(--lavender); transform: translate(-50%, -50%); box-shadow: 0 4px 12px rgba(80,76,112,.18); }
.range-points { display: flex; justify-content: space-between; gap: 8px; font-size: 11px; color: var(--muted); }
.meta-line { display: flex; justify-content: space-between; gap: 12px; padding-top: 14px; border-top: 1px solid var(--line); font-size: 12px; }
.meta-line span:first-child { color: var(--muted); }
.metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 18px; }
.metric { padding: 20px; border-radius: 23px; background: var(--surface); border: 1px solid rgba(255,255,255,.82); box-shadow: 0 12px 36px rgba(57,79,73,.055); backdrop-filter: blur(14px); }
.metric-label { color: var(--muted); font-size: 12px; }
.metric-value { margin-top: 9px; font-size: 25px; font-weight: 820; letter-spacing: -.035em; }
.metric-note { margin-top: 6px; color: var(--muted); font-size: 11px; line-height: 1.45; }
.content-grid { display: grid; grid-template-columns: 1.45fr .55fr; gap: 18px; margin-top: 18px; }
.panel { min-width: 0; padding: 24px; border-radius: 28px; background: var(--surface); border: 1px solid rgba(255,255,255,.82); box-shadow: 0 16px 44px rgba(57,79,73,.06); backdrop-filter: blur(14px); }
.panel-head { display: flex; justify-content: space-between; align-items: start; gap: 14px; margin-bottom: 18px; }
.panel h2 { margin: 0; font-size: 17px; letter-spacing: -.02em; }
.panel-sub { margin-top: 5px; color: var(--muted); font-size: 12px; }
.chart { width: 100%; min-height: 280px; display: block; }
.chart-grid { stroke: rgba(74,101,95,.10); stroke-width: 1; }
.chart-label { fill: #87938f; font-size: 10px; }
.chart-line { fill: none; stroke: var(--sage-deep); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
.chart-area { fill: url(#none); opacity: .08; }
.forecast-band { fill: rgba(143,138,184,.18); }
.forecast-median { stroke: var(--lavender); stroke-width: 2.5; stroke-dasharray: 6 5; }
.marker-dot { stroke-width: 2; }
.marker-correct { fill: #e3f0eb; stroke: var(--correct); }
.marker-wrong { fill: #f5e6e4; stroke: var(--wrong); }
.marker-pending { fill: #f3eddf; stroke: var(--pending); }
.marker-muted { fill: #edf1ef; stroke: #9aa6a2; }
.marker-symbol { fill: var(--ink); font-size: 12px; font-weight: 900; }
.chart-legend { display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: 11px; }
.chart-legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.correct-dot { background: var(--correct); }
.wrong-dot { background: var(--wrong); }
.pending-dot { background: var(--pending); }
.legend-band { width: 18px; height: 8px; border-radius: 3px; background: rgba(143,138,184,.28); display: inline-block; }
.learning-stack { display: grid; gap: 12px; }
.learning-item { padding: 16px; border-radius: 18px; background: rgba(255,255,255,.56); border: 1px solid var(--line); }
.learning-top { display: flex; justify-content: space-between; gap: 10px; font-size: 12px; }
.learning-top span:first-child { color: var(--muted); }
.learning-value { font-weight: 800; }
.mini-track { height: 7px; border-radius: 99px; background: #edf1ef; margin-top: 10px; overflow: hidden; }
.mini-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--sage), var(--lavender)); }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { padding: 8px 10px; border-radius: 999px; background: rgba(245,230,222,.72); color: #846257; border: 1px solid rgba(201,144,120,.16); font-size: 10px; font-weight: 750; }
.chip-good { background: var(--mint); color: var(--sage-deep); }
.history-panel { margin-top: 18px; }
.table-wrap { overflow: auto; max-height: 520px; border-radius: 18px; border: 1px solid var(--line); background: rgba(255,255,255,.45); }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 13px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { position: sticky; top: 0; z-index: 2; background: rgba(246,249,248,.96); color: var(--muted); font-weight: 700; }
.direction-mini { display: inline-flex; min-width: 46px; justify-content: center; padding: 5px 8px; border-radius: 999px; font-size: 10px; font-weight: 850; }
.direction-mini.up { background: var(--mint); color: var(--sage-deep); }
.direction-mini.down { background: var(--peach-soft); color: #9c625f; }
.table-result { display: inline-flex; padding: 5px 8px; border-radius: 999px; font-size: 10px; font-weight: 800; }
.result-correct { color: var(--correct); background: rgba(77,139,118,.10); }
.result-wrong { color: var(--wrong); background: rgba(189,114,110,.10); }
.result-pending { color: var(--pending); background: rgba(155,134,92,.10); }
.result-muted { color: var(--muted); background: rgba(114,129,125,.08); }
.empty, .empty-chart { text-align: center; color: var(--muted); padding: 60px 18px; }
.footer { margin-top: 24px; padding: 20px 4px 0; display: flex; justify-content: space-between; align-items: center; gap: 16px; color: var(--muted); font-size: 11px; border-top: 1px solid var(--line); }
.footer a { color: var(--sage-deep); text-decoration: none; font-weight: 800; }
.footer a:hover { text-decoration: underline; }
@media (max-width: 980px) {
  .hero, .content-grid { grid-template-columns: 1fr; }
  .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 620px) {
  .shell { width: 93%; padding-top: 18px; }
  .topbar { align-items: flex-start; }
  .brand p { display: none; }
  .hero { border-radius: 26px; padding: 22px; }
  .direction { font-size: 62px; }
  .metrics { grid-template-columns: 1fr; }
  .panel { border-radius: 22px; padding: 18px; }
  .footer { flex-direction: column; align-items: flex-start; }
}
"""

    return f'''<!doctype html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <meta http-equiv="refresh" content="300">
  <meta name="theme-color" content="#f4f7f5">
  <meta name="description" content="Adaptive next-candle BTC direction and price-range forecast">
  <title>BTC Next-Candle Forecast</title>
  <style>{styles}</style>
</head>
<body>
<div class="shell">
  <nav class="topbar">
    <div class="brand">
      <div class="brand-mark">₿</div>
      <div>
        <h1>BTC Next-Candle Forecast</h1>
        <p>Direction-first forecasting with adaptive price learning</p>
      </div>
    </div>
    <div class="run-pill {'failed' if not run_ok else ''}">
      <span class="run-dot"></span>
      {'System healthy' if run_ok else 'Fail-safe active'}
    </div>
  </nav>

  <main>
    <section class="hero">
      <div class="hero-main">
        <div class="eyebrow">Next closed 1-hour candle</div>
        <div class="direction-row">
          <div class="direction {direction_class}">{esc(direction)}</div>
          <div class="confidence-badge">{esc(confidence)} confidence · {esc(signal_strength)}</div>
        </div>
        <p class="hero-copy">
          The model expects the next candle to close <strong>{'above' if direction == 'UP' else 'below'} {esc(price(contract.get('reference_close')))}</strong>.
          Direction is scored independently from the price interval. The forecast is frozen until the target candle closes.
        </p>
        <div class="probability">
          <div class="probability-head"><span>Down {esc(percent(contract.get('probability_down')))}</span><span>Up {esc(percent(contract.get('probability_up')))}</span></div>
          <div class="probability-track"><div class="probability-fill" style="width:{max(0.0, min(100.0, (probability_up or 0.5) * 100)):.2f}%"></div></div>
        </div>
      </div>

      <aside class="hero-side">
        <div>
          <div class="range-label">Model-estimated next close</div>
          <div class="range-value">{esc(price(contract.get('median_close')))}</div>
          <div class="range-caption">Probable {esc(percent(contract.get('interval_probability'), 0))} interval generated from model residuals, not a generic candle range.</div>
          <div class="range-scale"></div>
          <div class="range-points"><span>{esc(price(contract.get('likely_close_low')))}</span><span>{esc(price(contract.get('median_close')))}</span><span>{esc(price(contract.get('likely_close_high')))}</span></div>
        </div>
        <div>
          <div class="meta-line"><span>Forecast source</span><strong>{esc(source_label)}</strong></div>
          <div class="meta-line"><span>Target closes</span><strong>{esc(compact_time(contract.get('target_close_time')))}</strong></div>
          <div class="meta-line"><span>Evaluation</span><strong>{esc(evaluation_text)}</strong></div>
        </div>
      </aside>
    </section>

    <section class="metrics">
      <article class="metric"><div class="metric-label">Direction accuracy</div><div class="metric-value">{esc(direction_rate_text)}</div><div class="metric-note">{direction_correct} correct across {direction_total} resolved forecasts</div></article>
      <article class="metric"><div class="metric-label">Interval coverage</div><div class="metric-value">{esc(coverage_text)}</div><div class="metric-note">{interval_inside} closes inside {interval_total} resolved intervals</div></article>
      <article class="metric"><div class="metric-label">Decision</div><div class="metric-value">{esc(action)}</div><div class="metric-note">Trade gates remain separate from direction forecasting</div></article>
      <article class="metric"><div class="metric-label">Market regime</div><div class="metric-value">{esc(value(latest.get('regime')).replace('_', ' ').title())}</div><div class="metric-note">Scenario: {esc(value(contract.get('scenario')).replace('_', ' ').title())}</div></article>
    </section>

    <section class="content-grid">
      <article class="panel">
        <div class="panel-head"><div><h2>Price path and next forecast</h2><div class="panel-sub">Source closes, direction outcomes and the current model interval</div></div></div>
        {chart(history)}
      </article>

      <aside class="panel">
        <div class="panel-head"><div><h2>Adaptive learning</h2><div class="panel-sub">Performance-weighted Batch and Online fusion</div></div></div>
        <div class="learning-stack">
          <div class="learning-item"><div class="learning-top"><span>Batch influence</span><span class="learning-value">{esc(percent(batch_share))}</span></div><div class="mini-track"><div class="mini-fill" style="width:{batch_share*100:.2f}%"></div></div></div>
          <div class="learning-item"><div class="learning-top"><span>Online influence</span><span class="learning-value">{esc(percent(online_share))}</span></div><div class="mini-track"><div class="mini-fill" style="width:{online_share*100:.2f}%"></div></div></div>
          <div class="learning-item"><div class="learning-top"><span>Online evaluation samples</span><span class="learning-value">{esc(value(model_metrics.get('samples')))}</span></div><div class="range-caption">Online predictions receive weight only when Brier score, accuracy and return error remain competitive with the Batch champion.</div></div>
          <div class="learning-item"><div class="learning-top"><span>Batch direction accuracy</span><span class="learning-value">{esc(percent(model_metrics.get('base_direction_accuracy')))}</span></div><div class="learning-top" style="margin-top:8px"><span>Online direction accuracy</span><span class="learning-value">{esc(percent(model_metrics.get('online_direction_accuracy')))}</span></div></div>
        </div>
        <div class="panel-head" style="margin-top:22px"><div><h2>Decision blockers</h2><div class="panel-sub">These affect trade execution, not the forecast score</div></div></div>
        <div class="chips">{blocker_chips(latest.get('blockers'))}</div>
      </aside>
    </section>

    <section class="panel history-panel">
      <div class="panel-head"><div><h2>Forecast ledger</h2><div class="panel-sub">Immutable forecasts resolved only after the target candle closes</div></div></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Source candle</th><th>Source close</th><th>Direction</th><th>Direction confidence</th><th>Expected close</th><th>Probable range</th><th>Actual close</th><th>Direction result</th><th>Interval result</th></tr></thead>
          <tbody>{history_rows(history)}</tbody>
        </table>
      </div>
    </section>
  </main>

  <footer class="footer">
    <span>© 2026 Mahdi Ghahremani · ID: TheLouisMahdi · Research and paper-trading only.</span>
    <a href="https://github.com/TheLouisMahdi" target="_blank" rel="noopener noreferrer">GitHub · @TheLouisMahdi</a>
  </footer>
</div>
</body>
</html>'''


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    state_dir = (root / args.state_dir).resolve()
    runtime_dir = (root / args.runtime_dir).resolve()
    site_dir = (root / args.site_dir).resolve()
    history = load_json(state_dir / "history.json", [])
    latest = load_json(state_dir / "latest.json", {})
    provider = str(latest.get("provider") or "") or None
    history = resolve_outcomes(
        history,
        load_candles(
            runtime_dir / "btc_hourly.sqlite3",
            provider,
        ),
    )[-MAX_HISTORY:]
    latest_key = latest.get("candle_time") or latest.get(
        "run_finished_at"
    )
    latest = next(
        (
            item
            for item in reversed(history)
            if (
                item.get("candle_time")
                or item.get("run_finished_at")
            )
            == latest_key
        ),
        latest,
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)
    write_json(state_dir / "latest.json", latest)
    write_json(state_dir / "history.json", history)
    write_json(site_dir / "latest.json", latest)
    write_json(site_dir / "history.json", history)
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    (site_dir / "index.html").write_text(
        render(latest, history),
        encoding="utf-8",
    )
    correct, scored, rate = direction_accuracy(history)
    inside, intervals, coverage = interval_coverage(history)
    print(
        json.dumps(
            {
                "dashboard": "updated",
                "direction_resolved": scored,
                "direction_correct": correct,
                "direction_accuracy": rate,
                "interval_resolved": intervals,
                "interval_inside": inside,
                "interval_coverage": coverage,
            },
            indent=2,
        )
    )
    return 0


def _candle_map(candles: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    if candles.empty:
        return {}
    return {
        _utc(row.open_time): {
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
        }
        for row in candles.drop_duplicates(
            "open_time",
            keep="last",
        ).itertuples(index=False)
        if not pd.isna(row.close)
    }


def _utc(value_: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value_)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _finite(value_: Any) -> float | None:
    try:
        numeric = float(value_)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


if __name__ == "__main__":
    raise SystemExit(main())
