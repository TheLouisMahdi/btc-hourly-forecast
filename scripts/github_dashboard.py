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
FINAL_RESULTS = {"IN_RANGE", "OUT_OF_RANGE"}
SETTLEMENT_DELAY_SECONDS = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve next-candle interval forecasts and render the static dashboard"
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


def load_candles(database_path: Path, provider: str | None) -> pd.DataFrame:
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
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    return frame


def pending(item: dict[str, Any], result: str = "PENDING") -> dict[str, Any]:
    output = dict(item)
    output["prediction_result"] = result
    output.setdefault("direction_result", "PENDING")
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
            item.get("prediction_result") in FINAL_RESULTS
            and item.get("resolved_at")
        ):
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
        forecast_direction = str(contract.get("direction") or "RANGE").upper()
        if forecast_direction in {"UP", "DOWN"}:
            direction_result = (
                "DIRECTION_CORRECT"
                if actual_direction == forecast_direction
                else "DIRECTION_WRONG"
            )
        else:
            direction_result = "DIRECTION_NOT_SCORED"
        in_range = min(predicted_low, predicted_high) <= actual_close <= max(
            predicted_low,
            predicted_high,
        )
        item.update(
            {
                "prediction_result": "IN_RANGE" if in_range else "OUT_OF_RANGE",
                "direction_result": direction_result,
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


def accuracy(history: list[dict[str, Any]]) -> tuple[int, int, float | None]:
    scored = [
        item for item in history if item.get("prediction_result") in FINAL_RESULTS
    ]
    inside = sum(item.get("prediction_result") == "IN_RANGE" for item in scored)
    return inside, len(scored), inside / len(scored) if scored else None


def direction_accuracy(history: list[dict[str, Any]]) -> tuple[int, int, float | None]:
    scored = [
        item
        for item in history
        if item.get("direction_result")
        in {"DIRECTION_CORRECT", "DIRECTION_WRONG"}
    ]
    correct = sum(
        item.get("direction_result") == "DIRECTION_CORRECT" for item in scored
    )
    return correct, len(scored), correct / len(scored) if scored else None


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"


def percent(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def value(item: Any) -> str:
    return "—" if item in (None, "") else str(item)


def price_range(contract: dict[str, Any] | None) -> str:
    if not isinstance(contract, dict):
        return "—"
    low = number(contract.get("likely_close_low"), 2)
    high = number(contract.get("likely_close_high"), 2)
    return "—" if low == "—" or high == "—" else f"${low} – ${high}"


def chart(history: list[dict[str, Any]]) -> str:
    points = [
        item for item in history[-168:] if _finite(item.get("price")) is not None
    ]
    if len(points) < 2:
        return '<div class="empty-chart">Not enough history to draw the chart.</div>'

    width, height, pad_x, pad_y = 1120, 340, 48, 28
    values: list[float] = [float(item["price"]) for item in points]
    for item in points:
        contract = item.get("next_candle_forecast")
        if isinstance(contract, dict):
            for key in ("likely_close_low", "likely_close_high", "median_close"):
                candidate = _finite(contract.get(key))
                if candidate is not None:
                    values.append(candidate)
        actual = _finite(item.get("actual_close"))
        if actual is not None:
            values.append(actual)
    low, high = min(values), max(values)
    margin = max((high - low) * 0.10, 1.0)
    low -= margin
    high += margin

    def xy(index: int, price: float) -> tuple[float, float]:
        x = pad_x + index * (width - 2 * pad_x) / max(1, len(points) - 1)
        y = height - pad_y - (price - low) * (height - 2 * pad_y) / (high - low)
        return x, y

    source_path = " ".join(
        ("M" if index == 0 else "L")
        + f" {xy(index, float(item['price']))[0]:.2f} {xy(index, float(item['price']))[1]:.2f}"
        for index, item in enumerate(points)
    )
    grid: list[str] = []
    labels: list[str] = []
    for line in range(5):
        ratio = line / 4
        y = pad_y + ratio * (height - 2 * pad_y)
        label_price = high - ratio * (high - low)
        grid.append(
            f'<line x1="{pad_x}" y1="{y:.2f}" x2="{width-pad_x}" y2="{y:.2f}" class="grid-line" />'
        )
        labels.append(
            f'<text x="6" y="{y+4:.2f}" class="axis-label">${label_price:,.0f}</text>'
        )

    ranges: list[str] = []
    markers: list[str] = []
    for index, item in enumerate(points):
        contract = item.get("next_candle_forecast")
        if not isinstance(contract, dict):
            continue
        low_value = _finite(contract.get("likely_close_low"))
        high_value = _finite(contract.get("likely_close_high"))
        median_value = _finite(contract.get("median_close"))
        if low_value is None or high_value is None or median_value is None:
            continue
        x, low_y = xy(index, low_value)
        _, high_y = xy(index, high_value)
        _, median_y = xy(index, median_value)
        result = str(item.get("prediction_result") or "PENDING")
        marker_class = {
            "IN_RANGE": "marker-correct",
            "OUT_OF_RANGE": "marker-wrong",
            "PENDING": "marker-pending",
        }.get(result, "marker-muted")
        ranges.append(
            f'<line x1="{x:.2f}" y1="{low_y:.2f}" x2="{x:.2f}" y2="{high_y:.2f}" class="range-line" />'
        )
        markers.append(
            f'<circle cx="{x:.2f}" cy="{median_y:.2f}" r="4.5" class="{marker_class}" />'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Source closes and next-candle probabilistic close ranges">'
        + "".join(grid)
        + "".join(labels)
        + "".join(ranges)
        + f'<path d="{source_path}" class="price-line" />'
        + "".join(markers)
        + "</svg>"
        + '<div class="legend">'
        + '<span class="marker-correct-text">● Close inside range</span>'
        + '<span class="marker-wrong-text">● Close outside range</span>'
        + '<span class="marker-pending-text">● Pending</span>'
        + '<span>│ Probable close range</span>'
        + '<span>— Source candle close</span>'
        + "</div>"
    )


def rows(history: list[dict[str, Any]]) -> str:
    output: list[str] = []
    classes = {
        "IN_RANGE": "correct",
        "OUT_OF_RANGE": "wrong",
        "PENDING": "pending",
        "NOT_SCORED": "muted-result",
        "LEGACY_NOT_SCORED": "muted-result",
    }
    for item in reversed(history[-30:]):
        contract = item.get("next_candle_forecast")
        result = str(item.get("prediction_result") or "PENDING")
        target_close = (
            contract.get("target_close_time")
            if isinstance(contract, dict)
            else item.get("target_candle_close_time")
        )
        actual = number(item.get("actual_close"), 2)
        actual_text = "Waiting for candle close" if actual == "—" else f"${actual}"
        scenario = (
            value(contract.get("scenario")) if isinstance(contract, dict) else "Legacy"
        )
        probability = (
            percent(contract.get("interval_probability"), 0)
            if isinstance(contract, dict)
            else "—"
        )
        output.append(
            "<tr>"
            f'<td>{esc(value(item.get("candle_time") or item.get("run_finished_at")))}</td>'
            f'<td>${esc(number(item.get("price"), 2))}</td>'
            f"<td>{esc(scenario)}</td>"
            f"<td>{esc(price_range(contract))}</td>"
            f"<td>{esc(probability)}</td>"
            f"<td>{esc(value(target_close))}</td>"
            f"<td>{esc(actual_text)}</td>"
            f'<td><span class="result {classes.get(result, "muted-result")}">{esc(result)}</span></td>'
            f"<td>{esc(value(item.get('direction_result')))}</td>"
            f"<td>{esc(value(item.get('action') or item.get('run_status')))}</td>"
            "</tr>"
        )
    return "".join(output) if output else '<tr><td colspan="10" class="empty">No forecasts recorded yet.</td></tr>'


def adaptive_rows(adaptive: dict[str, Any]) -> str:
    metrics = adaptive.get("metrics") if isinstance(adaptive, dict) else None
    if not isinstance(metrics, dict) or not metrics:
        return '<tr><td colspan="8" class="empty">Adaptive metrics are not available yet.</td></tr>'
    active = {int(item) for item in adaptive.get("active_horizons", [])}
    output: list[str] = []
    for key in sorted(metrics, key=lambda item: int(item)):
        horizon = int(key)
        item = metrics[key] if isinstance(metrics[key], dict) else {}
        status = "ACTIVE" if horizon in active else "SHADOW"
        status_class = "correct" if status == "ACTIVE" else "pending"
        output.append(
            "<tr>"
            f"<td>{horizon}h</td>"
            f'<td><span class="result {status_class}">{status}</span></td>'
            f'<td>{esc(value(item.get("direction_samples")))}</td>'
            f'<td>{esc(value(item.get("event_samples")))}</td>'
            f'<td>{esc(number(item.get("base_direction_brier"), 4))}</td>'
            f'<td>{esc(number(item.get("online_direction_brier"), 4))}</td>'
            f'<td>{esc(percent(item.get("base_direction_accuracy"), 1))}</td>'
            f'<td>{esc(percent(item.get("online_direction_accuracy"), 1))}</td>'
            "</tr>"
        )
    return "".join(output)


def render(latest: dict[str, Any], history: list[dict[str, Any]]) -> str:
    action = value(latest.get("action") or latest.get("status") or latest.get("run_status"))
    action_class = "wait" if action == "WAIT" else "bad" if action in {"FAIL_SAFE", "BLOCKED"} else "good"
    run_ok = latest.get("run_status") == "OK" and latest.get("status") != "FAIL_SAFE"
    inside, scored, coverage = accuracy(history)
    direction_correct, direction_scored, direction_rate = direction_accuracy(history)
    coverage_text = "—" if coverage is None else f"{coverage * 100:.1f}%"
    direction_text = "—" if direction_rate is None else f"{direction_rate * 100:.1f}%"
    blockers = latest.get("blockers")
    blockers_text = "\n".join(map(str, blockers)) if isinstance(blockers, list) and blockers else "No decision blockers were recorded."
    adaptive = latest.get("adaptive") if isinstance(latest.get("adaptive"), dict) else {}
    adaptive_status = value(adaptive.get("status"))
    active_horizons = adaptive.get("active_horizons") or []
    active_text = ", ".join(f"{item}h" for item in active_horizons) or "None"
    contract = latest.get("next_candle_forecast") if isinstance(latest.get("next_candle_forecast"), dict) else {}
    latest_range = price_range(contract)
    interval_probability = percent(contract.get("interval_probability"), 0)
    target_close_time = value(contract.get("target_close_time"))
    scenario = value(contract.get("scenario"))
    reference_close = number(contract.get("reference_close", latest.get("price")), 2)
    reference_close = "—" if reference_close == "—" else f"${reference_close}"
    median_close = number(contract.get("median_close"), 2)
    median_close = "—" if median_close == "—" else f"${median_close}"
    edge = number(latest.get("expected_net_edge_bps"), 1)
    edge = "—" if edge == "—" else f"{edge} bps"
    details = {
        "source_candle_open_time": latest.get("candle_time"),
        "source_candle_close_time": contract.get("source_close_time"),
        "target_candle_open_time": contract.get("target_open_time"),
        "target_candle_close_time": contract.get("target_close_time"),
        "interval_method": contract.get("interval_method"),
        "calibration_samples": contract.get("calibration_samples"),
        "provider": latest.get("provider"),
        "model_id": latest.get("model_id"),
        "adaptive_status": adaptive_status,
        "adaptive_decision_source": adaptive.get("decision_source"),
        "latest_prediction_result": latest.get("prediction_result"),
        "resolved_at": latest.get("resolved_at"),
        "data_health": latest.get("data_health"),
        "error": latest.get("error"),
    }
    details_text = json.dumps(json_safe(details), ensure_ascii=False, indent=2, allow_nan=False)

    return f'''<!doctype html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <meta http-equiv="refresh" content="300">
  <meta name="theme-color" content="#090d16">
  <title>BTC Next-Candle Forecast</title>
  <style>
    :root{{color-scheme:dark;--line:#263246;--text:#eef2ff;--muted:#9ca3af;--good:#34d399;--bad:#fb7185;--wait:#fbbf24;--blue:#60a5fa}}
    *{{box-sizing:border-box}}html{{-webkit-text-size-adjust:100%}}body{{margin:0;min-height:100vh;background:radial-gradient(circle at top,#172033 0,#090d16 48%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
    .wrap{{width:min(1240px,94%);margin:28px auto 60px;padding-bottom:env(safe-area-inset-bottom)}}header{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:18px}}h1{{margin:0;font-size:clamp(28px,7vw,44px);letter-spacing:-.03em}}.sub{{color:var(--muted);margin-top:8px;line-height:1.65;max-width:820px}}.badge{{padding:9px 13px;border:1px solid var(--line);border-radius:999px;white-space:nowrap;background:#0d1422;font-weight:700}}
    .grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}}.card{{min-width:0;background:rgba(17,24,39,.94);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 16px 40px rgba(0,0,0,.18)}}.label{{color:var(--muted);font-size:13px;margin-bottom:9px}}.value{{font-size:clamp(18px,4vw,25px);font-weight:750;overflow-wrap:anywhere}}.meta{{color:var(--muted);font-size:12px;margin-top:6px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.wait{{color:var(--wait)}}.wide{{grid-column:span 5}}.half{{grid-column:span 2}}.section{{margin-top:14px}}.section h2{{font-size:18px;margin:0 0 12px}}
    .chart{{width:100%;height:auto;min-height:240px;display:block;border:1px solid var(--line);border-radius:12px;background:#0b1220}}.grid-line{{stroke:#263246;stroke-width:1}}.axis-label{{fill:#94a3b8;font-size:11px}}.price-line{{fill:none;stroke:var(--blue);stroke-width:2.5}}.range-line{{stroke:#64748b;stroke-width:4;stroke-linecap:round;opacity:.7}}.marker-correct{{fill:var(--good)}}.marker-wrong{{fill:var(--bad)}}.marker-pending{{fill:var(--wait)}}.marker-muted{{fill:#94a3b8}}.marker-correct-text{{color:var(--good)}}.marker-wrong-text{{color:var(--bad)}}.marker-pending-text{{color:var(--wait)}}.legend{{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:12px;margin-top:10px}}.empty-chart{{height:220px;display:grid;place-items:center;color:var(--muted);border:1px solid var(--line);border-radius:12px}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:11px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--muted);font-weight:600;position:sticky;top:0;background:#111827}}.scroll{{overflow:auto;max-height:480px;-webkit-overflow-scrolling:touch}}.empty{{text-align:center;color:var(--muted)}}.result{{display:inline-flex;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:800;letter-spacing:.04em;border:1px solid currentColor}}.correct{{color:var(--good);background:rgba(52,211,153,.08)}}.wrong{{color:var(--bad);background:rgba(251,113,133,.08)}}.pending{{color:var(--wait);background:rgba(251,191,36,.08)}}.muted-result{{color:#94a3b8;background:rgba(148,163,184,.08)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#d1d5db;margin:0;font-size:12px}}footer{{color:var(--muted);text-align:center;margin-top:20px;line-height:1.8}}
    @media(max-width:1050px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.wide,.half{{grid-column:span 2}}header{{flex-direction:column}}}}@media(max-width:520px){{.wrap{{width:92%;margin-top:18px}}.grid{{grid-template-columns:1fr}}.wide,.half{{grid-column:span 1}}.card{{padding:14px;border-radius:14px}}.chart{{min-height:190px}}h1{{font-size:30px}}}}
  </style>
</head>
<body><div class="wrap">
<header><div><h1>BTC Next-Candle Forecast</h1><div class="sub">Every forecast is created only after an hourly candle closes. It estimates an {esc(interval_probability)} probable range for the next candle close. Results remain pending until the target candle has fully closed and are immutable after resolution.</div></div><div class="badge {'good' if run_ok else 'bad'}">{'Run successful' if run_ok else 'FAIL-SAFE'}</div></header>
<main class="grid">
<div class="card"><div class="label">Source candle close</div><div class="value">{esc(reference_close)}</div></div>
<div class="card"><div class="label">Next close range</div><div class="value">{esc(latest_range)}</div></div>
<div class="card"><div class="label">Median estimate</div><div class="value">{esc(median_close)}</div><div class="meta">Not a guaranteed price</div></div>
<div class="card"><div class="label">Scenario</div><div class="value">{esc(scenario)}</div></div>
<div class="card"><div class="label">Evaluation after</div><div class="value">{esc(target_close_time)}</div></div>
<div class="card"><div class="label">Decision</div><div class="value {action_class}">{esc(action)}</div></div>
<div class="card"><div class="label">Interval coverage</div><div class="value">{esc(coverage_text)}</div><div class="meta">{inside} of {scored} resolved closes inside range</div></div>
<div class="card"><div class="label">Direction accuracy</div><div class="value">{esc(direction_text)}</div><div class="meta">{direction_correct} of {direction_scored} directional forecasts</div></div>
<div class="card"><div class="label">Expected net edge</div><div class="value">{esc(edge)}</div></div>
<div class="card"><div class="label">Adaptive learner</div><div class="value">{esc(adaptive_status)}</div><div class="meta">Active horizons: {esc(active_text)}</div></div>
<section class="card wide section"><h2>Source closes and probable next-close ranges</h2>{chart(history)}</section>
<section class="card wide section"><h2>Adaptive learning performance</h2><div class="scroll"><table><thead><tr><th>Horizon</th><th>Mode</th><th>Direction samples</th><th>Event samples</th><th>Base Brier</th><th>Online Brier</th><th>Base accuracy</th><th>Online accuracy</th></tr></thead><tbody>{adaptive_rows(adaptive)}</tbody></table></div></section>
<section class="card half section"><h2>Forecast timing and model status</h2><pre>{esc(details_text)}</pre></section>
<section class="card half section"><h2>Trade decision blockers</h2><pre>{esc(blockers_text)}</pre></section>
<section class="card wide section"><h2>Latest 30 next-candle forecasts</h2><div class="scroll"><table><thead><tr><th>Source candle UTC</th><th>Source close</th><th>Scenario</th><th>Probable close range</th><th>Interval</th><th>Target closes at</th><th>Actual close</th><th>Range result</th><th>Direction result</th><th>Action</th></tr></thead><tbody>{rows(history)}</tbody></table></div></section>
</main><footer>Research and paper-trading only. The median is an estimate and the range is probabilistic, not guaranteed.<br>Last server run: {esc(value(latest.get('run_finished_at')))}</footer>
</div></body></html>'''


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
        load_candles(runtime_dir / "btc_hourly.sqlite3", provider),
    )[-MAX_HISTORY:]
    latest_key = latest.get("candle_time") or latest.get("run_finished_at")
    latest = next(
        (
            item
            for item in reversed(history)
            if (item.get("candle_time") or item.get("run_finished_at")) == latest_key
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
    (site_dir / "index.html").write_text(render(latest, history), encoding="utf-8")
    inside, scored, coverage = accuracy(history)
    direction_correct, direction_scored, direction_rate = direction_accuracy(history)
    print(
        json.dumps(
            {
                "dashboard": "updated",
                "resolved_intervals": scored,
                "inside_range": inside,
                "interval_coverage": coverage,
                "resolved_directions": direction_scored,
                "correct_directions": direction_correct,
                "direction_accuracy": direction_rate,
                "adaptive_status": (
                    latest.get("adaptive", {}).get("status")
                    if isinstance(latest.get("adaptive"), dict)
                    else None
                ),
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
        for row in candles.drop_duplicates("open_time", keep="last").itertuples(index=False)
        if not pd.isna(row.close)
    }


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _finite(value: Any) -> float | None:
    try:
        number_value = float(value)
    except (TypeError, ValueError):
        return None
    return number_value if np.isfinite(number_value) else None


if __name__ == "__main__":
    raise SystemExit(main())
