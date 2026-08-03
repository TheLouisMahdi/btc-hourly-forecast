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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve forecast outcomes and render the static dashboard"
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
    output.setdefault("entry_price", None)
    output.setdefault("actual_direction", None)
    output.setdefault("actual_price", None)
    output.setdefault("actual_return", None)
    output.setdefault("target_candle_time", None)
    return output


def resolve_outcomes(
    history: list[dict[str, Any]],
    candles: pd.DataFrame,
) -> list[dict[str, Any]]:
    if candles.empty:
        return [pending(item) for item in history]
    candle_map = {
        pd.Timestamp(row.open_time): {
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
    resolved: list[dict[str, Any]] = []
    for source in history:
        item = dict(source)
        direction = str(
            item.get("forecast_direction") or ""
        ).upper()
        if (
            item.get("run_status") != "OK"
            or direction not in {"UP", "DOWN"}
        ):
            resolved.append(pending(item, "NOT_SCORED"))
            continue
        try:
            candle_time = _utc(item["candle_time"])
            horizon = max(
                1,
                int(item.get("selected_horizon") or 1),
            )
        except (KeyError, TypeError, ValueError):
            resolved.append(pending(item, "NOT_SCORED"))
            continue
        entry_time = candle_time + pd.Timedelta(hours=1)
        target_time = candle_time + pd.Timedelta(hours=horizon)
        item["target_candle_time"] = target_time.isoformat()
        entry_candle = candle_map.get(entry_time)
        target_candle = candle_map.get(target_time)
        if entry_candle is None or target_candle is None:
            resolved.append(pending(item))
            continue
        entry_price = float(entry_candle["open"])
        target_price = float(target_candle["close"])
        if entry_price <= 0:
            resolved.append(pending(item, "NOT_SCORED"))
            continue
        actual_return = target_price / entry_price - 1.0
        actual_direction = (
            "UP"
            if actual_return > 0
            else "DOWN"
            if actual_return < 0
            else "FLAT"
        )
        item.update(
            {
                "prediction_result": (
                    "CORRECT"
                    if actual_direction == direction
                    else "WRONG"
                ),
                "entry_price": entry_price,
                "actual_direction": actual_direction,
                "actual_price": target_price,
                "actual_return": actual_return,
            }
        )
        resolved.append(item)
    return resolved


def accuracy(
    history: list[dict[str, Any]],
) -> tuple[int, int, float | None]:
    scored = [
        item
        for item in history
        if item.get("prediction_result") in {"CORRECT", "WRONG"}
    ]
    correct = sum(
        item.get("prediction_result") == "CORRECT"
        for item in scored
    )
    return (
        correct,
        len(scored),
        correct / len(scored) if scored else None,
    )


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


def chart(history: list[dict[str, Any]]) -> str:
    points = [
        item
        for item in history[-168:]
        if _finite(item.get("price")) is not None
    ]
    if len(points) < 2:
        return (
            '<div class="empty-chart">'
            "Not enough history to draw the chart."
            "</div>"
        )
    width = 1120
    height = 330
    pad_x = 38
    pad_y = 28
    prices = [float(item["price"]) for item in points]
    low = min(prices)
    high = max(prices)
    margin = max((high - low) * 0.12, 1.0)
    low -= margin
    high += margin

    def xy(index: int, price: float) -> tuple[float, float]:
        x = (
            pad_x
            + index
            * (width - 2 * pad_x)
            / max(1, len(points) - 1)
        )
        y = (
            height
            - pad_y
            - (price - low)
            * (height - 2 * pad_y)
            / (high - low)
        )
        return x, y

    path = " ".join(
        ("M" if index == 0 else "L")
        + f" {xy(index, price)[0]:.2f} {xy(index, price)[1]:.2f}"
        for index, price in enumerate(prices)
    )
    grid = []
    labels = []
    for line in range(5):
        ratio = line / 4
        y = pad_y + ratio * (height - 2 * pad_y)
        price = high - ratio * (high - low)
        grid.append(
            f'<line x1="{pad_x}" y1="{y:.2f}" '
            f'x2="{width-pad_x}" y2="{y:.2f}" '
            'class="grid-line" />'
        )
        labels.append(
            f'<text x="6" y="{y+4:.2f}" '
            f'class="axis-label">${price:,.0f}</text>'
        )

    markers = []
    for index, item in enumerate(points):
        result = str(item.get("prediction_result") or "PENDING")
        direction = str(item.get("forecast_direction") or "")
        if direction not in {"UP", "DOWN"}:
            continue
        x, y = xy(index, float(item["price"]))
        marker_class = {
            "CORRECT": "marker-correct",
            "WRONG": "marker-wrong",
            "PENDING": "marker-pending",
        }.get(result, "marker-muted")
        shape = "▲" if direction == "UP" else "▼"
        markers.append(
            f'<text x="{x:.2f}" y="{y-9:.2f}" '
            f'text-anchor="middle" '
            f'class="forecast-marker {marker_class}">{shape}</text>'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="BTC price and forecast outcome chart">'
        + "".join(grid)
        + "".join(labels)
        + f'<path d="{path}" class="price-line" />'
        + "".join(markers)
        + "</svg>"
        + '<div class="legend">'
        + '<span class="marker-correct">● Correct</span>'
        + '<span class="marker-wrong">● Wrong</span>'
        + '<span class="marker-pending">● Pending</span>'
        + "<span>▲ Up forecast</span>"
        + "<span>▼ Down forecast</span>"
        + "</div>"
    )


def rows(history: list[dict[str, Any]]) -> str:
    output = []
    classes = {
        "CORRECT": "correct",
        "WRONG": "wrong",
        "PENDING": "pending",
        "NOT_SCORED": "muted-result",
    }
    for item in reversed(history[-30:]):
        result = str(item.get("prediction_result") or "PENDING")
        actual_direction = item.get("actual_direction")
        actual_price = number(item.get("actual_price"), 2)
        entry_price = number(item.get("entry_price"), 2)
        actual = (
            f"{actual_direction} · entry ${entry_price} · "
            f"close ${actual_price} · "
            f"{percent(item.get('actual_return'))}"
            if actual_direction
            and actual_price != "—"
            and entry_price != "—"
            else "Waiting for target candle"
        )
        horizon = (
            "—"
            if item.get("selected_horizon") in (None, "")
            else f"{item.get('selected_horizon')}h"
        )
        edge = number(item.get("expected_net_edge_bps"), 1)
        edge = "—" if edge == "—" else f"{edge} bps"
        action = value(
            item.get("action")
            or item.get("status")
            or item.get("run_status")
        )
        output.append(
            "<tr>"
            f'<td>{esc(value(item.get("candle_time") or item.get("run_finished_at")))}</td>'
            f'<td>${esc(number(item.get("price"), 2))}</td>'
            f'<td><strong>{esc(value(item.get("forecast_direction")))}</strong></td>'
            f"<td>{esc(horizon)}</td>"
            f"<td>{esc(actual)}</td>"
            f'<td><span class="result {classes.get(result, "muted-result")}">{esc(result)}</span></td>'
            f"<td>{esc(action)}</td>"
            f"<td>{esc(percent(item.get('confidence')))}</td>"
            f"<td>{esc(edge)}</td>"
            "</tr>"
        )
    return (
        "".join(output)
        if output
        else (
            '<tr><td colspan="9" class="empty">'
            "No forecasts recorded yet."
            "</td></tr>"
        )
    )


def adaptive_rows(adaptive: dict[str, Any]) -> str:
    metrics = adaptive.get("metrics") if isinstance(adaptive, dict) else None
    if not isinstance(metrics, dict) or not metrics:
        return (
            '<tr><td colspan="8" class="empty">'
            "Adaptive metrics are not available yet."
            "</td></tr>"
        )
    active = {
        int(item) for item in adaptive.get("active_horizons", [])
    }
    output = []
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


def render(
    latest: dict[str, Any],
    history: list[dict[str, Any]],
) -> str:
    action = value(
        latest.get("action")
        or latest.get("status")
        or latest.get("run_status")
    )
    action_class = (
        "wait"
        if action == "WAIT"
        else "bad"
        if action in {"FAIL_SAFE", "BLOCKED"}
        else "good"
    )
    run_ok = (
        latest.get("run_status") == "OK"
        and latest.get("status") != "FAIL_SAFE"
    )
    correct, scored, rate = accuracy(history)
    accuracy_text = "—" if rate is None else f"{rate * 100:.1f}%"
    blockers = latest.get("blockers")
    blockers_text = (
        "\n".join(map(str, blockers))
        if isinstance(blockers, list) and blockers
        else "No decision blockers were recorded."
    )
    adaptive = (
        latest.get("adaptive")
        if isinstance(latest.get("adaptive"), dict)
        else {}
    )
    adaptive_status = value(adaptive.get("status"))
    active_horizons = adaptive.get("active_horizons") or []
    active_text = (
        ", ".join(f"{item}h" for item in active_horizons)
        or "None"
    )
    details = {
        "candle_time": latest.get("candle_time"),
        "created_at": latest.get("created_at"),
        "provider": latest.get("provider"),
        "event_type": latest.get("event_type"),
        "model_id": latest.get("model_id"),
        "qualification_passed": latest.get("qualification_passed"),
        "weekly_model_loaded": latest.get("weekly_model_loaded"),
        "latest_prediction_result": latest.get("prediction_result"),
        "target_candle_time": latest.get("target_candle_time"),
        "adaptive_status": adaptive_status,
        "adaptive_decision_source": adaptive.get("decision_source"),
        "adaptive_rebase_count": adaptive.get("rebase_count"),
        "data_health": latest.get("data_health"),
        "market_refresh": latest.get("market_refresh"),
        "error": latest.get("error"),
    }
    details_text = json.dumps(
        json_safe(details),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    price = number(latest.get("price"), 2)
    price = "—" if price == "—" else f"${price}"
    horizon = (
        "—"
        if latest.get("selected_horizon") in (None, "")
        else f"{latest.get('selected_horizon')}h"
    )
    edge = number(latest.get("expected_net_edge_bps"), 1)
    edge = "—" if edge == "—" else f"{edge} bps"
    return f'''<!doctype html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <meta http-equiv="refresh" content="300">
  <meta name="theme-color" content="#090d16">
  <title>BTC Hourly Forecast</title>
  <style>
    :root{{color-scheme:dark;--line:#263246;--text:#eef2ff;--muted:#9ca3af;--good:#34d399;--bad:#fb7185;--wait:#fbbf24;--blue:#60a5fa}}
    *{{box-sizing:border-box}}
    html{{-webkit-text-size-adjust:100%}}
    body{{margin:0;min-height:100vh;background:radial-gradient(circle at top,#172033 0,#090d16 48%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
    .wrap{{width:min(1240px,94%);margin:28px auto 60px;padding-bottom:env(safe-area-inset-bottom)}}
    header{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:18px}}
    h1{{margin:0;font-size:clamp(28px,7vw,44px);letter-spacing:-.03em}}
    .sub{{color:var(--muted);margin-top:8px;line-height:1.65;max-width:780px}}
    .badge{{padding:9px 13px;border:1px solid var(--line);border-radius:999px;white-space:nowrap;background:#0d1422;font-weight:700}}
    .grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}}
    .card{{min-width:0;background:rgba(17,24,39,.94);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 16px 40px rgba(0,0,0,.18)}}
    .label{{color:var(--muted);font-size:13px;margin-bottom:9px}}
    .value{{font-size:clamp(19px,4vw,25px);font-weight:750;overflow-wrap:anywhere}}
    .meta{{color:var(--muted);font-size:12px;margin-top:6px}}
    .good{{color:var(--good)}}
    .bad{{color:var(--bad)}}
    .wait{{color:var(--wait)}}
    .wide{{grid-column:span 5}}
    .half{{grid-column:span 2}}
    .section{{margin-top:14px}}
    .section h2{{font-size:18px;margin:0 0 12px}}
    .chart{{width:100%;height:auto;min-height:240px;display:block;border:1px solid var(--line);border-radius:12px;background:#0b1220}}
    .grid-line{{stroke:#263246;stroke-width:1}}
    .axis-label{{fill:#94a3b8;font-size:11px}}
    .price-line{{fill:none;stroke:var(--blue);stroke-width:2.5}}
    .forecast-marker{{font-size:17px;font-weight:800}}
    .marker-correct{{fill:var(--good);color:var(--good)}}
    .marker-wrong{{fill:var(--bad);color:var(--bad)}}
    .marker-pending{{fill:var(--wait);color:var(--wait)}}
    .marker-muted{{fill:#94a3b8;color:#94a3b8}}
    .legend{{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:12px;margin-top:10px}}
    .empty-chart{{height:220px;display:grid;place-items:center;color:var(--muted);border:1px solid var(--line);border-radius:12px}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    th,td{{padding:11px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
    th{{color:var(--muted);font-weight:600;position:sticky;top:0;background:#111827}}
    .scroll{{overflow:auto;max-height:480px;-webkit-overflow-scrolling:touch}}
    .empty{{text-align:center;color:var(--muted)}}
    .result{{display:inline-flex;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:800;letter-spacing:.04em;border:1px solid currentColor}}
    .correct{{color:var(--good);background:rgba(52,211,153,.08)}}
    .wrong{{color:var(--bad);background:rgba(251,113,133,.08)}}
    .pending{{color:var(--wait);background:rgba(251,191,36,.08)}}
    .muted-result{{color:#94a3b8;background:rgba(148,163,184,.08)}}
    pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#d1d5db;margin:0;font-size:12px}}
    footer{{color:var(--muted);text-align:center;margin-top:20px;line-height:1.8}}
    @media(max-width:1050px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.wide,.half{{grid-column:span 2}}header{{flex-direction:column}}}}
    @media(max-width:520px){{.wrap{{width:92%;margin-top:18px}}.grid{{grid-template-columns:1fr}}.wide,.half{{grid-column:span 1}}.card{{padding:14px;border-radius:14px}}.chart{{min-height:190px}}h1{{font-size:30px}}}}
  </style>
</head>
<body>
<div class="wrap">
<header><div><h1>BTC Hourly Forecast</h1><div class="sub">Adaptive BTCUSDT one-hour forecasting with delayed labels, prequential evaluation, drift protection and fail-safe paper-trade qualification. The page refreshes every five minutes.</div></div><div class="badge {'good' if run_ok else 'bad'}">{'Run successful' if run_ok else 'FAIL-SAFE'}</div></header>
<main class="grid">
<div class="card"><div class="label">Decision</div><div class="value {action_class}">{esc(action)}</div></div>
<div class="card"><div class="label">Forecast direction</div><div class="value">{esc(value(latest.get('forecast_direction')))}</div></div>
<div class="card"><div class="label">Latest candle close</div><div class="value">{esc(price)}</div></div>
<div class="card"><div class="label">Selected horizon</div><div class="value">{esc(horizon)}</div></div>
<div class="card"><div class="label">Confidence</div><div class="value">{esc(percent(latest.get('confidence')))}</div></div>
<div class="card"><div class="label">Tradeability probability</div><div class="value">{esc(percent(latest.get('tradeability_probability')))}</div></div>
<div class="card"><div class="label">Expected net edge</div><div class="value">{esc(edge)}</div></div>
<div class="card"><div class="label">Market regime</div><div class="value">{esc(value(latest.get('regime')))}</div></div>
<div class="card"><div class="label">Resolved direction accuracy</div><div class="value">{esc(accuracy_text)}</div><div class="meta">{scored} resolved</div></div>
<div class="card"><div class="label">Adaptive learner</div><div class="value">{esc(adaptive_status)}</div><div class="meta">Active horizons: {esc(active_text)}</div></div>
<section class="card wide section"><h2>Price and forecast outcomes</h2>{chart(history)}</section>
<section class="card wide section"><h2>Adaptive learning performance</h2><div class="scroll"><table><thead><tr><th>Horizon</th><th>Mode</th><th>Direction samples</th><th>Event samples</th><th>Base Brier</th><th>Online Brier</th><th>Base accuracy</th><th>Online accuracy</th></tr></thead><tbody>{adaptive_rows(adaptive)}</tbody></table></div></section>
<section class="card half section"><h2>Model and run status</h2><pre>{esc(details_text)}</pre></section>
<section class="card half section"><h2>Decision blockers</h2><pre>{esc(blockers_text)}</pre></section>
<section class="card wide section"><h2>Latest 30 forecasts</h2><div class="scroll"><table><thead><tr><th>Candle UTC</th><th>Source close</th><th>Forecast</th><th>Horizon</th><th>Actual outcome</th><th>Result</th><th>Action</th><th>Confidence</th><th>Edge</th></tr></thead><tbody>{rows(history)}</tbody></table></div></section>
</main>
<footer>Research and paper-trading only. This dashboard is not financial advice.<br>Last server run: {esc(value(latest.get('run_finished_at')))}</footer>
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
    correct, scored, rate = accuracy(history)
    print(
        json.dumps(
            {
                "dashboard": "updated",
                "resolved": scored,
                "correct": correct,
                "accuracy": rate,
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


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _finite(value: Any) -> float | None:
    try:
        number_value = float(value)
    except (TypeError, ValueError):
        return None
    return number_value if np.isfinite(number_value) else None


if __name__ == "__main__":
    raise SystemExit(main())
