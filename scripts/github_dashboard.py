from __future__ import annotations

import argparse
import html
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from github_common import write_json

MAX_HISTORY = 24 * 30
SETTLEMENT_DELAY_SECONDS = 90
FINAL_DIRECTION_RESULTS = {"DIRECTION_CORRECT", "DIRECTION_WRONG"}
FINAL_INTERVAL_RESULTS = {"IN_RANGE", "OUT_OF_RANGE"}


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


def load_candles(database_path: Path, provider: str | None) -> pd.DataFrame:
    if not database_path.exists():
        return pd.DataFrame()
    clauses = ["closed = 1"]
    params: list[Any] = []
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    query = (
        "SELECT open_time, open, high, low, close FROM candles WHERE "
        + " AND ".join(clauses)
        + " ORDER BY open_time"
    )
    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(query, connection, params=params)
    if not frame.empty:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    return frame


def pending(item: dict[str, Any], result: str = "PENDING") -> dict[str, Any]:
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
    output: list[dict[str, Any]] = []

    for source in history:
        item = dict(source)
        if item.get("resolved_at") and (
            item.get("direction_result") in FINAL_DIRECTION_RESULTS
            or item.get("direction_result") == "LEGACY_NOT_SCORED"
        ):
            item["prediction_result"] = item["direction_result"]
            output.append(item)
            continue
        if item.get("run_status") != "OK":
            output.append(pending(item, "NOT_SCORED"))
            continue

        contract = item.get("next_candle_forecast")
        if not isinstance(contract, dict):
            output.append(pending(item, "LEGACY_NOT_SCORED"))
            continue

        try:
            target_open = _utc(contract["target_open_time"])
            target_close = _utc(contract["target_close_time"])
            reference_close = float(contract["reference_close"])
            predicted_low = float(contract["likely_close_low"])
            predicted_high = float(contract["likely_close_high"])
        except (KeyError, TypeError, ValueError):
            output.append(pending(item, "NOT_SCORED"))
            continue

        item["target_candle_open_time"] = target_open.isoformat()
        item["target_candle_close_time"] = target_close.isoformat()
        evaluation_time = target_close + pd.Timedelta(
            seconds=SETTLEMENT_DELAY_SECONDS
        )
        item["evaluation_available_at"] = evaluation_time.isoformat()
        if current_time < evaluation_time:
            item["seconds_until_evaluation"] = max(
                0,
                int((evaluation_time - current_time).total_seconds()),
            )
            output.append(pending(item))
            continue

        target = candle_map.get(target_open)
        if target is None:
            output.append(pending(item))
            continue
        if reference_close <= 0 or predicted_low <= 0 or predicted_high <= 0:
            output.append(pending(item, "NOT_SCORED"))
            continue

        actual_close = float(target["close"])
        actual_return = actual_close / reference_close - 1.0
        actual_direction = (
            "UP" if actual_return > 0 else "DOWN" if actual_return < 0 else "FLAT"
        )
        contract_version = int(contract.get("contract_version") or 1)
        forecast_direction = str(contract.get("direction") or "").upper()
        direction_is_scoreable = (
            contract_version >= 2 and forecast_direction in {"UP", "DOWN"}
        )
        if direction_is_scoreable:
            direction_result = (
                "DIRECTION_CORRECT"
                if actual_direction == forecast_direction
                else "DIRECTION_WRONG"
            )
        else:
            direction_result = "LEGACY_NOT_SCORED"

        interval_result = (
            "IN_RANGE"
            if min(predicted_low, predicted_high)
            <= actual_close
            <= max(predicted_low, predicted_high)
            else "OUT_OF_RANGE"
        )
        item.update(
            {
                "prediction_result": direction_result,
                "direction_result": direction_result,
                "interval_result": interval_result,
                "actual_close": actual_close,
                "actual_price": actual_close,
                "actual_close_return": actual_return,
                "actual_return": actual_return,
                "actual_direction": actual_direction,
                "actual_candle_open": float(target["open"]),
                "actual_candle_high": float(target["high"]),
                "actual_candle_low": float(target["low"]),
                "resolved_at": current_time.isoformat(),
                "seconds_until_evaluation": 0,
            }
        )
        output.append(item)
    return output


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
    return correct, len(scored), correct / len(scored) if scored else None


def interval_coverage(
    history: list[dict[str, Any]],
) -> tuple[int, int, float | None]:
    scored = [
        item
        for item in history
        if item.get("interval_result") in FINAL_INTERVAL_RESULTS
    ]
    inside = sum(item.get("interval_result") == "IN_RANGE" for item in scored)
    return inside, len(scored), inside / len(scored) if scored else None


def render(latest: dict[str, Any], history: list[dict[str, Any]]) -> str:
    contract = latest.get("next_candle_forecast")
    contract = contract if isinstance(contract, dict) else {}
    probability_up = _finite(contract.get("probability_up"))
    raw_direction = str(
        contract.get("direction") or latest.get("forecast_direction") or ""
    ).upper()
    if raw_direction in {"UP", "DOWN"}:
        direction = raw_direction
        legacy_display = False
    elif probability_up is not None:
        direction = "UP" if probability_up >= 0.5 else "DOWN"
        legacy_display = True
    else:
        direction = "—"
        legacy_display = True

    direction_probability = (
        probability_up
        if direction == "UP"
        else 1.0 - probability_up
        if probability_up is not None
        else None
    )
    signal_strength = str(contract.get("signal_strength") or "LOW")
    forecast_source = str(contract.get("forecast_source") or "BATCH_CHAMPION")
    model = latest.get("price_forecast_model")
    model = model if isinstance(model, dict) else {}
    model_metrics = model.get("metrics")
    model_metrics = model_metrics if isinstance(model_metrics, dict) else {}
    direction_weight = _finite(contract.get("direction_blend_weight")) or 0.0
    return_weight = _finite(contract.get("return_blend_weight")) or 0.0
    online_influence = max(direction_weight, return_weight)
    batch_influence = max(0.0, 1.0 - online_influence)

    correct, resolved_directions, direction_rate = direction_accuracy(history)
    inside, resolved_intervals, coverage = interval_coverage(history)
    action = str(
        latest.get("action")
        or latest.get("status")
        or latest.get("run_status")
        or "—"
    )
    evaluation = (
        latest.get("evaluation_available_at")
        if latest.get("prediction_result") == "PENDING"
        else latest.get("resolved_at")
    )
    direction_note = (
        "Legacy display only · excluded from direction accuracy"
        if legacy_display
        else "Primary forecast · scored after the target candle closes"
    )

    css = """
:root{color-scheme:light;--bg:#f3f7f5;--paper:rgba(255,255,255,.78);--ink:#263733;--muted:#74837f;--line:rgba(65,96,88,.14);--sage:#6f9b91;--sage2:#47746b;--mint:#deece7;--lav:#8f8ab8;--lav2:#eceaf5;--peach:#c98f79;--peach2:#f5e6df;--ok:#4d8b76;--bad:#bd726e;--wait:#9a865d;--shadow:0 22px 70px rgba(55,80,73,.10)}
*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:radial-gradient(circle at 10% 5%,rgba(222,236,231,.95),transparent 32%),radial-gradient(circle at 92% 2%,rgba(236,234,245,.9),transparent 30%),linear-gradient(180deg,#fafcfb,var(--bg))}.shell{width:min(1240px,92%);margin:auto;padding:26px 0 42px}.top{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:24px}.brand{display:flex;align-items:center;gap:12px}.logo{width:44px;height:44px;border-radius:15px;display:grid;place-items:center;background:linear-gradient(135deg,var(--mint),var(--lav2));color:var(--sage2);font-weight:900}.brand h1{margin:0;font-size:18px;letter-spacing:-.02em}.brand p{margin:3px 0 0;color:var(--muted);font-size:12px}.health{display:flex;align-items:center;gap:8px;padding:9px 13px;border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.72);font-size:12px;font-weight:750}.health i{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 5px rgba(77,139,118,.12)}.hero{display:grid;grid-template-columns:1.15fr .85fr;gap:22px;padding:clamp(24px,4vw,46px);border:1px solid rgba(255,255,255,.9);border-radius:34px;background:linear-gradient(135deg,rgba(255,255,255,.96),rgba(247,250,249,.76));box-shadow:var(--shadow)}.eyebrow{color:var(--sage2);font-size:11px;font-weight:850;letter-spacing:.1em;text-transform:uppercase}.direction-row{display:flex;align-items:end;flex-wrap:wrap;gap:14px;margin:19px 0 10px}.direction{font-size:clamp(58px,10vw,104px);line-height:.88;letter-spacing:-.075em;font-weight:880}.direction.up{color:var(--sage2)}.direction.down{color:#a56663}.confidence{margin-bottom:8px;padding:9px 12px;border-radius:14px;background:var(--mint);color:var(--sage2);font-size:13px;font-weight:820}.copy{max-width:650px;color:var(--muted);font-size:14px;line-height:1.7}.copy strong{color:var(--ink)}.prob{margin-top:25px}.prob-head{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-bottom:8px}.track{height:12px;border-radius:99px;overflow:hidden;background:var(--peach2)}.fill{height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--sage),#87aca4)}.forecast-card{padding:22px;border:1px solid var(--line);border-radius:25px;background:rgba(246,249,248,.72)}.label{color:var(--muted);font-size:12px}.expected{margin-top:7px;font-size:clamp(27px,4vw,39px);font-weight:850;letter-spacing:-.04em}.range{margin-top:18px;padding:14px;border-radius:18px;background:linear-gradient(90deg,var(--peach2),var(--lav2),var(--mint))}.range-values{display:flex;justify-content:space-between;gap:8px;font-size:11px;color:#667570}.meta{display:flex;justify-content:space-between;gap:12px;padding-top:14px;margin-top:14px;border-top:1px solid var(--line);font-size:12px}.meta span{color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:18px}.metric,.panel{background:var(--paper);border:1px solid rgba(255,255,255,.88);box-shadow:0 14px 42px rgba(55,80,73,.06);backdrop-filter:blur(14px)}.metric{padding:20px;border-radius:23px}.metric strong{display:block;margin-top:8px;font-size:25px;letter-spacing:-.035em}.metric small{display:block;margin-top:6px;color:var(--muted);line-height:1.45}.grid{display:grid;grid-template-columns:1.45fr .55fr;gap:18px;margin-top:18px}.panel{min-width:0;padding:24px;border-radius:28px}.panel h2{margin:0;font-size:17px}.sub{margin:5px 0 18px;color:var(--muted);font-size:12px}.chart{width:100%;min-height:280px;display:block}.grid-line{stroke:rgba(65,96,88,.10)}.axis{fill:#87938f;font-size:10px}.line{fill:none;stroke:var(--sage2);stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.band{fill:rgba(143,138,184,.18)}.median{stroke:var(--lav);stroke-width:2.5;stroke-dasharray:6 5}.dot{stroke-width:2}.correct{fill:#e3f0eb;stroke:var(--ok)}.wrong{fill:#f5e6e4;stroke:var(--bad)}.pending-dot{fill:#f3eddf;stroke:var(--wait)}.symbol{fill:var(--ink);font-size:12px;font-weight:900}.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:11px;margin-top:8px}.legend span{display:flex;align-items:center;gap:6px}.legend i{width:8px;height:8px;border-radius:50%;display:inline-block}.learning{display:grid;gap:12px}.learn{padding:15px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.58)}.learn-head{display:flex;justify-content:space-between;font-size:12px}.learn-head span:first-child{color:var(--muted)}.mini{height:7px;margin-top:9px;border-radius:99px;overflow:hidden;background:#edf1ef}.mini div{height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--sage),var(--lav))}.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{padding:8px 10px;border-radius:999px;background:rgba(245,230,223,.78);color:#846257;border:1px solid rgba(201,143,121,.16);font-size:10px;font-weight:760}.ledger{margin-top:18px}.scroll{overflow:auto;max-height:520px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.48)}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:13px 12px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}th{position:sticky;top:0;background:rgba(246,249,248,.97);color:var(--muted);z-index:2}.pill{display:inline-flex;padding:5px 8px;border-radius:999px;font-size:10px;font-weight:850}.pill.up{background:var(--mint);color:var(--sage2)}.pill.down{background:var(--peach2);color:#9c625f}.result-ok{color:var(--ok);background:rgba(77,139,118,.10)}.result-bad{color:var(--bad);background:rgba(189,114,110,.10)}.result-wait{color:var(--wait);background:rgba(154,134,93,.10)}.result-muted{color:var(--muted);background:rgba(116,131,127,.08)}.empty{padding:56px;text-align:center;color:var(--muted)}footer{display:flex;justify-content:space-between;gap:16px;margin-top:24px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:11px}footer a{color:var(--sage2);font-weight:800;text-decoration:none}@media(max-width:980px){.hero,.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:620px){.shell{width:93%;padding-top:18px}.brand p{display:none}.hero{padding:22px;border-radius:26px}.direction{font-size:62px}.metrics{grid-template-columns:1fr}.panel{padding:18px;border-radius:22px}footer{flex-direction:column}}
"""

    probability_width = max(
        0.0,
        min(100.0, (probability_up if probability_up is not None else 0.5) * 100),
    )
    return f'''<!doctype html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta http-equiv="refresh" content="300">
<meta name="theme-color" content="#f3f7f5">
<meta name="description" content="Adaptive next-candle BTC direction and price forecast">
<title>BTC Next-Candle Forecast</title>
<style>{css}</style>
</head>
<body>
<div class="shell">
<nav class="top"><div class="brand"><div class="logo">₿</div><div><h1>BTC Next-Candle Forecast</h1><p>Direction-first forecasting with adaptive price learning</p></div></div><div class="health"><i></i>{'System healthy' if latest.get('run_status') == 'OK' else 'Fail-safe active'}</div></nav>
<main>
<section class="hero">
<div><div class="eyebrow">Next closed 1-hour candle</div><div class="direction-row"><div class="direction {'up' if direction == 'UP' else 'down'}">{esc(direction)}</div><div class="confidence">{esc(percent(direction_probability))} confidence · {esc(signal_strength)}</div></div><p class="copy">The model expects the next close to finish <strong>{'above' if direction == 'UP' else 'below'} {esc(price(contract.get('reference_close')))}</strong>. {esc(direction_note)}. Price uncertainty is reported separately.</p><div class="prob"><div class="prob-head"><span>Down {esc(percent(contract.get('probability_down')))}</span><span>Up {esc(percent(contract.get('probability_up')))}</span></div><div class="track"><div class="fill" style="width:{probability_width:.2f}%"></div></div></div></div>
<aside class="forecast-card"><div class="label">Model-estimated next close</div><div class="expected">{esc(price(contract.get('median_close')))}</div><div class="label" style="margin-top:6px">Fused Batch and Online return estimate</div><div class="range"><div class="range-values"><span>{esc(price(contract.get('likely_close_low')))}</span><strong>{esc(price(contract.get('median_close')))}</strong><span>{esc(price(contract.get('likely_close_high')))}</span></div></div><div class="meta"><span>Forecast source</span><strong>{esc(forecast_source.replace('_', ' ').title())}</strong></div><div class="meta"><span>Target closes</span><strong>{esc(compact_time(contract.get('target_close_time')))}</strong></div><div class="meta"><span>Evaluation</span><strong>{esc(compact_time(evaluation))}</strong></div></aside>
</section>
<section class="metrics"><article class="metric"><span class="label">Direction accuracy</span><strong>{esc(percent(direction_rate))}</strong><small>{correct} correct across {resolved_directions} scored v2 forecasts</small></article><article class="metric"><span class="label">Interval coverage</span><strong>{esc(percent(coverage))}</strong><small>{inside} closes inside {resolved_intervals} intervals</small></article><article class="metric"><span class="label">Decision</span><strong>{esc(action)}</strong><small>Trade gates remain independent from forecast scoring</small></article><article class="metric"><span class="label">Market regime</span><strong>{esc(str(latest.get('regime') or '—').replace('_', ' ').title())}</strong><small>{esc(str(contract.get('scenario') or '—').replace('_', ' ').title())}</small></article></section>
<section class="grid"><article class="panel"><h2>Price path and next forecast</h2><p class="sub">Closed-candle prices, direction outcomes and current model uncertainty</p>{chart(history)}</article><aside class="panel"><h2>Adaptive learning</h2><p class="sub">Online influence is performance-weighted and bounded</p><div class="learning"><div class="learn"><div class="learn-head"><span>Batch influence</span><strong>{esc(percent(batch_influence))}</strong></div><div class="mini"><div style="width:{batch_influence*100:.2f}%"></div></div></div><div class="learn"><div class="learn-head"><span>Online influence</span><strong>{esc(percent(online_influence))}</strong></div><div class="mini"><div style="width:{online_influence*100:.2f}%"></div></div></div><div class="learn"><div class="learn-head"><span>Evaluation samples</span><strong>{esc(value(model_metrics.get('samples')))}</strong></div><p class="sub" style="margin:8px 0 0">Online output receives weight only when its Brier score, direction accuracy and return error remain competitive.</p></div><div class="learn"><div class="learn-head"><span>Batch direction accuracy</span><strong>{esc(percent(model_metrics.get('base_direction_accuracy')))}</strong></div><div class="learn-head" style="margin-top:8px"><span>Online direction accuracy</span><strong>{esc(percent(model_metrics.get('online_direction_accuracy')))}</strong></div></div></div><h2 style="margin-top:23px">Trade blockers</h2><p class="sub">These block execution, not direction evaluation</p><div class="chips">{blocker_chips(latest.get('blockers'))}</div></aside></section>
<section class="panel ledger"><h2>Forecast ledger</h2><p class="sub">Immutable results resolved only after each target candle closes</p><div class="scroll"><table><thead><tr><th>Source candle</th><th>Source close</th><th>Direction</th><th>Confidence</th><th>Expected close</th><th>Model range</th><th>Actual close</th><th>Direction result</th><th>Interval result</th></tr></thead><tbody>{history_rows(history)}</tbody></table></div></section>
</main>
<footer><span>© 2026 Mahdi Ghahremani · ID: TheLouisMahdi · Research and paper-trading only.</span><a href="https://github.com/TheLouisMahdi" target="_blank" rel="noopener noreferrer">GitHub · @TheLouisMahdi</a></footer>
</div>
</body>
</html>'''


def chart(history: list[dict[str, Any]]) -> str:
    points = [
        item
        for item in history[-168:]
        if _finite(item.get("price")) is not None
    ]
    if len(points) < 2:
        return '<div class="empty">More forecast history is required.</div>'
    width, height, pad_x, pad_y = 1120, 350, 42, 28
    prices = [float(item["price"]) for item in points]
    contract = points[-1].get("next_candle_forecast")
    contract = contract if isinstance(contract, dict) else {}
    scale = list(prices)
    for key in ("likely_close_low", "likely_close_high", "median_close"):
        candidate = _finite(contract.get(key))
        if candidate is not None:
            scale.append(candidate)
    low, high = min(scale), max(scale)
    margin = max((high - low) * 0.15, 1.0)
    low, high = low - margin, high + margin

    def xy(index: float, price_value: float) -> tuple[float, float]:
        x = pad_x + index * (width - 2 * pad_x) / max(1, len(points))
        y = height - pad_y - (price_value - low) * (height - 2 * pad_y) / max(
            high - low, 1e-9
        )
        return x, y

    path = " ".join(
        ("M" if index == 0 else "L")
        + f" {xy(index, price_value)[0]:.2f} {xy(index, price_value)[1]:.2f}"
        for index, price_value in enumerate(prices)
    )
    grid: list[str] = []
    labels: list[str] = []
    for index in range(4):
        ratio = index / 3
        y = pad_y + ratio * (height - 2 * pad_y)
        label = high - ratio * (high - low)
        grid.append(
            f'<line x1="{pad_x}" y1="{y:.2f}" x2="{width-pad_x}" '
            f'y2="{y:.2f}" class="grid-line" />'
        )
        labels.append(
            f'<text x="6" y="{y+4:.2f}" class="axis">${label:,.0f}</text>'
        )
    markers: list[str] = []
    for index, item in enumerate(points):
        item_contract = item.get("next_candle_forecast")
        item_contract = item_contract if isinstance(item_contract, dict) else {}
        direction = str(item_contract.get("direction") or "").upper()
        if direction not in {"UP", "DOWN"}:
            continue
        result = str(item.get("direction_result") or "PENDING")
        marker_class = {
            "DIRECTION_CORRECT": "correct",
            "DIRECTION_WRONG": "wrong",
            "PENDING": "pending-dot",
        }.get(result, "pending-dot")
        x, y = xy(index, float(item["price"]))
        symbol = "↑" if direction == "UP" else "↓"
        markers.append(
            f'<g transform="translate({x:.2f},{y-12:.2f})"><circle r="11" '
            f'class="dot {marker_class}"/><text y="4" text-anchor="middle" '
            f'class="symbol">{symbol}</text></g>'
        )
    band = ""
    band_low = _finite(contract.get("likely_close_low"))
    band_high = _finite(contract.get("likely_close_high"))
    median = _finite(contract.get("median_close"))
    if band_low is not None and band_high is not None and median is not None:
        x0 = xy(len(points) - 1, prices[-1])[0]
        x1 = width - pad_x
        top = xy(len(points), band_high)[1]
        bottom = xy(len(points), band_low)[1]
        median_y = xy(len(points), median)[1]
        band = (
            f'<rect x="{x0:.2f}" y="{top:.2f}" width="{x1-x0:.2f}" '
            f'height="{max(2.0,bottom-top):.2f}" class="band"/>'
            f'<line x1="{x0:.2f}" y1="{median_y:.2f}" x2="{x1:.2f}" '
            f'y2="{median_y:.2f}" class="median"/>'
        )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="BTC close history and next forecast">'
        + "".join(grid)
        + "".join(labels)
        + f'<path d="{path}" class="line"/>'
        + band
        + "".join(markers)
        + "</svg>"
        + '<div class="legend"><span><i style="background:var(--ok)"></i>Correct direction</span><span><i style="background:var(--bad)"></i>Wrong direction</span><span><i style="background:var(--wait)"></i>Pending</span><span><i style="background:rgba(143,138,184,.35);border-radius:3px;width:18px"></i>Model interval</span></div>'
    )


def history_rows(history: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    result_classes = {
        "DIRECTION_CORRECT": "result-ok",
        "DIRECTION_WRONG": "result-bad",
        "PENDING": "result-wait",
        "LEGACY_NOT_SCORED": "result-muted",
        "NOT_SCORED": "result-muted",
    }
    for item in reversed(history[-36:]):
        contract = item.get("next_candle_forecast")
        contract = contract if isinstance(contract, dict) else {}
        probability_up = _finite(contract.get("probability_up"))
        direction = str(contract.get("direction") or "").upper()
        if direction not in {"UP", "DOWN"} and probability_up is not None:
            direction = "UP" if probability_up >= 0.5 else "DOWN"
        confidence = (
            probability_up
            if direction == "UP"
            else 1.0 - probability_up
            if probability_up is not None
            else None
        )
        range_text = (
            f"{price(contract.get('likely_close_low'))} – "
            f"{price(contract.get('likely_close_high'))}"
            if contract
            else "—"
        )
        direction_result = str(item.get("direction_result") or "PENDING")
        interval_result = str(item.get("interval_result") or "PENDING")
        rows.append(
            "<tr>"
            f"<td>{esc(compact_time(item.get('candle_time') or item.get('run_finished_at')))}</td>"
            f"<td>{esc(price(item.get('price')))}</td>"
            f'<td><span class="pill {"up" if direction == "UP" else "down"}">{esc(direction or "—")}</span></td>'
            f"<td>{esc(percent(confidence))}</td>"
            f"<td>{esc(price(contract.get('median_close')))}</td>"
            f"<td>{esc(range_text)}</td>"
            f"<td>{esc(price(item.get('actual_close')))}</td>"
            f'<td><span class="pill {result_classes.get(direction_result, "result-muted")}">{esc(direction_result.replace("DIRECTION_", ""))}</span></td>'
            f"<td>{esc(interval_result.replace('_', ' '))}</td>"
            "</tr>"
        )
    return "".join(rows) if rows else '<tr><td colspan="9" class="empty">No forecasts recorded yet.</td></tr>'


def blocker_chips(blockers: Any) -> str:
    if not isinstance(blockers, list) or not blockers:
        return '<span class="chip">No active blockers</span>'
    return "".join(
        f'<span class="chip">{esc(str(item).replace("_", " ").title())}</span>'
        for item in blockers
    )


def esc(value_: Any) -> str:
    return html.escape(str(value_), quote=True)


def value(value_: Any) -> str:
    return "—" if value_ in (None, "") else str(value_)


def price(value_: Any) -> str:
    numeric = _finite(value_)
    return "—" if numeric is None else f"${numeric:,.2f}"


def percent(value_: Any, digits: int = 1) -> str:
    numeric = _finite(value_)
    return "—" if numeric is None else f"{numeric * 100:.{digits}f}%"


def compact_time(value_: Any) -> str:
    try:
        return _utc(value_).strftime("%b %d · %H:%M UTC")
    except Exception:
        return value(value_)


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
            if (item.get("candle_time") or item.get("run_finished_at"))
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
        render(latest, history), encoding="utf-8"
    )
    correct, direction_total, direction_rate = direction_accuracy(history)
    inside, interval_total, coverage = interval_coverage(history)
    print(
        json.dumps(
            {
                "dashboard": "updated",
                "direction_resolved": direction_total,
                "direction_correct": correct,
                "direction_accuracy": direction_rate,
                "interval_resolved": interval_total,
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
        for row in candles.drop_duplicates("open_time", keep="last").itertuples(
            index=False
        )
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
        number = float(value_)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


if __name__ == "__main__":
    raise SystemExit(main())
