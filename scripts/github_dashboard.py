from __future__ import annotations

import argparse
import html
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from github_common import json_safe, write_json

MAX_HISTORY = 24 * 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve forecast outcomes and render the English static dashboard")
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
    sql = "SELECT open_time, close FROM candles WHERE " + " AND ".join(clauses) + " ORDER BY open_time"
    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    if not frame.empty:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    return frame


def pending(item: dict[str, Any], result: str = "PENDING") -> dict[str, Any]:
    output = dict(item)
    output["prediction_result"] = result
    output.setdefault("actual_direction", None)
    output.setdefault("actual_price", None)
    output.setdefault("actual_return", None)
    output.setdefault("target_candle_time", None)
    return output


def resolve_outcomes(history: list[dict[str, Any]], candles: pd.DataFrame) -> list[dict[str, Any]]:
    if candles.empty:
        return [pending(item) for item in history]
    closes = {
        pd.Timestamp(row.open_time): float(row.close)
        for row in candles.drop_duplicates("open_time", keep="last").itertuples(index=False)
        if not pd.isna(row.close)
    }
    resolved: list[dict[str, Any]] = []
    for source in history:
        item = dict(source)
        direction = str(item.get("forecast_direction") or "").upper()
        if item.get("run_status") != "OK" or direction not in {"UP", "DOWN"}:
            resolved.append(pending(item, "NOT_SCORED"))
            continue
        try:
            candle_time = pd.Timestamp(item["candle_time"])
            candle_time = candle_time.tz_localize("UTC") if candle_time.tzinfo is None else candle_time.tz_convert("UTC")
            horizon = max(1, int(item.get("selected_horizon") or 1))
            source_price = float(item["price"])
        except (KeyError, TypeError, ValueError):
            resolved.append(pending(item, "NOT_SCORED"))
            continue
        target_time = candle_time + pd.Timedelta(hours=horizon)
        item["target_candle_time"] = target_time.isoformat()
        target_price = closes.get(target_time)
        if target_price is None:
            resolved.append(pending(item))
            continue
        actual_return = target_price / source_price - 1.0
        actual_direction = "UP" if actual_return > 0 else "DOWN" if actual_return < 0 else "FLAT"
        item.update({
            "prediction_result": "CORRECT" if actual_direction == direction else "WRONG",
            "actual_direction": actual_direction,
            "actual_price": target_price,
            "actual_return": actual_return,
        })
        resolved.append(item)
    return resolved


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any, digits: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(value):
        return "—"
    return f"{value:,.{digits}f}"


def percent(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%"


def value(value: Any) -> str:
    return "—" if value is None or value == "" else str(value)


def accuracy(history: list[dict[str, Any]]) -> tuple[int, int, float | None]:
    scored = [item for item in history if item.get("prediction_result") in {"CORRECT", "WRONG"}]
    correct = sum(item.get("prediction_result") == "CORRECT" for item in scored)
    return correct, len(scored), correct / len(scored) if scored else None


def chart(history: list[dict[str, Any]]) -> str:
    points: list[dict[str, Any]] = []
    for item in history[-168:]:
        try:
            price = float(item.get("price"))
        except (TypeError, ValueError):
            continue
        if pd.isna(price):
            continue
        points.append({"stamp": str(item.get("candle_time") or item.get("run_finished_at") or ""), "price": price, "direction": str(item.get("forecast_direction") or "").upper(), "result": str(item.get("prediction_result") or "PENDING").upper(), "horizon": item.get("selected_horizon")})
    width, height, left, right, top, bottom = 1000, 310, 64, 20, 62, 38
    if len(points) < 2:
        return f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Not enough forecast history"><rect width="100%" height="100%" rx="12" fill="#0b1220"/><text x="50%" y="50%" text-anchor="middle" fill="#9ca3af" font-size="18">Not enough history yet</text></svg>'
    plot_w, plot_h = width-left-right, height-top-bottom
    prices = [item["price"] for item in points]
    pad = max((max(prices)-min(prices))*.14, 1.0)
    low, high = min(prices)-pad, max(prices)+pad
    span = max(high-low, 1.0)
    coords = [(left+(i/(len(points)-1))*plot_w, top+(1-((item["price"]-low)/span))*plot_h) for i,item in enumerate(points)]
    grid, labels = [], []
    for i in range(5):
        ratio=i/4; y=top+ratio*plot_h; price=high-ratio*span
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#263246"/>')
        labels.append(f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end" fill="#9ca3af" font-size="12">{esc(number(price,0))}</text>')
    colors={"CORRECT":"#34d399","WRONG":"#fb7185","PENDING":"#fbbf24","NOT_SCORED":"#94a3b8"}
    markers=[]
    for item,(x,y) in zip(points,coords):
        direction=item["direction"]
        if direction not in {"UP","DOWN"}: continue
        color=colors.get(item["result"],"#94a3b8"); size=6.5
        triangle=(f"{x:.2f},{y-11:.2f} {x-size:.2f},{y+1.5:.2f} {x+size:.2f},{y+1.5:.2f}" if direction=="UP" else f"{x:.2f},{y+11:.2f} {x-size:.2f},{y-1.5:.2f} {x+size:.2f},{y-1.5:.2f}")
        tip=esc(f'{item["stamp"]} · {direction} · {item["result"]} · {item.get("horizon") or 1}h')
        markers.append(f'<g><title>{tip}</title><polygon points="{triangle}" fill="{color}" stroke="#020617" stroke-width="1.5"/></g>')
    correct,scored,rate=accuracy(history)
    accuracy_text="No resolved forecasts" if rate is None else f"Direction accuracy: {rate*100:.1f}% ({correct}/{scored})"
    line=" ".join(f"{x:.2f},{y:.2f}" for x,y in coords)
    first=esc(points[0]["stamp"][:16].replace("T"," ")); last=esc(points[-1]["stamp"][:16].replace("T"," "))
    legend='<line x1="68" y1="25" x2="102" y2="25" stroke="#60a5fa" stroke-width="3"/><text x="110" y="29" fill="#cbd5e1" font-size="12">BTC close</text><polygon points="222,17 215,30 229,30" fill="#34d399"/><text x="237" y="29" fill="#cbd5e1" font-size="12">Correct</text><polygon points="326,17 319,30 333,30" fill="#fb7185"/><text x="341" y="29" fill="#cbd5e1" font-size="12">Wrong</text><polygon points="418,17 411,30 425,30" fill="#fbbf24"/><text x="433" y="29" fill="#cbd5e1" font-size="12">Pending</text><text x="68" y="48" fill="#94a3b8" font-size="11">Triangle direction: ▲ UP · ▼ DOWN</text>'
    return f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="BTC prices with forecast outcomes"><rect width="100%" height="100%" rx="12" fill="#0b1220"/>{legend}<text x="{width-right}" y="29" text-anchor="end" fill="#e2e8f0" font-size="12">{esc(accuracy_text)}</text>{"".join(grid)}{"".join(labels)}<polyline points="{line}" fill="none" stroke="#60a5fa" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>{"".join(markers)}<text x="{left}" y="{height-12}" fill="#9ca3af" font-size="12">{first}</text><text x="{width-right}" y="{height-12}" text-anchor="end" fill="#9ca3af" font-size="12">{last}</text></svg>'


def rows(history: list[dict[str, Any]]) -> str:
    output=[]
    classes={"CORRECT":"correct","WRONG":"wrong","PENDING":"pending","NOT_SCORED":"muted-result"}
    for item in reversed(history[-30:]):
        result=str(item.get("prediction_result") or "PENDING").upper()
        actual_price=number(item.get("actual_price"),2); actual_direction=item.get("actual_direction")
        actual=(f"{actual_direction} · ${actual_price} · {percent(item.get('actual_return'))}" if actual_direction and actual_price!="—" else "Waiting for target candle")
        horizon="—" if item.get("selected_horizon") in (None,"") else f"{item.get('selected_horizon')}h"
        edge=number(item.get("expected_net_edge_bps"),1); edge="—" if edge=="—" else f"{edge} bps"
        action=value(item.get("action") or item.get("status") or item.get("run_status"))
        output.append(f'<tr><td>{esc(value(item.get("candle_time") or item.get("run_finished_at")))}</td><td>${esc(number(item.get("price"),2))}</td><td><strong>{esc(value(item.get("forecast_direction")))}</strong></td><td>{esc(horizon)}</td><td>{esc(actual)}</td><td><span class="result {classes.get(result,"muted-result")}">{esc(result)}</span></td><td>{esc(action)}</td><td>{esc(percent(item.get("confidence")))}</td><td>{esc(edge)}</td></tr>')
    return "".join(output) if output else '<tr><td colspan="9" class="empty">No forecasts recorded yet.</td></tr>'


def render(latest: dict[str, Any], history: list[dict[str, Any]]) -> str:
    action=value(latest.get("action") or latest.get("status") or latest.get("run_status")); action_class="wait" if action=="WAIT" else "bad" if action in {"FAIL_SAFE","BLOCKED"} else "good"
    run_ok=latest.get("run_status")=="OK" and latest.get("status")!="FAIL_SAFE"
    correct,scored,rate=accuracy(history); accuracy_text="—" if rate is None else f"{rate*100:.1f}%"
    blockers=latest.get("blockers"); blockers_text="\n".join(map(str,blockers)) if isinstance(blockers,list) and blockers else "No decision blockers were recorded."
    details={"candle_time":latest.get("candle_time"),"created_at":latest.get("created_at"),"provider":latest.get("provider"),"event_type":latest.get("event_type"),"model_id":latest.get("model_id"),"qualification_passed":latest.get("qualification_passed"),"weekly_model_loaded":latest.get("weekly_model_loaded"),"latest_prediction_result":latest.get("prediction_result"),"target_candle_time":latest.get("target_candle_time"),"data_health":latest.get("data_health"),"market_refresh":latest.get("market_refresh"),"error":latest.get("error")}
    details_text=json.dumps(json_safe(details),ensure_ascii=False,indent=2,allow_nan=False)
    price=number(latest.get("price"),2); price="—" if price=="—" else f"${price}"
    horizon="—" if latest.get("selected_horizon") in (None,"") else f"{latest.get('selected_horizon')}h"
    edge=number(latest.get("expected_net_edge_bps"),1); edge="—" if edge=="—" else f"{edge} bps"
    return f'''<!doctype html><html lang="en" dir="ltr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta http-equiv="refresh" content="300"><meta name="theme-color" content="#090d16"><title>BTC Hourly Forecast</title><style>
:root{{color-scheme:dark;--line:#263246;--text:#eef2ff;--muted:#9ca3af;--good:#34d399;--bad:#fb7185;--wait:#fbbf24}}*{{box-sizing:border-box}}html{{-webkit-text-size-adjust:100%}}body{{margin:0;min-height:100vh;background:radial-gradient(circle at top,#172033 0,#090d16 48%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}.wrap{{width:min(1220px,94%);margin:28px auto 60px;padding-bottom:env(safe-area-inset-bottom)}}header{{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:18px}}h1{{margin:0;font-size:clamp(28px,7vw,44px);letter-spacing:-.03em}}.sub{{color:var(--muted);margin-top:8px;line-height:1.65;max-width:760px}}.badge{{padding:9px 13px;border:1px solid var(--line);border-radius:999px;white-space:nowrap;background:#0d1422;font-weight:700}}.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}}.card{{min-width:0;background:rgba(17,24,39,.94);border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:0 16px 40px rgba(0,0,0,.18)}}.label{{color:var(--muted);font-size:13px;margin-bottom:9px}}.value{{font-size:clamp(19px,4vw,25px);font-weight:750;overflow-wrap:anywhere}}.meta{{color:var(--muted);font-size:12px;margin-top:6px}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.wait{{color:var(--wait)}}.wide{{grid-column:span 5}}.half{{grid-column:span 2}}.section{{margin-top:14px}}.section h2{{font-size:18px;margin:0 0 12px}}.chart{{width:100%;height:auto;min-height:240px;display:block;border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:11px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{color:var(--muted);font-weight:600;position:sticky;top:0;background:#111827}}.scroll{{overflow:auto;max-height:480px;-webkit-overflow-scrolling:touch}}.empty{{text-align:center;color:var(--muted)}}.result{{display:inline-flex;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:800;letter-spacing:.04em;border:1px solid currentColor}}.correct{{color:var(--good);background:rgba(52,211,153,.08)}}.wrong{{color:var(--bad);background:rgba(251,113,133,.08)}}.pending{{color:var(--wait);background:rgba(251,191,36,.08)}}.muted-result{{color:#94a3b8;background:rgba(148,163,184,.08)}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;color:#d1d5db;margin:0;font-size:12px}}footer{{color:var(--muted);text-align:center;margin-top:20px;line-height:1.8}}@media(max-width:980px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.wide,.half{{grid-column:span 2}}header{{flex-direction:column}}}}@media(max-width:520px){{.wrap{{width:92%;margin-top:18px}}.grid{{grid-template-columns:1fr}}.wide,.half{{grid-column:span 1}}.card{{padding:14px;border-radius:14px}}.chart{{min-height:190px}}h1{{font-size:30px}}}}
</style></head><body><div class="wrap"><header><div><h1>BTC Hourly Forecast</h1><div class="sub">Automated BTCUSDT one-hour forecasting, event detection and fail-safe paper-trade qualification. The page refreshes every five minutes.</div></div><div class="badge {'good' if run_ok else 'bad'}">{'Run successful' if run_ok else 'FAIL-SAFE'}</div></header><main class="grid">
<div class="card"><div class="label">Decision</div><div class="value {action_class}">{esc(action)}</div></div><div class="card"><div class="label">Forecast direction</div><div class="value">{esc(value(latest.get('forecast_direction')))}</div></div><div class="card"><div class="label">Latest candle close</div><div class="value">{esc(price)}</div></div><div class="card"><div class="label">Selected horizon</div><div class="value">{esc(horizon)}</div></div><div class="card"><div class="label">Confidence</div><div class="value">{esc(percent(latest.get('confidence')))}</div></div><div class="card"><div class="label">Tradeability probability</div><div class="value">{esc(percent(latest.get('tradeability_probability')))}</div></div><div class="card"><div class="label">Expected net edge</div><div class="value">{esc(edge)}</div></div><div class="card"><div class="label">Market regime</div><div class="value">{esc(value(latest.get('regime')))}</div></div><div class="card"><div class="label">Resolved direction accuracy</div><div class="value">{esc(accuracy_text)}</div><div class="meta">{scored} resolved</div></div><div class="card"><div class="label">Correct forecasts</div><div class="value good">{correct}</div><div class="meta">Out of {scored} resolved</div></div>
<section class="card wide section"><h2>Price and forecast outcomes</h2>{chart(history)}</section><section class="card half section"><h2>Model and run status</h2><pre>{esc(details_text)}</pre></section><section class="card half section"><h2>Decision blockers</h2><pre>{esc(blockers_text)}</pre></section><section class="card wide section"><h2>Latest 30 forecasts</h2><div class="scroll"><table><thead><tr><th>Candle UTC</th><th>Source price</th><th>Forecast</th><th>Horizon</th><th>Actual outcome</th><th>Result</th><th>Action</th><th>Confidence</th><th>Edge</th></tr></thead><tbody>{rows(history)}</tbody></table></div></section></main><footer>Research and paper-trading only. This dashboard is not financial advice.<br>Last server run: {esc(value(latest.get('run_finished_at')))}</footer></div></body></html>'''


def main() -> int:
    args=parse_args(); root=Path(__file__).resolve().parents[1]
    state_dir=(root/args.state_dir).resolve(); runtime_dir=(root/args.runtime_dir).resolve(); site_dir=(root/args.site_dir).resolve()
    history=load_json(state_dir/"history.json",[]); latest=load_json(state_dir/"latest.json",{})
    provider=str(latest.get("provider") or "") or None
    history=resolve_outcomes(history,load_candles(runtime_dir/"btc_hourly.sqlite3",provider))[-MAX_HISTORY:]
    latest_key=latest.get("candle_time") or latest.get("run_finished_at")
    latest=next((item for item in reversed(history) if (item.get("candle_time") or item.get("run_finished_at"))==latest_key),latest)
    state_dir.mkdir(parents=True,exist_ok=True); site_dir.mkdir(parents=True,exist_ok=True)
    write_json(state_dir/"latest.json",latest); write_json(state_dir/"history.json",history); write_json(site_dir/"latest.json",latest); write_json(site_dir/"history.json",history)
    (site_dir/".nojekyll").write_text("",encoding="utf-8"); (site_dir/"index.html").write_text(render(latest,history),encoding="utf-8")
    correct,scored,rate=accuracy(history); print(json.dumps({"dashboard":"updated","resolved":scored,"correct":correct,"accuracy":rate},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
