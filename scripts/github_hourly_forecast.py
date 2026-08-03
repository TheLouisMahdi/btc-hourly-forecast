from __future__ import annotations

import argparse
import html
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from btc_ema_trader.logging_setup import configure_logging
from btc_ema_trader.market import fetch_and_store
from btc_ema_trader.model import latest_bundle
from btc_ema_trader.runtime import RuntimeEngine
from btc_ema_trader.storage import Database

from github_common import (
    build_github_settings,
    copy_latest_model_from_state,
    json_safe,
    write_json,
)

LOGGER = logging.getLogger("github_hourly_forecast")
MAX_HISTORY = 24 * 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one stateless hourly forecast and build a static dashboard")
    parser.add_argument("--state-dir", default=".github_state")
    parser.add_argument("--model-state-dir", default=".model_state")
    parser.add_argument("--site-dir", default="site")
    parser.add_argument("--runtime-dir", default=".github_runtime/hourly")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    state_dir = (root / args.state_dir).resolve()
    model_state_dir = (root / args.model_state_dir).resolve()
    site_dir = (root / args.site_dir).resolve()
    runtime_dir = (root / args.runtime_dir).resolve()

    shutil.rmtree(runtime_dir, ignore_errors=True)
    shutil.rmtree(site_dir, ignore_errors=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)

    used_weekly_model = copy_latest_model_from_state(root, model_state_dir)
    settings = build_github_settings(root, runtime_dir)
    configure_logging(settings, verbose=True)
    database = Database(settings)
    database.initialize()

    started_at = pd.Timestamp.now(tz="UTC")
    try:
        bundle = latest_bundle(settings)
        market = fetch_and_store(settings, database, days=180, provider=bundle.provider)
        result = RuntimeEngine(settings, database).run_once(force=True)
        status = "OK" if result.get("status") != "FAIL_SAFE" else "FAIL_SAFE"
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Hourly forecast failed")
        market = None
        result = {"status": "FAIL_SAFE", "error": f"{type(exc).__name__}: {exc}"}
        status = "FAIL_SAFE"

    finished_at = pd.Timestamp.now(tz="UTC")
    record = json_safe(
        {
            **result,
            "run_status": status,
            "run_started_at": started_at,
            "run_finished_at": finished_at,
            "run_duration_seconds": (finished_at - started_at).total_seconds(),
            "market_refresh": market,
            "weekly_model_loaded": used_weekly_model,
        }
    )

    history_path = state_dir / "history.json"
    history = load_history(history_path)
    history = append_unique(history, record)[-MAX_HISTORY:]

    write_json(state_dir / "latest.json", record)
    write_json(history_path, history)
    write_json(site_dir / "latest.json", record)
    write_json(site_dir / "history.json", history)
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")
    (site_dir / "index.html").write_text(render_dashboard(record, history), encoding="utf-8")

    print(json.dumps(record, ensure_ascii=False, indent=2))
    if status != "OK":
        print("::warning::Forecast completed in FAIL_SAFE mode; the diagnostic dashboard was still published.")
    return 0


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def append_unique(history: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    key = record.get("candle_time") or record.get("run_finished_at")
    filtered = [item for item in history if (item.get("candle_time") or item.get("run_finished_at")) != key]
    filtered.append(record)
    return sorted(filtered, key=lambda item: str(item.get("candle_time") or item.get("run_finished_at") or ""))


def render_dashboard(latest: dict[str, Any], history: list[dict[str, Any]]) -> str:
    payload = json.dumps(json_safe({"latest": latest, "history": history}), ensure_ascii=False).replace("</", "<\\/")
    title = "BTC Hourly Forecast"
    return f"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="300">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#090d16; --panel:#111827; --line:#263246; --text:#eef2ff; --muted:#9ca3af; --good:#34d399; --bad:#fb7185; --wait:#fbbf24; }}
    * {{ box-sizing:border-box }}
    body {{ margin:0; background:radial-gradient(circle at top,#172033 0,#090d16 48%); color:var(--text); font-family:Tahoma,Arial,sans-serif; }}
    .wrap {{ width:min(1180px,94%); margin:28px auto 60px; }}
    header {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:18px; }}
    h1 {{ margin:0; font-size:clamp(24px,4vw,42px); }}
    .sub {{ color:var(--muted); margin-top:8px; line-height:1.8; }}
    .badge {{ padding:9px 13px; border:1px solid var(--line); border-radius:999px; white-space:nowrap; background:#0d1422; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .card {{ background:rgba(17,24,39,.9); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 16px 40px rgba(0,0,0,.18); }}
    .label {{ color:var(--muted); font-size:13px; margin-bottom:9px; }}
    .value {{ font-size:25px; font-weight:700; direction:ltr; text-align:right; overflow-wrap:anywhere; }}
    .good {{ color:var(--good) }} .bad {{ color:var(--bad) }} .wait {{ color:var(--wait) }}
    .wide {{ grid-column:span 2 }}
    .section {{ margin-top:14px; }}
    .section h2 {{ font-size:18px; margin:0 0 12px; }}
    .chart {{ height:260px; width:100%; display:block; background:#0b1220; border-radius:12px; border:1px solid var(--line); }}
    table {{ width:100%; border-collapse:collapse; direction:ltr; font-size:13px; }}
    th,td {{ padding:10px 8px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }}
    th {{ color:var(--muted); font-weight:500; }}
    .scroll {{ overflow:auto; max-height:430px; }}
    pre {{ direction:ltr; text-align:left; white-space:pre-wrap; overflow-wrap:anywhere; color:#d1d5db; margin:0; font-size:12px; }}
    footer {{ color:var(--muted); text-align:center; margin-top:20px; line-height:1.8; }}
    @media(max-width:850px) {{ .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .wide {{ grid-column:span 2; }} header {{ flex-direction:column; }} }}
    @media(max-width:520px) {{ .grid {{ grid-template-columns:1fr; }} .wide {{ grid-column:span 1; }} }}
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <div><h1>پیش‌بینی ساعتی بیت‌کوین</h1><div class="sub">اجرای خودکار با GitHub Actions؛ این صفحه هر پنج دقیقه تازه‌سازی می‌شود.</div></div>
    <div id="health" class="badge">در حال بارگذاری…</div>
  </header>
  <main class="grid">
    <div class="card"><div class="label">تصمیم</div><div id="action" class="value">—</div></div>
    <div class="card"><div class="label">جهت پیش‌بینی</div><div id="direction" class="value">—</div></div>
    <div class="card"><div class="label">قیمت آخرین کندل</div><div id="price" class="value">—</div></div>
    <div class="card"><div class="label">افق منتخب</div><div id="horizon" class="value">—</div></div>
    <div class="card"><div class="label">اعتماد</div><div id="confidence" class="value">—</div></div>
    <div class="card"><div class="label">احتمال معامله‌پذیری</div><div id="tradeability" class="value">—</div></div>
    <div class="card"><div class="label">لبه خالص مورد انتظار</div><div id="edge" class="value">—</div></div>
    <div class="card"><div class="label">رژیم بازار</div><div id="regime" class="value">—</div></div>
    <section class="card wide section"><h2>روند قیمت در خروجی‌های ثبت‌شده</h2><canvas id="chart" class="chart"></canvas></section>
    <section class="card wide section"><h2>وضعیت مدل و اجرا</h2><pre id="details"></pre></section>
    <section class="card wide section"><h2>موانع تصمیم</h2><pre id="blockers"></pre></section>
    <section class="card wide section"><h2>۳۰ خروجی اخیر</h2><div class="scroll"><table><thead><tr><th>Candle UTC</th><th>Price</th><th>Direction</th><th>Action</th><th>Confidence</th><th>Edge bps</th></tr></thead><tbody id="rows"></tbody></table></div></section>
  </main>
  <footer>این سامانه فقط برای پژوهش و Paper Trading است و توصیه مالی نیست.<br><span id="updated"></span></footer>
</div>
<script id="data" type="application/json">{payload}</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent); const x=data.latest||{{}}, hist=data.history||[];
const el=id=>document.getElementById(id); const num=(v,d=2)=>Number.isFinite(Number(v))?Number(v).toLocaleString('en-US',{{maximumFractionDigits:d,minimumFractionDigits:d}}):'—';
const pct=v=>Number.isFinite(Number(v))?(Number(v)*100).toFixed(2)+'%':'—';
const action=x.action||x.status||x.run_status||'—'; el('action').textContent=action; el('direction').textContent=x.forecast_direction||'—'; el('price').textContent=x.price?'$'+num(x.price,2):'—'; el('horizon').textContent=x.selected_horizon?x.selected_horizon+'h':'—'; el('confidence').textContent=pct(x.confidence); el('tradeability').textContent=pct(x.tradeability_probability); el('edge').textContent=Number.isFinite(Number(x.expected_net_edge_bps))?num(x.expected_net_edge_bps,1)+' bps':'—'; el('regime').textContent=x.regime||'—';
const ok=x.run_status==='OK' && x.status!=='FAIL_SAFE'; el('health').textContent=ok?'اجرای موفق':'FAIL-SAFE'; el('health').classList.add(ok?'good':'bad'); el('action').classList.add(action==='WAIT'?'wait':action==='FAIL_SAFE'?'bad':'good');
el('blockers').textContent=(x.blockers&&x.blockers.length)?x.blockers.join('\n'):'مانعی ثبت نشده است.';
el('details').textContent=JSON.stringify({{candle_time:x.candle_time,created_at:x.created_at,provider:x.provider,event_type:x.event_type,model_id:x.model_id,qualification_passed:x.qualification_passed,data_health:x.data_health,market_refresh:x.market_refresh,error:x.error}},null,2);
el('updated').textContent='آخرین اجرای سرور: '+(x.run_finished_at?new Date(x.run_finished_at).toLocaleString():'—');
const rows=hist.slice(-30).reverse().map(r=>`<tr><td>${{r.candle_time||r.run_finished_at||'—'}}</td><td>${{r.price?num(r.price,2):'—'}}</td><td>${{r.forecast_direction||'—'}}</td><td>${{r.action||r.status||'—'}}</td><td>${{pct(r.confidence)}}</td><td>${{Number.isFinite(Number(r.expected_net_edge_bps))?num(r.expected_net_edge_bps,1):'—'}}</td></tr>`).join(''); el('rows').innerHTML=rows;
function draw(){{const c=el('chart'),ctx=c.getContext('2d'),dpr=window.devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const pts=hist.filter(r=>Number.isFinite(Number(r.price))).slice(-168);if(pts.length<2){{ctx.fillStyle='#9ca3af';ctx.fillText('داده کافی نیست',20,30);return}}const vals=pts.map(r=>Number(r.price)),mn=Math.min(...vals),mx=Math.max(...vals),pad=Math.max((mx-mn)*.12,1),lo=mn-pad,hi=mx+pad;ctx.strokeStyle='#263246';ctx.lineWidth=1;for(let i=1;i<5;i++){{const y=i*h/5;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}}ctx.strokeStyle='#60a5fa';ctx.lineWidth=2;ctx.beginPath();pts.forEach((p,i)=>{{const xx=i*(w-20)/(pts.length-1)+10,yy=h-((Number(p.price)-lo)/(hi-lo))*(h-20)-10;i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)}});ctx.stroke();}} draw(); addEventListener('resize',draw);
</script>
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
