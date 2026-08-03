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
    parser = argparse.ArgumentParser(
        description="Run one stateless hourly forecast and build a static dashboard"
    )
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
    (site_dir / "index.html").write_text(
        render_dashboard(record, history),
        encoding="utf-8",
    )

    print(json.dumps(record, ensure_ascii=False, indent=2))
    if status != "OK":
        print(
            "::warning::Forecast completed in FAIL_SAFE mode; "
            "the diagnostic dashboard was still published."
        )
    return 0


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def append_unique(
    history: list[dict[str, Any]],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    key = record.get("candle_time") or record.get("run_finished_at")
    filtered = [
        item
        for item in history
        if (item.get("candle_time") or item.get("run_finished_at")) != key
    ]
    filtered.append(record)
    return sorted(
        filtered,
        key=lambda item: str(
            item.get("candle_time") or item.get("run_finished_at") or ""
        ),
    )


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(number):
        return "—"
    return f"{number:,.{digits}f}"


def _percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(number):
        return "—"
    return f"{number * 100:.2f}%"


def _value_or_dash(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value)


def _status_class(action: str) -> str:
    if action == "WAIT":
        return "wait"
    if action in {"FAIL_SAFE", "BLOCKED"}:
        return "bad"
    return "good"


def _render_chart_svg(history: list[dict[str, Any]]) -> str:
    points: list[tuple[str, float]] = []
    for item in history[-168:]:
        try:
            price = float(item.get("price"))
        except (TypeError, ValueError):
            continue
        if pd.isna(price):
            continue
        stamp = str(item.get("candle_time") or item.get("run_finished_at") or "")
        points.append((stamp, price))

    width = 1000
    height = 260
    left = 58
    right = 18
    top = 20
    bottom = 34
    plot_w = width - left - right
    plot_h = height - top - bottom

    if len(points) < 2:
        return (
            f'<svg class="chart" viewBox="0 0 {width} {height}" '
            'role="img" aria-label="داده کافی برای نمودار وجود ندارد">'
            '<rect width="100%" height="100%" rx="12" fill="#0b1220"/>'
            '<text x="50%" y="50%" text-anchor="middle" '
            'fill="#9ca3af" font-size="18">داده کافی نیست</text>'
            "</svg>"
        )

    values = [price for _, price in points]
    minimum = min(values)
    maximum = max(values)
    padding = max((maximum - minimum) * 0.12, 1.0)
    low = minimum - padding
    high = maximum + padding
    span = max(high - low, 1.0)

    coords: list[tuple[float, float]] = []
    for index, (_, price) in enumerate(points):
        x = left + (index / (len(points) - 1)) * plot_w
        y = top + (1 - ((price - low) / span)) * plot_h
        coords.append((x, y))

    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    grid_lines = []
    labels = []
    for index in range(5):
        ratio = index / 4
        y = top + ratio * plot_h
        value = high - ratio * span
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" '
            'stroke="#263246" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" '
            f'fill="#9ca3af" font-size="12">{_escape(_number(value, 0))}</text>'
        )

    first_stamp = _escape(points[0][0][:16].replace("T", " "))
    last_stamp = _escape(points[-1][0][:16].replace("T", " "))

    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="روند قیمت خروجی‌های ثبت‌شده">'
        '<rect width="100%" height="100%" rx="12" fill="#0b1220"/>'
        + "".join(grid_lines)
        + "".join(labels)
        + f'<polyline points="{polyline}" fill="none" stroke="#60a5fa" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        + f'<circle cx="{coords[-1][0]:.2f}" cy="{coords[-1][1]:.2f}" '
        'r="5" fill="#22d3ee"/>'
        + f'<text x="{left}" y="{height-10}" fill="#9ca3af" '
        f'font-size="12">{first_stamp}</text>'
        + f'<text x="{width-right}" y="{height-10}" text-anchor="end" '
        f'fill="#9ca3af" font-size="12">{last_stamp}</text>'
        + "</svg>"
    )


def _render_history_rows(history: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for item in reversed(history[-30:]):
        candle = _escape(
            _value_or_dash(item.get("candle_time") or item.get("run_finished_at"))
        )
        price = _escape(_number(item.get("price"), 2))
        direction = _escape(_value_or_dash(item.get("forecast_direction")))
        action = _escape(
            _value_or_dash(
                item.get("action") or item.get("status") or item.get("run_status")
            )
        )
        confidence = _escape(_percent(item.get("confidence")))
        edge = _number(item.get("expected_net_edge_bps"), 1)
        edge_text = "—" if edge == "—" else f"{edge} bps"
        rows.append(
            "<tr>"
            f"<td>{candle}</td>"
            f"<td>{price}</td>"
            f"<td>{direction}</td>"
            f"<td>{action}</td>"
            f"<td>{confidence}</td>"
            f"<td>{_escape(edge_text)}</td>"
            "</tr>"
        )

    if not rows:
        return '<tr><td colspan="6" class="empty">هنوز خروجی ثبت نشده است.</td></tr>'
    return "".join(rows)


def render_dashboard(latest: dict[str, Any], history: list[dict[str, Any]]) -> str:
    action = _value_or_dash(
        latest.get("action") or latest.get("status") or latest.get("run_status")
    )
    direction = _value_or_dash(latest.get("forecast_direction"))
    price = _number(latest.get("price"), 2)
    price = "—" if price == "—" else f"${price}"
    horizon = latest.get("selected_horizon")
    horizon_text = "—" if horizon in (None, "") else f"{horizon}h"
    confidence = _percent(latest.get("confidence"))
    tradeability = _percent(latest.get("tradeability_probability"))
    edge = _number(latest.get("expected_net_edge_bps"), 1)
    edge_text = "—" if edge == "—" else f"{edge} bps"
    regime = _value_or_dash(latest.get("regime"))

    run_ok = latest.get("run_status") == "OK" and latest.get("status") != "FAIL_SAFE"
    health_text = "اجرای موفق" if run_ok else "FAIL-SAFE"
    health_class = "good" if run_ok else "bad"
    action_class = _status_class(str(action))

    blockers = latest.get("blockers")
    if isinstance(blockers, list) and blockers:
        blockers_text = "\n".join(str(item) for item in blockers)
    else:
        blockers_text = "مانعی ثبت نشده است."

    details = {
        "candle_time": latest.get("candle_time"),
        "created_at": latest.get("created_at"),
        "provider": latest.get("provider"),
        "event_type": latest.get("event_type"),
        "model_id": latest.get("model_id"),
        "qualification_passed": latest.get("qualification_passed"),
        "weekly_model_loaded": latest.get("weekly_model_loaded"),
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

    updated = _value_or_dash(latest.get("run_finished_at"))
    chart_svg = _render_chart_svg(history)
    history_rows = _render_history_rows(history)

    return f"""<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta http-equiv="refresh" content="300">
  <meta name="theme-color" content="#090d16">
  <title>BTC Hourly Forecast</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg:#090d16;
      --panel:#111827;
      --line:#263246;
      --text:#eef2ff;
      --muted:#9ca3af;
      --good:#34d399;
      --bad:#fb7185;
      --wait:#fbbf24;
    }}
    * {{ box-sizing:border-box; }}
    html {{ -webkit-text-size-adjust:100%; }}
    body {{
      margin:0;
      min-height:100vh;
      background:radial-gradient(circle at top,#172033 0,#090d16 48%);
      color:var(--text);
      font-family:Tahoma,Arial,sans-serif;
    }}
    .wrap {{
      width:min(1180px,94%);
      margin:28px auto 60px;
      padding-bottom:env(safe-area-inset-bottom);
    }}
    header {{
      display:flex;
      justify-content:space-between;
      gap:16px;
      align-items:flex-start;
      margin-bottom:18px;
    }}
    h1 {{ margin:0; font-size:clamp(24px,7vw,42px); }}
    .sub {{ color:var(--muted); margin-top:8px; line-height:1.8; }}
    .badge {{
      padding:9px 13px;
      border:1px solid var(--line);
      border-radius:999px;
      white-space:nowrap;
      background:#0d1422;
    }}
    .grid {{
      display:grid;
      grid-template-columns:repeat(4,minmax(0,1fr));
      gap:12px;
    }}
    .card {{
      min-width:0;
      background:rgba(17,24,39,.94);
      border:1px solid var(--line);
      border-radius:16px;
      padding:16px;
      box-shadow:0 16px 40px rgba(0,0,0,.18);
    }}
    .label {{ color:var(--muted); font-size:13px; margin-bottom:9px; }}
    .value {{
      font-size:clamp(20px,5vw,25px);
      font-weight:700;
      direction:ltr;
      text-align:right;
      overflow-wrap:anywhere;
    }}
    .good {{ color:var(--good); }}
    .bad {{ color:var(--bad); }}
    .wait {{ color:var(--wait); }}
    .wide {{ grid-column:span 2; }}
    .section {{ margin-top:14px; }}
    .section h2 {{ font-size:18px; margin:0 0 12px; }}
    .chart {{
      width:100%;
      height:auto;
      min-height:220px;
      display:block;
      border:1px solid var(--line);
      border-radius:12px;
    }}
    table {{
      width:100%;
      border-collapse:collapse;
      direction:ltr;
      font-size:13px;
    }}
    th,td {{
      padding:10px 8px;
      border-bottom:1px solid var(--line);
      text-align:left;
      white-space:nowrap;
    }}
    th {{ color:var(--muted); font-weight:500; }}
    .scroll {{ overflow:auto; max-height:430px; -webkit-overflow-scrolling:touch; }}
    .empty {{ text-align:center; color:var(--muted); }}
    pre {{
      direction:ltr;
      text-align:left;
      white-space:pre-wrap;
      overflow-wrap:anywhere;
      color:#d1d5db;
      margin:0;
      font-size:12px;
    }}
    footer {{
      color:var(--muted);
      text-align:center;
      margin-top:20px;
      line-height:1.8;
    }}
    @media(max-width:850px) {{
      .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .wide {{ grid-column:span 2; }}
      header {{ flex-direction:column; }}
    }}
    @media(max-width:520px) {{
      .wrap {{ width:92%; margin-top:18px; }}
      .grid {{ grid-template-columns:1fr; }}
      .wide {{ grid-column:span 1; }}
      .card {{ padding:14px; border-radius:14px; }}
      .chart {{ min-height:180px; }}
    }}
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>پیش‌بینی ساعتی بیت‌کوین</h1>
      <div class="sub">
        اجرای خودکار با GitHub Actions؛ این صفحه هر پنج دقیقه تازه‌سازی می‌شود.
      </div>
    </div>
    <div class="badge {health_class}">{_escape(health_text)}</div>
  </header>

  <main class="grid">
    <div class="card">
      <div class="label">تصمیم</div>
      <div class="value {action_class}">{_escape(action)}</div>
    </div>
    <div class="card">
      <div class="label">جهت پیش‌بینی</div>
      <div class="value">{_escape(direction)}</div>
    </div>
    <div class="card">
      <div class="label">قیمت آخرین کندل</div>
      <div class="value">{_escape(price)}</div>
    </div>
    <div class="card">
      <div class="label">افق منتخب</div>
      <div class="value">{_escape(horizon_text)}</div>
    </div>
    <div class="card">
      <div class="label">اعتماد</div>
      <div class="value">{_escape(confidence)}</div>
    </div>
    <div class="card">
      <div class="label">احتمال معامله‌پذیری</div>
      <div class="value">{_escape(tradeability)}</div>
    </div>
    <div class="card">
      <div class="label">لبه خالص مورد انتظار</div>
      <div class="value">{_escape(edge_text)}</div>
    </div>
    <div class="card">
      <div class="label">رژیم بازار</div>
      <div class="value">{_escape(regime)}</div>
    </div>

    <section class="card wide section">
      <h2>روند قیمت در خروجی‌های ثبت‌شده</h2>
      {chart_svg}
    </section>

    <section class="card wide section">
      <h2>وضعیت مدل و اجرا</h2>
      <pre>{_escape(details_text)}</pre>
    </section>

    <section class="card wide section">
      <h2>موانع تصمیم</h2>
      <pre>{_escape(blockers_text)}</pre>
    </section>

    <section class="card wide section">
      <h2>۳۰ خروجی اخیر</h2>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Candle UTC</th>
              <th>Price</th>
              <th>Direction</th>
              <th>Action</th>
              <th>Confidence</th>
              <th>Edge bps</th>
            </tr>
          </thead>
          <tbody>{history_rows}</tbody>
        </table>
      </div>
    </section>
  </main>

  <footer>
    این سامانه فقط برای پژوهش و Paper Trading است و توصیه مالی نیست.
    <br>
    آخرین اجرای سرور: {_escape(updated)}
  </footer>
</div>
</body>
</html>"""


if __name__ == "__main__":
    raise SystemExit(main())
