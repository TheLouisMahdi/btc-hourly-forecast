"""Render a minimal, independent rule-based paper trading dashboard."""
from __future__ import annotations

import html
import json
from pathlib import Path


def read(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def main():
    source, target = Path(".rule_state"), Path("site")
    target.mkdir(exist_ok=True)
    latest = read(source / "latest.json", {})
    state = read(source / "state.json", {})
    trades = read(source / "trades.json", [])
    candles = read(source / "candles.json", [])
    history = read(source / "history.json", [])
    regime_source = Path(".regime_shadow")
    regime_latest = read(regime_source / "latest.json", {})
    regime_state = read(regime_source / "state.json", {})
    regime_trades = read(regime_source / "trades.json", [])
    shadow_source = Path(".sma_shadow")
    shadow_latest = read(shadow_source / "latest.json", {})
    shadow_state = read(shadow_source / "state.json", {})
    shadow_trades = read(shadow_source / "trades.json", [])
    shadow_history = read(shadow_source / "history.json", [])
    audit_source = Path(".sma_audit")
    audit_report = read(audit_source / "report.json", {})
    audit_executions = read(audit_source / "executions.json", [])
    audit_snapshots = read(audit_source / "snapshots.json", [])

    def safe(value):
        return html.escape(str(value if value is not None else "n/a"))

    def money(value):
        return "n/a" if value is None else f"{float(value):,.2f}"

    prices = [float(c["close"]) for c in candles]
    graph = ""
    if len(prices) > 1:
        lo, hi = min(prices), max(prices)
        span = max(hi - lo, 1)
        coords = " ".join(
            f"{i*960/(len(prices)-1):.1f},{210-(v-lo)/span*190:.1f}"
            for i, v in enumerate(prices)
        )
        graph = f'<svg viewBox="0 0 960 220"><polyline points="{coords}" stroke="#49c5b6" stroke-width="3" fill="none"/></svg>'
    rows = "".join(
        f'<tr><td>{safe(t.get("reason"))}</td><td>{money(t.get("pnl_usd"))}</td>'
        f'<td>{money(t.get("entry_price"))}</td><td>{money(t.get("exit_price"))}</td></tr>'
        for t in reversed(trades[-15:])
    )
    blocks = [
        ("System", latest.get("run_status", "NO_DATA")),
        ("Signal", latest.get("action", "NO_TRADE")),
        ("BTC price", money(latest.get("price"))),
        ("Paper equity", money(latest.get("equity_usd", state.get("cash", 10000)))),
        ("BTC quantity", round(float(state.get("qty", 0)), 6)),
        ("Stop", money(state.get("stop"))),
    ]
    cards = "".join(f'<section><small>{safe(k)}</small><strong>{safe(v)}</strong></section>' for k, v in blocks)
    regime_blocks = [
        ("Status", regime_latest.get("run_status", "PENDING_FIRST_RUN")),
        ("Decision", regime_latest.get("action", "NO_DATA")),
        ("Reason", regime_latest.get("reason", "NO_DATA")),
        ("Last closed 4h RSI", (regime_latest.get("features") or {}).get("rsi4h")),
        ("Paper equity", money(regime_latest.get("equity_usd", regime_state.get("cash")))),
        ("Closed trades", len(regime_trades)),
    ]
    regime_cards = "".join(
        f'<section><small>{safe(key)}</small><strong>{safe(value)}</strong></section>'
        for key, value in regime_blocks
    )
    shadow_blocks = [
        ("Status", shadow_latest.get("run_status", "PENDING_FIRST_RUN")),
        ("Daily signal", shadow_latest.get("signal", "NO_DATA")),
        ("Action", shadow_latest.get("action", "NO_DATA")),
        ("SMA(200)", money(shadow_latest.get("sma200"))),
        ("Paper equity", money(shadow_latest.get("equity_usd"))),
        ("Closed trades", len(shadow_trades)),
    ]
    shadow_cards = "".join(
        f'<section><small>{safe(key)}</small><strong>{safe(value)}</strong></section>'
        for key, value in shadow_blocks
    )
    if audit_report.get("status") == "OK":
        audit_blocks = [
            ("Forward SMA return", str(audit_report.get("sma_net_return_pct")) + "%"),
            ("Same-timestamp Buy & Hold", str(audit_report.get("hold_return_pct")) + "%"),
            ("Paper checkpoints", audit_report.get("observed_daily_decisions")),
            ("Execution prices measured", str(audit_report.get("execution_events_measured")) +
             " / " + str(audit_report.get("execution_events"))),
            ("Average extra delay", str(audit_report.get("average_execution_delay_minutes"))
             + " min" if audit_report.get("average_execution_delay_minutes") is not None else "n/a"),
            ("Adverse quote drift", str(audit_report.get("average_adverse_price_drift_bps"))
             + " bps" if audit_report.get("average_adverse_price_drift_bps") is not None else "n/a"),
        ]
        audit_cards = "".join(
            f'<section><small>{safe(name)}</small><strong>{safe(value)}</strong></section>'
            for name, value in audit_blocks
        )
        recent_audits = "".join(
            f'<tr><td>{safe(e.get("side"))}</td>'
            f'<td>{safe(e.get("paper_execution_time"))}</td>'
            f'<td>{money(e.get("reference_open_usd"))}</td>'
            f'<td>{money(e.get("paper_quote_usd"))}</td>'
            f'<td>{safe(e.get("adverse_price_drift_bps"))}</td>'
            f'<td>{safe(e.get("status"))}</td></tr>'
            for e in reversed(audit_executions[-10:])
        )
        audit_panel = (
            '<section><h2>Forward execution audit</h2>'
            '<p>Actual recorded paper decisions only. Quote drift is relative to the '
            'UTC next-day opening price, not verified exchange slippage.</p>'
            f'<div class="grid">{audit_cards}</div>'
            f'<small>Paper valuation: {safe(audit_report.get("as_of"))}. '
            'Buy & Hold uses 100% exposure with the same timestamps; '
            'SMA200 limits new entries to 50% cash.</small>'
            '<div class="scroll"><table><thead><tr><th>Side</th><th>Paper time</th>'
            '<th>Next-open</th><th>Paper quote</th><th>Adverse drift (bps)</th>'
            '<th>Status</th></tr></thead><tbody>'
            f'{recent_audits or "<tr><td colspan=6>No executed paper events yet</td></tr>"}'
            '</tbody></table></div></section>'
        )
    else:
        audit_panel = ('<section><h2>Forward execution audit</h2>'
                       '<p>Awaiting the first validated forward audit.</p></section>')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTC Rule Desk</title><style>
:root{{color-scheme:dark;font-family:system-ui;background:#10181d;color:#eef3f4}}
body{{max-width:1050px;margin:auto;padding:32px 18px;line-height:1.6}}
small{{display:block;color:#9bb0b7}}h1{{font-size:clamp(26px,5vw,45px)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
section,.chart{{background:#1b292f;border:1px solid #33464c;border-radius:14px;padding:18px;margin-bottom:12px}}
strong{{display:block;font-size:22px;overflow-wrap:anywhere}}
svg{{width:100%}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;border-bottom:1px solid #33464c;text-align:left}}
.scroll{{overflow-x:auto}}footer{{font-size:13px;color:#a0b3ba;margin-top:28px}}
</style></head><body><small>BTC-USD / HOURLY / PAPER ONLY / NO AI</small>
<h1>Bitcoin Rule Desk</h1>
<p>Hourly baseline: Donchian 55 breakout, EMA 200 and ATR 14. Spot only, no leverage.</p>
<div class="grid">{cards}</div>
<section><h2>4h RSI regime / 1h entries (experimental shadow)</h2>
<p>Independent paper-only hypothesis: completed 4h RSI(20) above 70 permits
1h pullback or breakout entries. Maximum three entries per regime, capped spot
allocation, 1.25% candle-close exit, 2.5% simulated emergency stop, 24h time exit.
Fees and slippage are assumed; no exchange stops or real orders.</p>
<div class="grid">{regime_cards}</div>
<small>Last observed paper decision: {safe(regime_latest.get("time", "Not started"))}.
The baseline and regime portfolios have separate cash and start dates; raw
equity is not a fair performance comparison until aligned.</small></section>
<section><h2>SMA200 Daily Shadow</h2>
<p>Independent BTC-USD daily close vs SMA(200). Long or cash, max 50% paper allocation.
Only completed daily candles. 15 bps fee and 3 bps slippage assumed per side.
No leverage, no real orders, and no guarantee of profit.</p>
<div class="grid">{shadow_cards}</div>
<small>Last daily decision: {safe(shadow_latest.get("time", "Not started"))}.
This equity is from a separately initialized paper portfolio and may have a different valuation timestamp.</small>
</section>{audit_panel}<section class="chart"><h2>Recent closed candles</h2>{graph or "No chart data"}</section>
<section><h2>Closed trades: {len(trades)}</h2><div class="scroll"><table><thead>
<tr><th>Reason</th><th>Net PnL</th><th>Entry</th><th>Exit</th></tr></thead>
<tbody>{rows or '<tr><td colspan="4">No closed trades</td></tr>'}</tbody></table></div></section>
<footer>Simulated only. No real orders. Stop fills are estimates, not protected exchange orders.
Assumed fee 15 bps and slippage 3 bps per side. Last run: {safe(latest.get("time"))}.
<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast" style="color:#76d6c9">Code</a>.</footer>
</body></html>'''
    (target / "index.html").write_text(page, encoding="utf-8")
    for name, payload in (("latest", latest), ("history", history), ("trades", trades),
                          ("candles", candles)):
        (target / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n",
                                             encoding="utf-8")
    for name, payload in (("sma200_latest", shadow_latest),
                          ("sma200_history", shadow_history),
                          ("sma200_trades", shadow_trades)):
        (target / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n",
                                             encoding="utf-8")
    for name, payload in (("sma200_forward_report", audit_report),
                          ("sma200_forward_executions", audit_executions),
                          ("sma200_forward_snapshots", audit_snapshots)):
        (target / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n",
                                             encoding="utf-8")
    (target / ".nojekyll").touch()


if __name__ == "__main__":
    main()
