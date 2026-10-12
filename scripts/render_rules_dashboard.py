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
    shadow_source = Path(".sma_shadow")
    shadow_latest = read(shadow_source / "latest.json", {})
    shadow_state = read(shadow_source / "state.json", {})
    shadow_trades = read(shadow_source / "trades.json", [])
    shadow_history = read(shadow_source / "history.json", [])

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
<h1>Bitcoin Rule Desk</h1><p>Donchian 55 breakout, EMA 200 regime filter, ATR 14 position sizing, spot only, no leverage.</p>
<div class="grid">{cards}</div>
<section><h2>SMA200 Daily Shadow</h2>
<p>Independent BTC-USD daily close vs SMA(200). Long or cash, max 50% paper allocation.
Only completed daily candles. 15 bps fee and 3 bps slippage assumed per side.
No leverage, no real orders, and no guarantee of profit.</p>
<div class="grid">{shadow_cards}</div>
<small>Last daily decision: {safe(shadow_latest.get("time", "Not started"))}.
This equity is from a separately initialized paper portfolio and may have a different valuation timestamp.</small>
</section><section class="chart"><h2>Recent closed candles</h2>{graph or "No chart data"}</section>
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
    (target / ".nojekyll").touch()


if __name__ == "__main__":
    main()
