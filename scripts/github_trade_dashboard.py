from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    index_path = root / "site" / "index.html"
    latest = _load(root / ".github_state" / "latest.json", {})
    trades = _load(root / ".github_state" / "trades.json", [])
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")
    document = index_path.read_text(encoding="utf-8")
    document = document.replace(
        "BTC Economic Breakout Forecast",
        "BTC Adaptive Target–Stop Trader",
    )
    document = document.replace(
        "Cost-aware structural signals with locked economic validation",
        "Aggressive paper orders with adaptive 5R targets, stop-loss and online learning",
    )
    document = document.replace(
        "NEXT CLOSED 1-HOUR CANDLE",
        "SECONDARY 1-HOUR FORECAST",
    )
    document = document.replace(
        "</style>",
        _styles() + "\n</style>",
        1,
    )
    panel = _panel(latest, trades if isinstance(trades, list) else [])
    markers = (
        '<section class="panel boundary-memory-panel">',
        '<section class="panel economic-panel">',
        '<section class="panel ledger">',
    )
    for marker in markers:
        if marker in document:
            document = document.replace(marker, panel + "\n" + marker, 1)
            break
    index_path.write_text(document, encoding="utf-8")
    return 0


def _panel(latest: dict[str, Any], trades: list[Any]) -> str:
    active = latest.get("active_trade")
    active = active if isinstance(active, dict) else _active_trade(trades)
    plan = latest.get("trade_plan")
    plan = plan if isinstance(plan, dict) else {}
    summary = latest.get("trade_lifecycle_summary")
    summary = summary if isinstance(summary, dict) else _summary(trades)

    source = active or plan
    status = str(active.get("status") if active else plan.get("status") or "WAIT")
    direction = str(
        active.get("direction")
        if active
        else latest.get("action") or latest.get("trade_forecast_direction") or "WAIT"
    )
    entry = _number(source.get("entry_price", source.get("entry_reference")))
    target = _number(source.get("target_price"))
    stop = _number(
        source.get("current_stop_price", source.get("stop_price"))
    )
    reward_r = _number(source.get("risk_reward", plan.get("adaptive_reward_r")))
    target_probability = _number(
        source.get(
            "adaptive_target_probability",
            plan.get("adaptive_target_probability"),
        )
    )
    stop_probability = _number(
        source.get(
            "adaptive_stop_probability",
            plan.get("adaptive_stop_probability"),
        )
    )
    target_profit = _number(
        source.get("target_net_profit_usd", plan.get("target_net_profit_usd"))
    )
    stop_loss = _number(
        source.get("stop_net_loss_usd", plan.get("stop_net_loss_usd"))
    )
    expected_value = _number(
        source.get("expected_value_usd", plan.get("expected_value_usd"))
    )
    margin = _number(
        source.get("margin_required_usd", plan.get("margin_required_usd"))
    )
    leverage = _number(
        source.get("suggested_leverage", plan.get("suggested_leverage"))
    )
    expiry = source.get("expires_at")
    holding = source.get(
        "maximum_holding_hours", plan.get("maximum_holding_hours")
    )
    samples = summary.get("samples_seen")
    resolved = summary.get("resolved_trades")
    pnl = _number(summary.get("net_pnl_usd"))
    average_r = _number(summary.get("average_r"))
    target_rate = _number(summary.get("target_hit_rate"))
    mode = str(plan.get("decision_mode") or latest.get("paper_trade_mode") or "AGGRESSIVE_PAPER")
    state_class = "open" if active else "waiting"
    verdict = "OPEN POSITION" if active else "SCANNING / READY"

    return f'''
<section class="panel trade-lifecycle-panel">
  <div class="trade-lifecycle-heading">
    <div>
      <div class="structure-eyebrow">Primary contract · target, stop or time exit</div>
      <h2>Adaptive paper-trade lifecycle</h2>
      <p class="sub">The next-candle range is secondary. The primary objective is to open one directional order, hold it across candles, close at the adaptive target or stop-loss, and feed the realized R-multiple back into the online learner.</p>
    </div>
    <span class="trade-lifecycle-state {state_class}">{_escape(verdict)}</span>
  </div>
  <div class="trade-route">
    <div><small>ENTRY</small><strong>{_price(entry)}</strong></div>
    <span>→</span>
    <div class="target"><small>TARGET</small><strong>{_price(target)}</strong></div>
    <span>or</span>
    <div class="stop"><small>STOP</small><strong>{_price(stop)}</strong></div>
  </div>
  <div class="trade-lifecycle-grid">
    {_tile("State", f"{direction} · {status}", mode)}
    {_tile("Adaptive reward", _r(reward_r), "Starts at 5R and moves between 3R and 8R after live outcomes")}
    {_tile("Target probability", _percent(target_probability), "Online probability of target before stop")}
    {_tile("Stop probability", _percent(stop_probability), "Online probability of stop before target")}
    {_tile("Target net profit", _money(target_profit), "After stress fees and slippage")}
    {_tile("Stop net loss", _money(stop_loss), "Risk including stress execution cost")}
    {_tile("Expected value", _money(expected_value), "Probability-weighted paper-trade value")}
    {_tile("Margin / leverage", f"{_money(margin)} · {_x(leverage)}", "Suggested paper margin and leverage")}
    {_tile("Holding contract", f"{_escape(holding)}h" if holding else "—", f"Expires: {_escape(expiry or '—')}")}
    {_tile("Online samples", _escape(samples if samples is not None else 0), "Only resolved target/stop/time-exit trades")}
    {_tile("Resolved trades", _escape(resolved if resolved is not None else 0), f"Target hit rate: {_percent(target_rate)}")}
    {_tile("Learned performance", f"{_money(pnl)} · {_r(average_r)} avg", "Cumulative paper PnL and average realized R")}
  </div>
</section>'''


def _active_trade(trades: list[Any]) -> dict[str, Any] | None:
    for item in reversed(trades):
        if isinstance(item, dict) and item.get("status") == "OPEN":
            return item
    return None


def _summary(trades: list[Any]) -> dict[str, Any]:
    resolved = [
        item
        for item in trades
        if isinstance(item, dict) and item.get("status") == "CLOSED"
    ]
    targets = sum(item.get("outcome") == "TARGET" for item in resolved)
    pnl = sum(_number(item.get("realized_net_pnl_usd")) or 0.0 for item in resolved)
    r_values = [
        _number(item.get("realized_r"))
        for item in resolved
        if _number(item.get("realized_r")) is not None
    ]
    return {
        "resolved_trades": len(resolved),
        "target_hit_rate": targets / len(resolved) if resolved else None,
        "net_pnl_usd": pnl,
        "average_r": sum(r_values) / len(r_values) if r_values else None,
        "samples_seen": sum(bool(item.get("adaptive_learned")) for item in resolved),
    }


def _tile(title: str, value: str, note: str) -> str:
    return f'''
<div class="trade-lifecycle-tile">
  <span>{_escape(title)}</span>
  <strong>{value}</strong>
  <small>{_escape(note)}</small>
</div>'''


def _styles() -> str:
    return r'''
.trade-lifecycle-panel{position:relative;margin-top:18px;overflow:hidden;background:linear-gradient(135deg,rgba(255,255,255,.93),rgba(236,234,245,.62),rgba(222,236,231,.62))}
.trade-lifecycle-panel:before{content:"";position:absolute;right:-90px;top:-130px;width:360px;height:360px;border-radius:50%;background:radial-gradient(circle,rgba(143,138,184,.18),transparent 68%);pointer-events:none}
.trade-lifecycle-heading{position:relative;display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:18px}
.trade-lifecycle-heading h2{margin:5px 0 8px;font-size:clamp(1.4rem,3vw,2.1rem)}
.trade-lifecycle-state{display:inline-flex;padding:10px 14px;border-radius:999px;font-size:.76rem;font-weight:900;letter-spacing:.07em;border:1px solid rgba(255,255,255,.12)}
.trade-lifecycle-state.open{background:rgba(80,210,160,.14);color:#47746b}
.trade-lifecycle-state.waiting{background:rgba(143,138,184,.13);color:#69648e}
.trade-route{position:relative;display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:10px;margin:0 0 16px;padding:15px;border:1px solid var(--line);border-radius:20px;background:rgba(255,255,255,.58)}
.trade-route div{min-width:0}.trade-route small{display:block;color:var(--muted);font-size:.65rem;letter-spacing:.09em}.trade-route strong{display:block;margin-top:4px;font-size:clamp(1rem,2.5vw,1.45rem);overflow-wrap:anywhere}.trade-route .target strong{color:var(--sage2)}.trade-route .stop strong{color:#a56663}.trade-route>span{color:var(--muted);font-size:.75rem}
.trade-lifecycle-grid{position:relative;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.trade-lifecycle-tile{padding:16px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.58);min-height:118px}
.trade-lifecycle-tile span{display:block;color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.07em}.trade-lifecycle-tile strong{display:block;margin-top:8px;font-size:1rem;line-height:1.35;overflow-wrap:anywhere}.trade-lifecycle-tile small{display:block;margin-top:7px;color:var(--muted);font-size:.68rem;line-height:1.45}
@media(max-width:980px){.trade-lifecycle-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:620px){.trade-lifecycle-heading{flex-direction:column}.trade-route{grid-template-columns:1fr}.trade-route>span{display:none}.trade-lifecycle-grid{grid-template-columns:1fr}}
'''


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _price(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"${number:,.2f}"


def _money(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"${number:+,.2f}"


def _percent(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number * 100:.1f}%"


def _r(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:+.2f}R"


def _x(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.2f}×"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
