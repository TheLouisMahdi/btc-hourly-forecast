from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    index_path = root / "site" / "index.html"
    latest = _load(root / ".github_state" / "latest.json", {})
    trades = _load(root / ".github_state" / "trades.json", [])
    if not index_path.exists():
        raise FileNotFoundError("Dashboard HTML must be rendered first")
    trades = trades if isinstance(trades, list) else []
    document = index_path.read_text(encoding="utf-8")
    document = document.replace(
        "BTC Economic Breakout Forecast",
        "BTC Adaptive Target–Stop Trader",
    )
    document = document.replace(
        "Cost-aware structural signals with locked economic validation",
        "Aggressive paper orders with adaptive targets, stop-loss and online learning",
    )
    document = document.replace(
        "NEXT CLOSED 1-HOUR CANDLE",
        "SECONDARY EXACT NEXT-CLOSE FORECAST",
    )
    document = document.replace(
        "</style>",
        _styles() + "\n</style>",
        1,
    )
    panel = _panel(latest, trades)
    markers = (
        '<section class="panel boundary-memory-panel">',
        '<section class="panel economic-panel">',
        '<section class="panel ledger">',
    )
    for marker in markers:
        if marker in document:
            document = document.replace(marker, panel + "\n" + marker, 1)
            break

    ledger = _position_ledger(latest, trades)
    document, replacements = re.subn(
        r'<section class="panel ledger">.*?</section>',
        ledger,
        document,
        count=1,
        flags=re.DOTALL,
    )
    if replacements == 0:
        document = document.replace("</main>", ledger + "\n</main>", 1)
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
    stop = _number(source.get("current_stop_price", source.get("stop_price")))
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
    holding = source.get("maximum_holding_hours", plan.get("maximum_holding_hours"))
    samples = summary.get("samples_seen")
    resolved = summary.get("resolved_trades")
    pnl = _number(summary.get("net_pnl_usd"))
    average_r = _number(summary.get("average_r"))
    target_rate = _number(summary.get("target_hit_rate"))
    mode = str(
        plan.get("decision_mode")
        or latest.get("paper_trade_mode")
        or "AGGRESSIVE_PAPER"
    )
    state_class = "open" if active else "waiting"
    verdict = "OPEN POSITION" if active else "SCANNING / READY"

    return f'''
<section class="panel trade-lifecycle-panel">
  <div class="trade-lifecycle-heading">
    <div>
      <div class="structure-eyebrow">Primary contract · target, stop or time exit</div>
      <h2>Adaptive paper-trade lifecycle</h2>
      <p class="sub">Each recommendation becomes one persistent LONG or SHORT paper position. It remains open across hourly candles until target, stop-loss or time exit, then its realized P&amp;L and R-multiple are learned online.</p>
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
    {_tile("Adaptive reward", _r(reward_r), "Starts at 5R and adapts only from resolved trade outcomes")}
    {_tile("Target probability", _percent(target_probability), "Probability of target before stop")}
    {_tile("Stop probability", _percent(stop_probability), "Probability of stop before target")}
    {_tile("Target net profit", _money(target_profit), "After stress fees and slippage")}
    {_tile("Stop net loss", _money(stop_loss), "Risk including stress execution cost")}
    {_tile("Expected value", _money(expected_value), "Probability-weighted paper-trade value")}
    {_tile("Margin / leverage", f"{_money(margin)} · {_x(leverage)}", "Suggested paper margin and leverage")}
    {_tile("Holding contract", f"{_escape(holding)}h" if holding else "—", f"Expires: {_escape(expiry or '—')}")}
    {_tile("Online samples", _escape(samples if samples is not None else 0), "Only resolved target/stop/time-exit positions")}
    {_tile("Resolved positions", _escape(resolved if resolved is not None else 0), f"Target hit rate: {_percent(target_rate)}")}
    {_tile("Learned performance", f"{_money(pnl)} · {_r(average_r)} avg", "Cumulative realized paper PnL and average R")}
  </div>
</section>'''


def _position_ledger(latest: dict[str, Any], trades: list[Any]) -> str:
    positions = [
        item
        for item in trades
        if isinstance(item, dict)
        and str(item.get("direction") or "").upper() in {"LONG", "SHORT"}
    ]
    latest_price = _number(latest.get("price"))
    snapshots = [_position_snapshot(item, latest_price) for item in positions]
    closed = [item for item in snapshots if item["status"] == "CLOSED"]
    opened = [item for item in snapshots if item["status"] == "OPEN"]
    realized = sum(item["pnl_usd"] or 0.0 for item in closed)
    unrealized = sum(item["pnl_usd"] or 0.0 for item in opened)
    wins = sum((item["pnl_usd"] or 0.0) > 0 for item in closed)
    win_rate = wins / len(closed) if closed else None
    rows = _position_rows(snapshots)
    return f'''
<section class="panel ledger position-ledger">
  <div class="position-ledger-heading">
    <div>
      <h2>LONG / SHORT position ledger</h2>
      <p class="sub">Only actual suggested paper positions are listed. Open P&amp;L is marked to the latest closed candle; closed P&amp;L is immutable and realized.</p>
    </div>
    <div class="position-summary">
      {_summary_chip("Positions", str(len(positions)), "neutral")}
      {_summary_chip("Open P/L", _money(unrealized), _pnl_class(unrealized))}
      {_summary_chip("Realized P/L", _money(realized), _pnl_class(realized))}
      {_summary_chip("Win rate", _percent(win_rate), "neutral")}
    </div>
  </div>
  <div class="scroll"><table>
    <thead><tr>
      <th>Opened</th><th>Position</th><th>Entry</th><th>Target</th><th>Stop</th>
      <th>Mark / Exit</th><th>P/L USD</th><th>P/L %</th><th>R</th><th>Status</th><th>Closed</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</section>'''


def _position_rows(snapshots: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for item in reversed(snapshots[-50:]):
        direction = item["direction"]
        pnl_class = _pnl_class(item["pnl_usd"])
        status_class = "result-wait" if item["status"] == "OPEN" else pnl_class
        rows.append(
            "<tr>"
            f"<td>{_escape(_time(item['opened_at']))}</td>"
            f'<td><span class="pill {"up" if direction == "LONG" else "down"}">{_escape(direction)}</span></td>'
            f"<td>{_price(item['entry'])}</td>"
            f"<td>{_price(item['target'])}</td>"
            f"<td>{_price(item['stop'])}</td>"
            f"<td>{_price(item['mark'])}</td>"
            f'<td class="{pnl_class}">{_money(item["pnl_usd"])}</td>'
            f'<td class="{pnl_class}">{_signed_percent(item["net_return"])}</td>'
            f'<td class="{pnl_class}">{_r(item["realized_r"])}</td>'
            f'<td><span class="pill {status_class}">{_escape(item["outcome"])}</span></td>'
            f"<td>{_escape(_time(item['closed_at']))}</td>"
            "</tr>"
        )
    if rows:
        return "".join(rows)
    return '<tr><td colspan="11" class="empty">No LONG or SHORT position has been opened yet.</td></tr>'


def _position_snapshot(
    trade: dict[str, Any],
    latest_price: float | None,
) -> dict[str, Any]:
    status = str(trade.get("status") or "OPEN").upper()
    direction = str(trade.get("direction") or "").upper()
    entry = _number(trade.get("entry_price"))
    target = _number(trade.get("target_price"))
    stop = _number(
        trade.get("current_stop_price")
        if status == "OPEN"
        else trade.get("initial_stop_price", trade.get("current_stop_price"))
    )
    mark = _number(trade.get("exit_price")) if status == "CLOSED" else latest_price
    if status == "CLOSED":
        pnl_usd = _number(trade.get("realized_net_pnl_usd"))
        net_return = _number(trade.get("realized_net_return"))
        realized_r = _number(trade.get("realized_r"))
        outcome = str(trade.get("outcome") or "CLOSED").replace("_", " ")
    else:
        pnl_usd, net_return, realized_r = _open_mark_to_market(
            trade,
            mark,
            entry,
            direction,
        )
        outcome = "OPEN"
    return {
        "status": status,
        "direction": direction,
        "opened_at": trade.get("opened_at"),
        "closed_at": trade.get("closed_at"),
        "entry": entry,
        "target": target,
        "stop": stop,
        "mark": mark,
        "pnl_usd": pnl_usd,
        "net_return": net_return,
        "realized_r": realized_r,
        "outcome": outcome,
    }


def _open_mark_to_market(
    trade: dict[str, Any],
    mark: float | None,
    entry: float | None,
    direction: str,
) -> tuple[float | None, float | None, float | None]:
    if mark is None or entry is None or entry <= 0:
        return None, None, None
    gross_return = mark / entry - 1.0
    aligned_return = gross_return if direction == "LONG" else -gross_return
    stress_cost = (_number(trade.get("stress_execution_cost_bps")) or 0.0) / 10_000.0
    net_return = aligned_return - stress_cost
    notional = _number(trade.get("notional_usd")) or 0.0
    pnl = notional * net_return
    risk_budget = _number(trade.get("risk_budget_usd")) or 0.0
    realized_r = pnl / risk_budget if risk_budget > 0 else None
    return pnl, net_return, realized_r


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


def _summary_chip(title: str, value: str, style: str) -> str:
    return f'<span class="position-summary-chip {style}"><small>{_escape(title)}</small><strong>{value}</strong></span>'


def _styles() -> str:
    return r'''
.trade-lifecycle-panel{position:relative;margin-top:18px;overflow:hidden;background:linear-gradient(135deg,rgba(255,255,255,.93),rgba(236,234,245,.62),rgba(222,236,231,.62))}
.trade-lifecycle-panel:before{content:"";position:absolute;right:-90px;top:-130px;width:360px;height:360px;border-radius:50%;background:radial-gradient(circle,rgba(143,138,184,.18),transparent 68%);pointer-events:none}
.trade-lifecycle-heading{position:relative;display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:18px}
.trade-lifecycle-heading h2{margin:5px 0 8px;font-size:clamp(1.4rem,3vw,2.1rem)}
.trade-lifecycle-state{display:inline-flex;padding:10px 14px;border-radius:999px;font-size:.76rem;font-weight:900;letter-spacing:.07em;border:1px solid rgba(255,255,255,.12)}
.trade-lifecycle-state.open{background:rgba(80,210,160,.14);color:#47746b}.trade-lifecycle-state.waiting{background:rgba(143,138,184,.13);color:#69648e}
.trade-route{position:relative;display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:10px;margin:0 0 16px;padding:15px;border:1px solid var(--line);border-radius:20px;background:rgba(255,255,255,.58)}
.trade-route div{min-width:0}.trade-route small{display:block;color:var(--muted);font-size:.65rem;letter-spacing:.09em}.trade-route strong{display:block;margin-top:4px;font-size:clamp(1rem,2.5vw,1.45rem);overflow-wrap:anywhere}.trade-route .target strong{color:var(--sage2)}.trade-route .stop strong{color:#a56663}.trade-route>span{color:var(--muted);font-size:.75rem}
.trade-lifecycle-grid{position:relative;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.trade-lifecycle-tile{padding:16px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.58);min-height:118px}.trade-lifecycle-tile span{display:block;color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.07em}.trade-lifecycle-tile strong{display:block;margin-top:8px;font-size:1rem;line-height:1.35;overflow-wrap:anywhere}.trade-lifecycle-tile small{display:block;margin-top:7px;color:var(--muted);font-size:.68rem;line-height:1.45}
.position-ledger-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;margin-bottom:16px}.position-summary{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px}.position-summary-chip{display:grid;padding:8px 11px;border-radius:13px;border:1px solid var(--line);background:rgba(255,255,255,.68)}.position-summary-chip small{color:var(--muted);font-size:8px;text-transform:uppercase}.position-summary-chip strong{margin-top:2px;font-size:11px}.pnl-positive,.position-summary-chip.pnl-positive{color:var(--ok);background:rgba(77,139,118,.08)}.pnl-negative,.position-summary-chip.pnl-negative{color:var(--bad);background:rgba(189,114,110,.08)}.pnl-neutral,.position-summary-chip.neutral{color:var(--muted)}
@media(max-width:980px){.trade-lifecycle-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.position-ledger-heading{flex-direction:column}.position-summary{justify-content:flex-start}}
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
    return "—" if number is None else f"{number * 100:.2f}%"


def _signed_percent(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number * 100:+.2f}%"


def _r(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:+.2f}R"


def _x(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.2f}×"


def _time(value: Any) -> str:
    if value in (None, ""):
        return "—"
    try:
        from pandas import Timestamp

        timestamp = Timestamp(value)
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        return timestamp.strftime("%b %d · %H:%M UTC")
    except Exception:
        return str(value)


def _pnl_class(value: Any) -> str:
    number = _number(value)
    if number is None or abs(number) < 1e-12:
        return "pnl-neutral"
    return "pnl-positive" if number > 0 else "pnl-negative"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
