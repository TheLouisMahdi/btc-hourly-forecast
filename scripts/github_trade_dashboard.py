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
        "Aggressive structural entries with risk-scaled sizing and adaptive exits",
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
    status = str(
        active.get("status") if active else plan.get("status") or "WAIT"
    )
    direction = str(
        active.get("direction")
        if active
        else latest.get("action")
        or latest.get("trade_forecast_direction")
        or "WAIT"
    )
    entry = _number(source.get("entry_price", source.get("entry_reference")))
    target = _number(source.get("target_price"))
    stop = _number(source.get("current_stop_price", source.get("stop_price")))
    target_profit = _number(
        source.get("target_net_profit_usd", plan.get("target_net_profit_usd"))
    )
    stop_loss = _number(
        source.get("stop_net_loss_usd", plan.get("stop_net_loss_usd"))
    )
    expected_value = _number(
        source.get("expected_value_usd", plan.get("expected_value_usd"))
    )
    risk_budget = _number(
        source.get("risk_budget_usd", plan.get("risk_budget_usd"))
    )
    risk_fraction = _number(
        source.get("risk_fraction", plan.get("risk_fraction"))
    )
    margin = _number(
        source.get("margin_required_usd", plan.get("margin_required_usd"))
    )
    leverage = _number(
        source.get("suggested_leverage", plan.get("suggested_leverage"))
    )
    notional = _number(
        source.get("notional_usd", plan.get("notional_usd"))
    )
    reward_r = _number(
        source.get("risk_reward", plan.get("adaptive_reward_r"))
    )
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
    profit_probability = _number(
        source.get(
            "adaptive_profit_probability",
            plan.get("adaptive_profit_probability"),
        )
    )
    loss_probability = _number(
        source.get(
            "adaptive_loss_probability",
            plan.get("adaptive_loss_probability"),
        )
    )
    metrics = source.get("position_metrics", plan.get("position_metrics", {}))
    metrics = metrics if isinstance(metrics, dict) else {}
    ev_to_risk = _number(metrics.get("expected_value_to_risk"))
    net_reward_risk = _number(metrics.get("net_reward_risk"))
    neighbor_loss_rate = _number(metrics.get("neighbor_loss_rate"))

    if ev_to_risk is None and expected_value is not None and risk_budget:
        ev_to_risk = expected_value / risk_budget
    if (
        net_reward_risk is None
        and target_profit is not None
        and stop_loss is not None
        and stop_loss < 0
    ):
        net_reward_risk = target_profit / abs(stop_loss)

    if active:
        decision = "MANAGE"
        decision_reason = "OPEN_POSITION_CONTRACT"
    else:
        decision = str(plan.get("position_decision") or "SCANNING")
        decision_reason = str(
            plan.get("position_decision_reason") or "WAITING_FOR_SETUP"
        )

    risks = source.get("position_risks", plan.get("position_risks", []))
    risks = risks if isinstance(risks, list) else []
    strengths = source.get(
        "position_strengths",
        plan.get("position_strengths", []),
    )
    strengths = strengths if isinstance(strengths, list) else []
    soft_flags = source.get("soft_risk_flags", plan.get("soft_risk_flags", []))
    soft_flags = soft_flags if isinstance(soft_flags, list) else []

    loss_guard = source.get("loss_memory_guard", plan.get("loss_memory_guard", {}))
    loss_guard = loss_guard if isinstance(loss_guard, dict) else {}
    neighbor_memory = loss_guard.get("neighbor_memory")
    neighbor_memory = (
        neighbor_memory if isinstance(neighbor_memory, dict) else {}
    )
    if bool(loss_guard.get("veto", False)):
        memory_state = "BLOCK"
    elif bool(neighbor_memory.get("supported_bad_pattern", False)):
        memory_state = "BAD PATTERN"
    elif neighbor_loss_rate is not None:
        memory_state = "CLEAR"
    else:
        memory_state = "LEARNING"

    pnl = _number(summary.get("net_pnl_usd"))
    win_rate = _number(summary.get("net_win_rate"))
    resolved = summary.get("resolved_trades")
    holding = source.get(
        "maximum_holding_hours",
        plan.get("maximum_holding_hours"),
    )
    expiry = source.get("expires_at")

    decision_class = (
        "open"
        if active
        else "good"
        if decision == "FAVORABLE"
        else "bad"
        if decision == "BLOCKED"
        else "waiting"
    )
    main_reason = decision_reason.replace("_", " ")
    evidence_note = (
        " · ".join(
            [
                *[str(item).replace("_", " ") for item in strengths[:2]],
                *[str(item).replace("_", " ") for item in risks[:2]],
            ]
        )
        or "Evidence is still mixed or limited"
    )
    warning_note = (
        ", ".join(str(item).replace("_", " ") for item in soft_flags[:3])
        if soft_flags
        else "No material soft warnings"
    )

    probability_value = (
        f"PROFIT {_percent(profit_probability)} · LOSS {_percent(loss_probability)}"
        if profit_probability is not None or loss_probability is not None
        else f"TARGET {_percent(target_probability)} · STOP {_percent(stop_probability)}"
    )
    probability_note = (
        f"Target {_percent(target_probability)} · Stop {_percent(stop_probability)}"
    )
    memory_note = (
        f"Nearest-pattern loss rate: {_percent(neighbor_loss_rate)}"
        if neighbor_loss_rate is not None
        else "Waiting for enough comparable resolved positions"
    )

    return f"""
<section class="panel trade-lifecycle-panel">
  <div class="trade-lifecycle-heading">
    <div>
      <div class="structure-eyebrow">Profit-first position brief</div>
      <h2>Position decision</h2>
      <p class="sub">The important trading information is concentrated here: decision, entry path, payoff, probability, risk and the main reason behind the position.</p>
    </div>
    <span class="trade-lifecycle-state {decision_class}">{_escape(direction)} · {_escape(decision)}</span>
  </div>

  <div class="profit-decision-strip">
    {_decision_card("Expected value", _money(expected_value), f"{_signed_percent(ev_to_risk)} of risk budget")}
    {_decision_card("Profit / loss", f"{_money(target_profit)} / {_money(stop_loss)}", f"Net reward/risk {_ratio(net_reward_risk)}")}
    {_decision_card("Probability", probability_value, probability_note)}
    {_decision_card("Loss memory", memory_state, memory_note)}
  </div>

  <div class="trade-route">
    <div><small>ENTRY</small><strong>{_price(entry)}</strong></div>
    <span>→</span>
    <div class="target"><small>TAKE PROFIT</small><strong>{_price(target)}</strong></div>
    <span>or</span>
    <div class="stop"><small>STOP</small><strong>{_price(stop)}</strong></div>
  </div>

  <div class="position-decision-note">
    <strong>{_escape(main_reason)}</strong>
    <span>{_escape(evidence_note)}</span>
  </div>

  <div class="trade-lifecycle-grid compact">
    {_tile("Risk capital", f"{_money(risk_budget)} · {_percent(risk_fraction)}", "Maximum planned capital risk")}
    {_tile("Position size", _money(notional), "Paper notional exposure")}
    {_tile("Margin / leverage", f"{_money(margin)} · {_x(leverage)}", "Capital required for this position")}
    {_tile("Reward target", _r(reward_r), "Adaptive target relative to initial risk")}
    {_tile("Holding", f"{_escape(holding)}h" if holding else "—", f"Expires: {_escape(expiry or '—')}")}
    {_tile("Track record", f"{_percent(win_rate)} · {_money(pnl)}", f"{_escape(resolved if resolved is not None else 0)} resolved positions")}
  </div>

  <div class="position-technical-line">
    <span><strong>Warnings:</strong> {_escape(warning_note)}</span>
    <span><strong>Status:</strong> {_escape(status)}</span>
  </div>
</section>"""


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
        outcome = str(
            trade.get("historical_corrected_result")
            or trade.get("outcome")
            or "CLOSED"
        ).replace("_", " ")
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


def _decision_card(title: str, value: str, note: str) -> str:
    return f"""
<div class="profit-decision-card">
  <span>{_escape(title)}</span>
  <strong>{value}</strong>
  <small>{_escape(note)}</small>
</div>"""


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
    return r"""
.trade-lifecycle-panel{position:relative;margin-top:18px;overflow:hidden;background:linear-gradient(135deg,rgba(255,255,255,.94),rgba(236,234,245,.56),rgba(222,236,231,.56))}
.trade-lifecycle-heading{position:relative;display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:16px}
.trade-lifecycle-heading h2{margin:5px 0 8px;font-size:clamp(1.4rem,3vw,2.1rem)}
.trade-lifecycle-state{display:inline-flex;padding:10px 14px;border-radius:999px;font-size:.76rem;font-weight:900;letter-spacing:.06em;border:1px solid var(--line);white-space:nowrap}
.trade-lifecycle-state.open{background:rgba(190,151,73,.14);color:#6f4d25}.trade-lifecycle-state.good{background:rgba(77,139,118,.10);color:var(--ok)}.trade-lifecycle-state.bad{background:rgba(189,114,110,.10);color:var(--bad)}.trade-lifecycle-state.waiting{background:rgba(167,111,77,.09);color:#80604b}
.profit-decision-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:14px}
.profit-decision-card{padding:14px;border:1px solid var(--line);border-radius:17px;background:rgba(255,255,255,.70);min-width:0}.profit-decision-card span{display:block;color:var(--muted);font-size:.65rem;text-transform:uppercase;letter-spacing:.07em}.profit-decision-card strong{display:block;margin-top:6px;font-size:1rem;line-height:1.3;overflow-wrap:anywhere}.profit-decision-card small{display:block;margin-top:5px;color:var(--muted);font-size:.66rem;line-height:1.4}
.trade-route{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:10px;margin:0 0 12px;padding:14px;border:1px solid var(--line);border-radius:18px;background:rgba(255,255,255,.58)}.trade-route div{min-width:0}.trade-route small{display:block;color:var(--muted);font-size:.64rem;letter-spacing:.09em}.trade-route strong{display:block;margin-top:4px;font-size:clamp(1rem,2.4vw,1.35rem);overflow-wrap:anywhere}.trade-route .target strong{color:var(--sage2)}.trade-route .stop strong{color:#a56663}.trade-route>span{color:var(--muted);font-size:.74rem}
.position-decision-note{display:flex;gap:12px;align-items:center;margin:0 0 12px;padding:11px 13px;border-left:3px solid var(--accent);background:rgba(255,255,255,.46);border-radius:0 13px 13px 0}.position-decision-note strong{font-size:.78rem}.position-decision-note span{color:var(--muted);font-size:.72rem}
.trade-lifecycle-grid.compact{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.trade-lifecycle-tile{padding:13px;border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.52);min-height:94px}.trade-lifecycle-tile span{display:block;color:var(--muted);font-size:.64rem;text-transform:uppercase;letter-spacing:.07em}.trade-lifecycle-tile strong{display:block;margin-top:6px;font-size:.94rem;line-height:1.3;overflow-wrap:anywhere}.trade-lifecycle-tile small{display:block;margin-top:5px;color:var(--muted);font-size:.64rem;line-height:1.4}
.position-technical-line{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-top:11px;color:var(--muted);font-size:.66rem}.position-technical-line strong{color:var(--ink)}
.position-ledger-heading{display:flex;justify-content:space-between;align-items:flex-start;gap:18px;margin-bottom:16px}.position-summary{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px}.position-summary-chip{display:grid;padding:8px 11px;border-radius:13px;border:1px solid var(--line);background:rgba(255,255,255,.68)}.position-summary-chip small{color:var(--muted);font-size:8px;text-transform:uppercase}.position-summary-chip strong{margin-top:2px;font-size:11px}.pnl-positive,.position-summary-chip.pnl-positive{color:var(--ok);background:rgba(77,139,118,.08)}.pnl-negative,.position-summary-chip.pnl-negative{color:var(--bad);background:rgba(189,114,110,.08)}.pnl-neutral,.position-summary-chip.neutral{color:var(--muted)}
@media(max-width:980px){.profit-decision-strip{grid-template-columns:repeat(2,minmax(0,1fr))}.trade-lifecycle-grid.compact{grid-template-columns:repeat(2,minmax(0,1fr))}.position-ledger-heading{flex-direction:column}.position-summary{justify-content:flex-start}}
@media(max-width:620px){.trade-lifecycle-heading{flex-direction:column}.trade-lifecycle-state{white-space:normal}.profit-decision-strip,.trade-lifecycle-grid.compact{grid-template-columns:1fr}.trade-route{grid-template-columns:1fr}.trade-route>span{display:none}.position-decision-note{align-items:flex-start;flex-direction:column}}
"""


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


def _ratio(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.2f}:1"


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
