# SMA200 Forward Paper Evaluation and Execution-Time Drift

This is a **read-only** audit of the running `SMA200_DAILY_SPOT_V1` paper strategy. The paper engine, SMA length (200), max new cash allocation (50%), fee (15bps/side), and modeled slippage (3bps/side) are **not changed**.

## Automatic flow

1. The existing daily shadow workflow records its true paper decision and ticker quote on branch `sma-shadow-state`.
2. After the shadow workflow succeeds, `.github/workflows/sma-forward-audit.yml` fetches the paper state, trades, decision history and candles. It also fetches the existing `sma-audit-state` output if available.
3. `scripts/sma_forward_audit.py` computes the next-day **00:00 UTC open** reference from the closed Coinbase daily candle or (for the still-open UTC day) from the first **completed 00:00-01:00 hourly candle**. If the reference is unavailable, it marks the event `PENDING_REFERENCE`; it NEVER makes up a fill.
4. For each real paper BUY or SELL, compare the logged public ticker **quote before modeled slippage** with the next-day open reference at the size recorded by the paper strategy. A positive adverse-price-drift number means a worse paper quote than the historical next-open assumption.
5. Preserve each new observed paper checkpoint in a separate cumulative `sma-audit-state` branch. Repeated workflows do not insert duplicates or recalculate decisions that did not occur.
6. The existing GitHub Pages deployment reads this audit branch and displays forward SMA equity, same-timestamp Buy & Hold, cash, execution counts, delay and quote drift. Neither trading-state branch is written by the auditor.

The audit reports only **real recorded paper observation dates**, not hypothetical intervening days. Its first checkpoint is the existing first paper observation, not a retroactively synthesized trade. The comparative Buy & Hold simulation starts at the paper inception timestamp with $10,000 theoretical investment and modeled entry fee and slippage. SMA returns are marked-to-market using the paper record; a separate estimated liquidation valuation is provided. The comparison is **not equal-risk**: Buy & Hold uses 100% exposure, SMA can invest at most half of cash on each new entry.

## Metrics

- `paper_mark_to_market_equity_usd`: the same number recorded in the SMA daily paper state, at its event time.
- `sma_net_return_pct`: return from $10,000 initial capital after actual *simulated* fees at entries and exits (open BTC is marked without final liquidation cost).
- `paper_liquidation_estimate_usd`: additionally assumes exiting remaining BTC at the last recorded quote, less the modeled exit fee and slippage. No such exit was executed.
- `hold_return_pct`: simulated Buy & Hold, measured on the **same observed price and time** as the SMA paper checkpoint, with entry fee and slippage applied.
- `sma_max_observed_drawdown_pct`: drawdown measured using daily paper decision checkpoints, which can understate intraday exposure risk.
- `execution_events_measured` and `execution_events_reference_pending`: number of actual modeled buy/sell events with or without a historical next-day opening price.
- `delay_minutes`: recorded paper execution time minus the scheduled next-day 00:00 UTC reference.
- `adverse_price_drift_bps`: signed quote difference in basis points (positive means worse quote versus reference, negative means better).
- `incremental_cost_usd_at_recorded_qty`: approximate price-only dollar difference **at the actual recorded BTC quantity**, not a counterfactual portfolio replay and not a fee added to the ledger.

**Important:** the "paper fill" is a modeled fill based on the *public last-trade quote*, not an executed exchange order. This comparison measures timing-related quote drift and does not measure true spread, order-book impact, broker slippage, liquidity or maker/taker account fees. Trade-count samples begin at zero and must not be interpreted as reliable expected profit forecasts.

## Reproduce

```bash
python -m unittest discover -s tests -p test_sma_forward_audit.py -v
python scripts/sma_forward_audit.py --shadow-dir .sma_shadow --out-dir .sma_audit
python scripts/render_rules_dashboard.py
```

The workflow first retrieves the five historical paper JSON files and the three previous audit JSON files from their GitHub state branches. Output is:

- `report.json`: daily summary, cost assumptions, execution-latency and quote-drift aggregates
- `snapshots.json`: only actually observed paper daily decisions and aligned baseline values
- `executions.json`: the executed paper BUY/SELL events with reference-price comparison or pending status

The auditor aborts without publishing a new snapshot on inconsistent balances, overwritten observed history, future timestamps or parameter changes. It is **research-only**. No exchange credentials or real orders are involved.

## Constraints

- GitHub cron can be delayed or skipped; this workflow cannot restore missed real-time observations.
- The actual SMA paper strategy does not rebalance existing BTC daily and has no intraday stop.
- When an older reference opening price is no longer in the rolling daily candle snapshot, a targeted Coinbase historical query is used. Missing reference prices remain explicitly unresolved.
- The audit's snapshot branch uses the repository's existing snapshot-push mechanism; use GitHub run artifacts and commit logs for redundancy until a genuinely append-only storage design is implemented.
- No change to the SMA200 rule or its paper account is authorized by audit findings alone.

Coinbase historical candle reference:
https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles

GitHub scheduled execution notes:
https://docs.github.com/en/actions/reference/events-that-trigger-workflows#schedule
