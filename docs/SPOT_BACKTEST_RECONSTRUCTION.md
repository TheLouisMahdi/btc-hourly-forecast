# Coinbase Spot Backtest Reconstruction

Status: **isolated research, not production deployment**. The live paper bot on main and the `rule-state` ledger are unchanged.

## Purpose

Reconstruct the current Donchian(55) / EMA(200) / ATR(14) BTC-USD spot strategy using the **same** `rule_based.step()` decision and position accounting functions as the running GitHub paper engine. The replay corrects an earlier research mismatch: historical EMA calculations now receive the latest **1,200** closed candles, not a 600-bar shortcut.

## Reproduce

On Python 3.11+:

```bash
python -m unittest discover -s tests -p test_spot_backtest.py -v
python scripts/spot_backtest.py \
  --fetch-coinbase --start 2024-01-01 --end 2026-01-01 \
  --save-csv research-output/coinbase_btcusd_hourly_2024_2025.csv \
  --report research-output/spot_validation.json \
  --fees 15 40 60 --slippage-bps 3
```

This fetches Coinbase Exchange BTC-USD **spot**, not BTCUSDT futures. No authentication or trading credentials are used. An Actions workflow runs the same script and publishes its input data and report as a workflow artifact. Historical requests use 288-hour windows to stay within the 300-candle API limit. The script checks duplicate, invalid and nonfinal candles, retries a missing hour with a narrow API request, and **reports any remaining gaps**. It never interpolates missing price data.

To reuse saved data:

```bash
python scripts/spot_backtest.py --csv research-output/coinbase_btcusd_hourly_2024_2025.csv \
  --report research-output/replay_from_csv.json --fees 15 40 60
```

## Study specification (fixed before observing results)

- History: 2024-01-01 00:00 UTC to 2026-01-01 00:00 UTC, exclusive end.
- Signals use a *closed* hourly bar, and paper entry/exit uses the **next hour's open** as the execution-price proxy.
- All evaluation windows have 1,200-bar past-only indicator history.
- 2024 is development/reference history; **2025 is a separately reset holdout portfolio**. Any genuine Coinbase gap splits an evaluation period into multiple independent segments, each with its own $10,000 paper capital and a new 1,200-bar warmup. Never add segment returns as though they were one continuous portfolio. All-years reports are descriptive, not independent holdout.
- Existing runtime functions `step`, `buy`, `sell` and `indicators` are reused without changing source or model parameters. Fixed fees are temporarily injected into these functions **only inside the offline replay process**, then restored.
- Trade fee scenarios (per side): 15, 40, 60 basis points plus 3 basis points slippage per side. All fills are treated as *taker* fills; no unverified maker discounts or rebates.
- Paper capital $10,000, risk fraction 0.5%, maximum allocation 50%, spot long/cash only.
- Compare net returns to *fee-adjusted* BTC buy-and-hold and cash; report drawdown, closed-trade expectancy, profit factor, win rate, time in market, and number of executions.

## Important limitations

1. An hourly OHLCV bar does not reveal whether a trailing stop hit before or after other intra-hour events. The live bot also performs additional 15-minute quote checks not reproduced by this hourly dataset. **This is not verified execution parity.**
2. The paper bot may enter later than the next hour's open because GitHub Actions are not guaranteed to start on time. Historical next-open fills can be optimistic.
3. Real quoted spreads, order-book depth, fee tier, and actual order fills are absent from candle data. Cost scenarios must not be represented as measured spreads or account-specific costs.
4. The model risk budget excludes execution costs and possible adverse gaps. Nominal 0.5% trade risk can be exceeded.
5. The holdout is a **single calendar period** and not enough to prove a statistically significant positive expectation. The downloaded data showed a missing hour near October 25, 2025; segment-wise reporting explicitly preserves this discontinuity. No parameter search, trade-quality filter, or strategy-changer is included here.

## Sources

- Coinbase Exchange API candles documentation: https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles
- Coinbase Exchange fee table: https://help.coinbase.com/en/exchange/trading-and-funding/exchange-fees
- Coinbase trading rules: https://www.coinbase.com/legal/trading_rules

## Acceptance before strategy changes

Only compare strategy challengers after the reference replay, report metadata, data coverage and cost assumptions pass validation. Keep the production bot and the historical paper ledger immutable until a challenger has demonstrated net-of-cost out-of-sample improvement. Do not trade real funds based on this research.
