# Reproducible Coinbase BTC-USD Spot Strategy Comparison

Status: paper-only historical research. Does not touch running SMA200 or Donchian ledgers. No broker orders.

## Four strategies, identical evaluation windows

- SMA200_DAILY_50PCT: calls the real [daily shadow strategy](../scripts/sma200_shadow.py) function (step) on 200 completed daily bars. A new LONG entry spends approximately **50% of currently available paper cash**, the other half remains in cash. No leverage, stops or daily rebalancing.
- DONCHIAN55_HOURLY: calls the live [hourly trading strategy](../scripts/rule_based.py) function (step), on the most recent **1,200 previous completed hourly candles**. Retains the original allocation cap, 3xATR stop simulation, trend exit and risk fraction. Its real GitHub 15-minute quote-based stop checks cannot be recovered from hourly OHLCV.
- BUY_HOLD_100PCT: buys near **100% of initial notional**, paying one simulated entry fee plus slippage; marks final NAV after a hypothetical exit fee plus slippage.
- CASH_100PCT: zero return, no interest.

All four start from an independent **$10,000** paper portfolio for each segment. SMA200's initial 50% exposure means raw returns are **not normalized to equal invested BTC notional** against 100% Buy & Hold. Include drawdown and exposure when assessing these choices. The hourly strategy can invest less than its nominal cap due to ATR risk limits.

## Historical data and methodology

- Product: Coinbase Exchange **BTC-USD Spot**, public OHLCV APIs.
- Hourly history: **November 1, 2023 to December 31, 2025**, for SMA and hourly backtest coverage and hourly causal warmup.
- Daily history: **January 1, 2023 to December 31, 2025** for a rolling 200-day SMA.
- Evaluation: calendar 2024 and 2025; the remaining 2025 segments are separated by missing Coinbase hourly data. A subsequent comparison begins only when **1,200 continuous hourly bars** are once more available. Period boundaries are UTC day boundaries and no capital is carried across gaps.
- Entry and exit prices: the **next period's opening price**, with fixed fee 15, 40 or 60 basis points per side, and 3 basis points slippage per side.
- Mark-to-market: at every **daily close** for each strategy, so maximum daily drawdowns share the same sampling frequency. For open BTC positions, final NAV includes a hypothetical liquidation fee and slippage.
- The daily provider candles and hourly provider candles are **fetched independently** and input SHA256 hashes are recorded. Coinciding daily/hourly open/close endpoints are audited for disagreement; missing candles are retried, reported and **never fabricated**.
- The 2024-2025 samples are now **observed**. Their results are descriptive and must not be called an independent untouched holdout after strategies were examined on these years.

## Run offline tests and download/replay

```bash
python -m unittest discover -s tests -p test_spot_comparison.py -v
python scripts/compare_spot_strategies.py --fetch-coinbase \
    --output-dir research-output/comparison \
    --fees 15 40 60 --slippage-bps 3

# Offline replay using the saved identical Coinbase candles:
python scripts/compare_spot_strategies.py \
    --csv-dir research-output/comparison \
    --output-dir research-output/replay \
    --fees 15 40 60 --slippage-bps 3
```

The GitHub Actions workflow `.github/workflows/spot-comparison.yml` publishes:

- `coinbase_hourly.csv`, `coinbase_daily.csv`: exact research candle inputs
- `comparison.json`: program-readable comparison and input data fingerprints
- `comparison.md`: cost-sensitive results table

These are workflow artifacts and are **not** uploaded to the active trading state branches.

## Interpretation constraints

1. A positive historical return is not proof of positive future expectancy. A handful of daily SMA200 round trips does not offer high statistical precision.
2. SMA200 trading in GitHub Actions happens some time **after** daily close. Next-open historical execution is a proxy and could be optimistic.
3. Hourly replay uses completed hourly OHLC and next-hour OPEN. It cannot reproduce every 15-minute live quote check or timing of stop triggers.
4. Taker fee assumptions require validation against actual eligible account tiers. Coinbase Exchange public fee table: https://help.coinbase.com/en/exchange/trading-and-funding/exchange-fees
5. No borrow fees, taxes, cash yield, partial fills, realistic order book, position capacity or operational outage loss are modeled. Liquidity/slippage and market access conditions can change.
6. Daily OHLC and hourly OHLC may not have identical endpoints despite using the same market. The comparison records this disagreement rather than silently equating them.
7. Do not combine segment returns as one equity curve. Every segment starts with a new $10,000 simulation to eliminate unobserved state after a data gap.

## Decision gate

Keep both paper bots in their **current** independent execution paths. Before changing any strategy settings, reproduce artifacts, evaluate fees, reconcile signal parity on the same data, and observe a new forward period. An edge is unverified until multiple independent periods are profitable after realistic costs and risk.
