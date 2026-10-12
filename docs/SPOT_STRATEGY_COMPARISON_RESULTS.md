# Historical Coinbase Spot Comparison Results

Reproduced on 2026-10-12. Data from Coinbase Exchange BTC-USD spot. Paper-only simulation.
These are **descriptive historical results**, not independently unseen OOS or a live profit guarantee.

[Actions input datasets and machine report](https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/runs/38219198153) | [Replay script](../scripts/compare_spot_strategies.py) | [Methodology](SPOT_STRATEGY_COMPARISON.md)

## Under paper fee assumption: 15 bps taker + 3 bps slippage per side

| Independent interval | Days | Strategy | Net return | Daily-close max DD | Closed trades | Profit factor |
|---|---:|---|---:|---:|---:|---:|
| 2024-01-01 to 2025-01-01 (exclusive) | 366 | SMA200 daily (50% initial allocation) | +37.240% | 16.434% | 5 | 4.114 |
| | | Hourly Donchian 55 | -4.159% | 10.309% | 76 | 0.818 |
| | | Buy & Hold (100% BTC) | +119.962% | 26.232% | 0 closed | n/a |
| | | Cash | 0.000% | 0% | 0 | n/a |
| 2025-01-01 to 2025-10-25 (exclusive) | 297 | SMA200 daily (50% initial allocation) | -6.570% | 17.660% | 7 | 0.466 |
| | | Hourly Donchian 55 | -11.317% | 12.923% | 69 | 0.438 |
| | | Buy & Hold (100% BTC) | +18.528% | 28.172% | 0 closed | n/a |
| | | Cash | 0.000% | 0% | 0 | n/a |
| 2025-12-15 to 2026-01-01 (exclusive) | 17 | SMA200 daily (50% initial allocation) | 0.000% | 0% | 0 | n/a |
| | | Hourly Donchian 55 | -1.667% | 1.667% | 3 | 0 |
| | | Buy & Hold (100% BTC) | -1.124% | 3.253% | 0 closed | n/a |
| | | Cash | 0.000% | 0% | 0 | n/a |

**Interpretation:** SMA200 improved over the old hourly model for these windows, but was still **loss-making in the main 2025 interval**. More importantly, its 2024 return was below 100%-invested Buy & Hold. SMA200 invests up to half of available cash on an entry, so its lower nominal market exposure is critical when interpreting these raw returns.

## Fee stress: 60 bps taker + 3 bps slippage per side

| Independent interval | SMA200 | Hourly | Buy & Hold | Cash |
|---|---:|---:|---:|---:|
| 2024 | +33.571% | -19.425% | +117.991% | 0.000% |
| 2025 Jan-Oct (complete days only) | -9.886% | -26.863% | +17.466% | 0.000% |
| 2025 Dec 15-31 | 0.000% | -2.608% | -2.010% | 0.000% |

40-bps scenarios are also available in the generated comparison.json.

## Data integrity and exact reproducibility

- History: hourly 2023-11-01 through 2025-12-31, daily 2023-01-01 through 2025-12-31.
- Coinbase hourly BTC-USD gap: 2025-10-25 16:00 to 21:00 UTC, five missing bars. No price is interpolated.
- Period ends are exclusive UTC boundaries; surrounding day and a new 1,200-hour warmup excluded from all results after the gap.
- **791 full days** compared between independently fetched hourly and daily OHLC endpoints, with **zero observed open/close endpoint disagreement**.
- Hourly SHA256: `5f9d191ac3a8b4b6faf6331fe9f9db95dff2ce87ce9fb5df0f9092f124753c36`
- Daily SHA256: `179d71f34a1c5762cb82f0fab011ef6a7bd2fae25bfdf77abd3c90469c92bfb7`
- Original paper engines are called without changing their source implementations. Every strategy starts from an independent $10,000, valuation is daily-close and open holdings are priced as if liquidated after fees/slippage at segment end.

## Unresolved constraints

- Hourly 15-minute live ticker stop checks cannot be recreated from hourly candles.
- Next-period opening prices are historical fill **proxies**, not observed orders.
- The actual production hourly market-data window may differ from the 1,200-candle replay window.
- Maker/taker account fee tier and spreads are not observed.
- The 2024 and 2025 periods were used to choose/inspect strategies. No fresh independent holdout remains in this dataset.
- Neither model should replace the other based only on past returns. Keep both in independent forward paper execution and evaluate timestamp-aligned, after-cost results over a sufficiently large future sample.

Coinbase Exchange fee schedule: https://help.coinbase.com/en/exchange/trading-and-funding/exchange-fees
