# Rule-based BTC Spot Strategy: Research & Implementation

## Evidence and selection

**Objective:** replace model-dependent one-hour direction forecasts with a verifiable, no-ML spot paper strategy.

Research branches reviewed:
- **Simple technical rules and breakouts:** Deprez & Frömmel (2024), *International Review of Economics & Finance*, tests 75,360 technical rules accounting for costs, data mining and out-of-sample performance. Selected combinations can beat buy-and-hold on risk/return, but no single configuration is universally optimal. https://doi.org/10.1016/j.iref.2024.05.003
- **Trend following:** Rozario et al. (2020), historical evidence for cryptocurrencies, but historical walk-forward results should not be treated as live returns. https://arxiv.org/abs/2009.12155
- **Counter-evidence against unconditional momentum:** Grobys et al. (2025), *Cryptocurrency momentum has (not) its moments*, documents severe momentum crashes and statistically weak results on a 30-coin, 2016-2023 sample; some volatility management helps, but this is not BTC-only hourly validation. https://doi.org/10.1007/s11408-025-00474-9
- **Other risk-managed momentum findings:** *Cryptocurrency market risk-managed momentum strategies* (2025), *Finance Research Letters*, finds improvements to mean returns and Sharpe, but this is a portfolio-level weekly study and does not establish our specific hourly system. https://doi.org/10.1016/j.frl.2025.107879
- **Trading-cost counter-evidence:** technical profitability weakens once friction is included. https://www.sciencedirect.com/science/article/pii/S1042443122000816
- **Funding-rate carry:** Pindza (2026) decomposes venue, spread, hedge, liquidation and execution risks. This requires perpetuals and is operationally unrelated to our spot-price-only system. https://doi.org/10.1007/s42521-026-00213-3
- **Backtest overfitting:** Bailey et al. demonstrate that testing large parameter grids leads to spurious historical winners. https://doi.org/10.21314/JCF.2016.322
- **Fees:** Coinbase states maker/taker fees vary by account and product. Our 15-bps fee plus 3-bps slippage are configurable-code assumptions, **not verified exchange pricing**. https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/advanced-trade-fees

**Selection:** long-or-cash trend-following breakout was chosen as the most straightforward first hypothesis to test under the existing hourly data workflow. It has not been demonstrated to be the highest-return strategy for this repository.

## Exact fixed rules

Use only closed, contiguous 1-hour candles for BTC-USD from Coinbase. For completed candle t:

1. EMA(200) calculated on close prices up to and including t.
2. Entry channel is max(high[t-55:t-1]), excluding the current candle.
3. ATR(14) is the average of 14 true ranges ending at t.
4. Buy when close[t] > entry channel AND close[t] > EMA(200) AND 0.002 <= ATR(14)/close[t] <= 0.06.
5. Simulate entry at the fresh quote after candle close. The backtest instead uses the next candle's open.
6. Buy size = min(0.5% of available paper cash / (3 x ATR), 50% of paper cash / entry fill, cash affordability after entry fee).
7. Initial stop = entry fill - 3 x ATR. If a full subsequent bar trades below the stored stop, paper stop fill is min(bar open, stop), less slippage; this is an approximation, not an exchange order.
8. If no stop, sell at fresh quote when close[t] < min(low[t-20:t-1]) OR close[t] < EMA(200).
9. Otherwise, trailing stop = max(previous stop, close[t] - 3 x ATR). It is updated **after** analyzing the completed bar and takes effect on future eligible bars.

We simulate fees as 0.15% and slippage as 0.03% per side. The quote, candles and decisions come from the same spot product to avoid a futures/spot execution mismatch.

## Data correctness and limitations

- Never score a nonfinal candle, a stale quote, malformed OHLC or discontinuous candles.
- On missed processing time, stop processing and require manual reconciliation. The rule-state branch does **not** reuse old model-state, forecast-state or adaptive-state.
- Repeated invocations within the same completed candle do not open another trade. The previous ledger remains append-only for completed trades.
- Stop monitoring is hourly rather than a real resting stop order. Intrabar order and outage behavior **can differ substantially**.
- Coinbase Exchange public historical endpoint returns at most 300 candle buckets per request; requests are time-windowed to 290 hours.
- There is no exchange authorization and absolutely no live order execution.
- Old machine learning code is retained for audit but is outside the workflows. The legacy retrain workflow may still be invoked manually.

## Verification gate before real money

1. Use at least multiple bull, bear and sideways regimes on clean real BTC-USD candles, without optimizing parameters on held-out data.
2. Compare net return and maximum drawdown against buy-and-hold and staying in cash, with realistic market-specific fees and fill spreads.
3. Verify numerical robustness under doubled transaction costs, delayed execution, outages and gaps. A surviving edge must not rely on one short historical segment.
4. Run forward paper trading for a meaningful sample. Compare published and replayed trades; do not rewrite old trade outcomes.
5. Require a separate review to introduce real execution, exchange-specific minimum sizes, actual stops, credentials and loss limits.

**Current evidence status:** local offline unit tests and synthetic-data smoke test only. Neither exchange-connected end-to-end workflow nor real-data out-of-sample profitability has been independently verified.
