# SMA200 Daily Shadow Strategy

## Scope

This runs **alongside** the existing 1-hour Donchian/EMA/ATR paper trader. It
does not alter or read the hourly bot's cash, trades, last candle or positions.
There is **no exchange authorization and no real order execution**.

## Daily decision

1. Download Coinbase Exchange BTC-USD spot daily OHLCV from its public candles
   API, maximum 260 requested UTC days (below the endpoint's 300-bucket cap).
2. Require the **latest completed UTC day** and 200 consecutive completed
   daily candles. Ignore the current, unfinished day.
3. Define SMA(200) as the arithmetic mean of the last 200 daily closes.
4. If the latest completed daily **close is strictly above SMA(200)**, target
   long BTC. Otherwise target cash. No short-selling, leverage, ML or stop order.
5. Use the fresh public ticker as the next available simulated execution price.
   The quote is validated for freshness. Note that this is **not bid/ask**;
   actual orders and outcomes would differ.
6. When going long, allocate at most **50% of remaining paper cash**.
   Model 15 basis points taker fee plus 3 basis points slippage **per side**.
   Close the full shadow BTC position when the signal becomes cash. Cash left
   idle does not earn interest.
7. Process a UTC candle at most once, persist ledger and daily decisions to
   branch `sma-shadow-state`. Missing workflow days are **not replayed as
   historical fills**; their count is included in the daily report.

The initial portfolio is **$10,000 simulated cash**. The strategy can lose
money, including extended drawdowns. It has no intraday stop. The fee inputs
are assumptions, **not verified Coinbase account pricing**.

## Execution

`.github/workflows/sma-shadow.yml` runs daily at **00:33 UTC**, after the
daily candle close, and upon push to `main`. There may be delays or skipped
scheduled runs in GitHub Actions. The standalone runtime fails closed when
market data or ticker is stale or invalid, and does not modify the saved
trading ledger on failure.

```bash
python -m unittest discover -s tests -p test_sma200_shadow.py -v
python scripts/sma200_shadow.py --state-dir .sma_shadow
python scripts/render_rules_dashboard.py
```

Only the SMA script requests public Coinbase data. It does not import the old
AI system. Its output JSON files are:

- `state.json`: cash, BTC quantity, entry and execution state
- `latest.json`: most recent daily SMA and LONG/CASH signal
- `trades.json`: closed shadow trades, including simulated fees and net P&L
- `history.json`: daily decisions, equity, signal and missing-day counts
- `candles.json`: latest completed daily OHLCV for auditing

The existing GitHub Pages dashboard renders two **independent** paper
portfolios and includes `sma200_latest.json`, `sma200_history.json`,
and `sma200_trades.json`. Equity values are not directly comparable until
both portfolios have the same start date and measurement horizon.

## Release and validation

- Run seven offline SMA200 tests, full repository tests and a read-only
  Coinbase data/quote smoke test before publishing.
- Promote only the separate paper shadow, not a replacement of the baseline.
- Compare after a substantial untouched forward period and with realistic
  costs; historical historical returns never guarantee future profitability.
- Full SMA backtesting with the same live signal, execution assumptions,
  side-by-side return/drawdown measures and suitable held-out data remains a
  separate research gate before changing the production hourly strategy.

Sources: Coinbase Exchange candles endpoint
https://docs.cdp.coinbase.com/exchange/reference/exchangerestapi_getproductcandles
and GitHub Actions schedule guidance
https://docs.github.com/en/actions/reference/events-that-trigger-workflows#schedule
