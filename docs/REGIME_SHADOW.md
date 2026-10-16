# Experimental 4h RSI regime / 1h pullback and breakout BTC spot shadow

## Scope and provenance

This is an **independent experimental strategy**, not a replacement for the existing
Donchian baseline or daily SMA200 shadow. It uses public Coinbase **BTC-USD spot**
candles and a simulated long-or-cash account. No futures, leverage, funding,
shorts, trading credentials or exchange orders are involved.

The architecture was **inspired at a conceptual level** by
[denniskid2520/btcusdt-perp-live](https://github.com/denniskid2520/btcusdt-perp-live):
higher-timeframe RSI regime gating, lower-timeframe pullback/breakout entries,
a two-layer exit discipline and bounded holding time. Its public performance claims
were **not reproduced or transferred** to this strategy. The referenced project
targets leveraged BTC perpetual futures, so its reported returns, fills,
risk assumptions and parameters are not comparable to Coinbase spot.
**No implementation code was copied from the reference project.**

## Exact causal rules

1. Retrieve contiguous final **1h BTC-USD Coinbase spot** candles. Any missing,
   invalid or stale market information puts the runtime in FAIL_SAFE, without
   placing or inventing orders. Use fresh public ticker prices for paper fills.
2. Aggregate UTC-aligned groups of four *completed* hourly candles. Discard
   incomplete groups. Calculate Wilder RSI(20) from these completed 4h closes.
3. Allow new long entries only when **4h RSI(20) > 70**.
4. Compare the last completed 1h close with the maximum **high of the prior 12
   hours**, excluding the current candle:
   - **BREAKOUT:** close exceeds that prior high by 0.25%.
   - **PULLBACK:** prior close was at least 0.75% below that high, and the
     just-closed candle is bullish, improves on the previous close and remains
     below the prior high. Breakout has priority if both conditions were possible.
5. A regime is a consecutive run of completed 4h RSI permission. Enter at most
   **3 sequential times** during one active regime, and hold at most one BTC
   position. A new regime resets the entry counter. Missed hours instead force
   a safe resync or a simulated exit, with no reconstruction of missed fills.
6. At entry, allocate at most **50% of available cash** and size the position so
   the 2.5% planned emergency-stop distance risks at most **0.5% of available
   paper cash** before costs. This is not guaranteed maximum realized loss:
   execution gaps can make a simulated stop fill significantly worse.
7. For an open trade, process exits conservatively in order:
   - Fresh quote at or below entry minus 2.5%: simulated emergency exit.
   - Completed eligible 1h bar whose low breached the 2.5% emergency stop:
     use the worse of the bar open and stop, then apply simulated slippage.
   - Completed candle close at or below entry minus 1.25%: exit at the next
     observed quote.
   - After **24 full hourly candle advances**, exit at the next quote.
   Quote checks can occur at subsequent 15-minute workflow polls, but no real
   resting stop exists. Candle intrabar ordering cannot be established from OHLC.
8. Charge **15 bps fee and 3 bps slippage per side** (assumptions, not an
   exchange fee quote), and track closed trades and portfolio equity.

All decisions use data known no later than the signal candle. There is no
ML training, hyperparameter search, short exposure or autonomous live execution.

## Runtime isolation

- Code: `scripts/regime_shadow.py`.
- Workflow: `.github/workflows/regime-shadow.yml`.
- Persistent branch: `regime-shadow-state`, independent of `rule-state` and
  `sma-shadow-state`.
- Local files: `state.json`, `candles.json`, `history.json`, `latest.json`,
  `trades.json` in `.regime_shadow/`.
- Existing GitHub Pages dashboard displays the regimes as separate paper accounts.
  Never compare their dollar equity without matching start times and exposures.

Commands:

```bash
python -m unittest discover -s tests -p 'test_regime_shadow.py' -v
python scripts/regime_shadow.py --state-dir .regime_shadow
python scripts/backtest_regime.py coinbase_hourly.csv --oos-fraction 0.30
```

The replay uses the **same strategy state machine**, closed hourly signals and
the subsequent hour's opening price for fills. It resets cash at the OOS
boundary, with prior candles available only as a *warmup*, not as future data.
CSV columns are `time,open,high,low,close,volume`, with UTC epoch seconds.
Because the tested rule ideas were considered using historical market results,
an ordinary chronological split **does not make this strategy an untouched
external validation**. Never claim live, forward or risk-adjusted outperformance
without independently reproducing results across regimes and exchange costs.

## Promotion criteria

Keep the original hourly Donchian and daily SMA200 strategies untouched while
this new hypothesis is assessed. Compare same-dated equity curves and matched
exposure on real candles; report net profit factor, win rate, trade count,
drawdown, fees, slippage sensitivity, market regimes and forward results.
The referenced 69.3% win rate belongs to another leveraged system and is
**not a predicted or observed win rate for this implementation**.
