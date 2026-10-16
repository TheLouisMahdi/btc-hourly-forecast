<div align="center">

<img src="docs/assets/candlestick-loop.svg" alt="Bitcoin candlesticks" width="100%" />

# BTC Rule-Based Spot Paper Trader

**Deterministic BTC-USD research. No model training, AI inference or exchange orders.**

[Live paper dashboard](https://thelouismahdi.github.io/btc-hourly-forecast/) | [Methodology](docs/RULE_BASED_STRATEGY.md) | [Legacy AI documentation](docs/LEGACY_AI_README.md)

![Version](https://img.shields.io/badge/version-5.6.0-6f9b91) ![Mode](https://img.shields.io/badge/mode-PAPER_ONLY-9fbcaa)

</div>

## Active architecture

The **hourly GitHub workflow** runs only scripts/rule_based.py, and its **dashboard workflow** runs only scripts/render_rules_dashboard.py. Both use the separate rule-state branch. Old AI training modules, historical model-state and forecast-state records remain archived for comparison, **not imported by the new runtime**. The retrain workflow is legacy manual tooling only.

This is a fixed-hypothesis system, **not an experimentally proven best Bitcoin strategy**. No live or out-of-sample profitable performance is claimed. No trading API credentials are required.

## Experimental RSI(20) regime / pullback shadow

A new **independent hourly spot paper hypothesis** uses the most recently
completed **4h Wilder RSI(20) > 70** as a permission gate and final **1h**
candles for pullback or breakout entries. It caps exposure, limits entries
per regime and uses a dual simulated stop / 24-hour holding limit.
This is **not an experimentally proven improvement**, and it is not a
replacement for the live Donchian paper ledger. No leverage or live orders.

- Script: `scripts/regime_shadow.py`; workflow:
  `.github/workflows/regime-shadow.yml` (separate `regime-shadow-state`).
- Backtest: `python scripts/backtest_regime.py hourly_ohlcv.csv --oos-fraction 0.30`
- Method and limitations: [RSI regime spot shadow](docs/REGIME_SHADOW.md).
- Reference credit: [denniskid2520/btcusdt-perp-live](https://github.com/denniskid2520/btcusdt-perp-live)
  informed the **high-level idea** of 4h regime gating plus 1h pullback/breakout
  execution. No source code was copied. That project's reported leveraged
  futures performance does not apply to this separately implemented spot system.

## Parallel SMA200 daily research

A separate paper-only daily SMA200 strategy runs on the **last completed UTC daily
Coinbase BTC-USD candle**. It holds BTC (maximum 50% initial paper cash allocation)
when daily close is above SMA(200), otherwise cash. It simulates 15 bps fees and
3 bps slippage per side, with no real exchange orders and **no intraday stop**.

The daily state is kept on `sma-shadow-state`; the existing hourly `rule-state`
and all archived ML ledgers are unchanged. The homepage compares independent
paper portfolios and displays their separate as-of times. This is a shadow
hypothesis, **not a proven profitable system**.

Scheduled entry: `.github/workflows/sma-shadow.yml` at 00:33 UTC daily
(subject to GitHub scheduling delays), also triggered on a merge to main.
See [SMA200 shadow contract](docs/SMA200_SHADOW.md).

## Strategy

* Product: Coinbase BTC-USD spot, long or cash, **never short or leverage**.
* Only finalized **hourly candles** are accepted.
* **Buy** if the closed price breaks the prior 55-candle Donchian high and exceeds EMA(200), with ATR(14)/price between 0.2% and 6%.
* **Exit** for an ATR-based simulated stop, a close below the prior 20-candle low or a close below EMA(200).
* **Position sizing**: at most 0.5% of available paper cash at the initial stop distance and at most 50% of paper cash allocated. Stop distance is 3 × ATR(14).
* Default paper assumptions: 15 bps trading fee and 3 bps slippage **per side**, not a claim about Coinbase's actual fee tier.
* Malformed or stale market data causes **FAIL_SAFE**. After missed cycles, the paper engine safely resyncs while flat or exits an open simulated position at the fresh quote, without inventing historical fills.
* These stops exist **only in the simulation** and are evaluated using completed candles. They are **not broker stop orders**.

## Run locally

Python 3.11 or newer; standard library only for the new runtime.

```bash
python -m unittest discover -s tests -p 'test_rule_based.py' -v
python scripts/rule_based.py --state-dir .rule_state
python scripts/render_rules_dashboard.py
```

The first command performs offline tests. The second requests public Coinbase market data and writes isolated paper-state JSON. The third builds site/index.html.

The scheduled workflow is .github/workflows/forecast.yml, running every 15 minutes and on main-branch pushes; deployment uses .github/workflows/dashboard.yml. For broken, discontinuous or stale market data, the workflow enters FAIL_SAFE until fresh valid data is available. After a missed execution interval with valid market data, the engine closes an open simulated position at the current quote or resyncs a flat state. It never fabricates retrospective fills.

## Backtest

```bash
python scripts/backtest_rules.py hourly_ohlcv.csv --out-of-sample-fraction 0.30
```


Required CSV columns: time (UTC epoch seconds), open, high, low, close, volume. Candles must be hourly, sorted, complete and gap-free. Signals use closed bars and **next bar's opening price**, with assumed transaction costs. The out-of-sample segment uses the most recent 30% of candles. Avoid searching dozens of parameters and then reporting the best historical result.

**Decision to deploy with real capital must be separate from this PR.** Live fills, actual fees, data resiliency, market regimes and long out-of-sample outcomes require independent validation.
