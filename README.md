<div align="center">

<img src="docs/assets/candlestick-loop.svg" alt="Animated BTC breakout chart" width="100%" />

<br />

<a href="https://thelouismahdi.github.io/btc-hourly-forecast/">
  <img src="https://img.shields.io/badge/OPEN_LIVE_DASHBOARD-GITHUB_PAGES-6f9b91?style=for-the-badge&logo=githubpages&logoColor=ffffff" alt="Open live dashboard" />
</a>

<br />
<br />

<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/hourly_forecast.yml">
  <img src="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/hourly_forecast.yml/badge.svg?branch=main" alt="Hourly BTC forecast status" />
</a>
<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/weekly_retrain.yml">
  <img src="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/weekly_retrain.yml/badge.svg?branch=main" alt="Weekly BTC model retraining status" />
</a>
<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/pages_dashboard.yml">
  <img src="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/pages_dashboard.yml/badge.svg?branch=main" alt="Dashboard deployment status" />
</a>

<br />
<br />

<img src="https://img.shields.io/badge/version-5.3.0-8f8ab8?style=flat-square" alt="Version 5.3.0" />
<img src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 or newer" />
<img src="https://img.shields.io/badge/training-no_random_sampling-6f9b91?style=flat-square" alt="No random sampling" />
<img src="https://img.shields.io/badge/mode-paper_trading_only-c99078?style=flat-square" alt="Paper trading only" />

</div>

# BTC Deterministic Directional Breakout Forecast

Version 5.3 separates two different trading problems that must not share one generic event model:

- `RESISTANCE_BREAKOUT_LONG`: a real close crossing above confirmed resistance;
- `SUPPORT_BREAKDOWN_SHORT`: a real close crossing below confirmed support.

Long and Short use separate candidate formulas, labels, targets, invalidation rules, probability thresholds, model heads and qualification results.

The public dashboard still publishes an immutable `NEXT_CLOSED_1H_CANDLE` direction and close interval. The trading layer independently evaluates breakout continuation over `3h`, `6h` and `12h` horizons.

## Non-random training contract

```text
sampling: NONE
shuffle: false
synthetic events: 0
oversampling: false
undersampling: false
split: chronological expanding window
```

Training is allowed to proceed only after the deterministic event inventory contains at least:

- **2,000 unique resistance-breakout Long events**;
- **2,000 unique support-breakdown Short events**.

The inventory gate also requires coverage across at least six calendar years, 24 quarters, four structure scales, three volatility buckets, three market regimes and 48 diversity groups per direction.

Events are never duplicated to meet the target. When the real inventory is insufficient, training fails with an explicit inventory error.

## Historical data

The configured batch retraining window is ten years of real hourly BTC candles, targeting at least 80,000 chronological rows.

Provider priority:

1. Coinbase BTC-USD spot;
2. Binance BTC-USDT spot;
3. Binance BTC-USDT futures;
4. OKX BTC-USDT swap.

The selected provider is stored in the model artifact and runtime state. Providers are fallbacks, not mixed rows inside one training run.

News history remains limited to the recent configured window. Directional breakout heads exclude news fields so missing old news cannot become an artificial regime indicator.

## Deterministic event mining

The miner evaluates confirmed support and resistance across these hourly scales:

```text
24h · 48h · 96h · 168h · 336h · 720h
```

A candidate requires a real close-to-close crossing. Remaining above resistance or below support after an earlier break is not counted as a new event.

Each event stores:

```text
unique event ID
open time
Long or Short direction
breakout source and scale
breakout level
structural invalidation level
ATR-normalized crossing distance
level touches and age
line slope and fit quality
event score
market regime and volatility context
diversity key
```

Near-duplicate events are removed deterministically using time separation and ATR-normalized level similarity.

## Direction-specific formulas

### Resistance breakout Long

The Long candidate formula emphasizes:

- resistance touch history;
- positive candle-body strength;
- close location near the upper candle range;
- resistance crossing distance;
- relative volume;
- long-term upward structure;
- resistance-line quality;
- optional upper triangle-boundary quality.

Long invalidation is placed below the broken level. Long target distance and required hold ratio are configured separately for every trade horizon.

### Support breakdown Short

The Short candidate formula independently emphasizes:

- support touch history;
- negative candle-body strength;
- close location near the lower candle range;
- support crossing distance;
- relative volume;
- long-term downward structure;
- support-line quality;
- optional lower triangle-boundary quality.

Short invalidation is placed above the broken level. Short targets, hold requirements and probability thresholds are intentionally different from Long settings.

## Labels

Every real event is evaluated from the open of the next hourly candle.

For each `3h`, `6h` and `12h` trade horizon, the path is classified using the direction-specific target and structural invalidation:

- `SUCCESS`: target is reached before invalidation, the final close remains beyond the broken level and the required hold ratio is satisfied;
- `FALSE_BREAKOUT`: invalidation is hit first or the final close re-enters the broken structure;
- `NEUTRAL`: neither success nor a confirmed false breakout occurs;
- `TRADEABLE`: success remains positive after execution costs and the configured profit buffer.

The model also learns event-aligned return, maximum favorable excursion, maximum adverse excursion and level-hold ratio.

## Model architecture

Each horizon contains:

- one general next-close direction classifier;
- one general close-return regressor;
- one independent Long success classifier;
- one independent Long tradeability classifier;
- one independent Long return regressor;
- one independent Short success classifier;
- one independent Short tradeability classifier;
- one independent Short return regressor.

The batch artifact uses:

```text
schema_version: 5
model_id prefix: directional-breakout-hourly-
```

All older artifacts are rejected by the runtime.

## Validation

General next-close prediction uses chronological `TimeSeriesSplit` validation.

Long and Short event models use independent expanding-window Walk-Forward validation with a time embargo. Every Out-of-Fold event is evaluated once, after all training events used for that fold.

Qualification is direction-specific. Long can qualify without Short, and Short can qualify without Long. A direction-horizon pair must pass:

- minimum chronological OOF event count;
- success AUC;
- tradeability AUC;
- success calibration error;
- tradeability calibration error;
- minimum selected OOF trades;
- minimum OOF hit rate;
- positive stress-adjusted net expectancy;
- minimum positive-fold fraction.

A qualified Long model never grants permission to a Short trade, or vice versa.

## Risk and decision gates

A trade requires a fresh structural event and a qualified model for that same direction and selected horizon.

Possible blockers include:

```text
MODEL_NOT_QUALIFIED
SELECTED_DIRECTION_NOT_QUALIFIED
NO_NEW_STRUCTURE_BREAKOUT
WEAK_BREAKOUT_STRUCTURE
BREAKOUT_LEVEL_UNAVAILABLE
INVALIDATION_LEVEL_UNAVAILABLE
LOW_BREAKOUT_SUCCESS_PROBABILITY
LOW_TRADEABILITY_PROBABILITY
BREAKOUT_HORIZON_DISAGREEMENT
INSUFFICIENT_STRESS_NET_EDGE
```

The preferred stop is anchored to the event invalidation level. ATR risk is only a bounded fallback.

## Public forecast contract

For a source candle with open time `T`:

| Field | Time |
|---|---|
| Source candle opens | `T` |
| Forecast is created after source close | `T + 1h` |
| Target candle opens | `T + 1h` |
| Target candle closes | `T + 2h` |
| Resolution becomes eligible | target close plus settlement delay |

The frozen direction result is one of:

```text
PENDING
DIRECTION_CORRECT
DIRECTION_WRONG
LEGACY_NOT_SCORED
```

Interval scoring is separate:

```text
IN_RANGE
OUT_OF_RANGE
```

A wide interval cannot turn a wrong direction into a correct prediction.

## Adaptive status

The general next-close price learner remains available. The previous generic online trade learner is disabled in version 5 because it does not contain independent Long and Short heads. It must not modify directional breakout probabilities until a direction-specific online architecture passes its own prequential validation.

## Automation

The weekly workflow:

1. fetches the configured ten-year hourly market history;
2. fetches the bounded news-history window;
3. calculates causal market structure;
4. mines unique Long and Short events deterministically;
5. enforces the 2,000-per-direction diversity gate;
6. creates direction-specific path labels;
7. trains separate Long and Short heads;
8. runs chronological OOF validation;
9. publishes a schema-v5 artifact only after successful completion.

The hourly workflow restores only the latest compatible artifact, refreshes recent closed candles, creates one immutable next-candle forecast and evaluates a trade only when the current event direction is independently qualified.

## Local setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Fetch and train from the configured real history:

```bash
btc-regime fetch --days 3650
btc-regime news --historical --days 365
btc-regime news
btc-regime train
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Repository layout

```text
config/default.yaml
scripts/github_weekly_retrain.py
scripts/github_hourly_forecast.py
src/btc_ema_trader/
  market.py                 Multi-year deterministic market ingestion
  market_structure.py       Causal pivots, levels and triangles
  directional_events.py     Long and Short mining, labels and inventory gate
  contract_features.py      Shared training/runtime feature contract
  structure_training.py     Separate chronological Long and Short training
  model.py                  Schema-v5 directional model bundle
  strategy.py               Direction-specific qualification and risk gates
  forecast_contract.py      Immutable next-candle public contract
tests/
```

## Status

Version 5.3 defines the required training contract and architecture. It does not claim profitability. A model is usable only after the GitHub retraining workflow confirms the real event inventory, completes schema-v5 Walk-Forward evaluation and publishes direction-specific qualification results.

This repository is research software and is not financial advice.

---

© 2026 Mahdi Ghahremani · ID: [TheLouisMahdi](https://github.com/TheLouisMahdi)
