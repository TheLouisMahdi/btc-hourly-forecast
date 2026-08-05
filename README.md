<div align="center">

<img src="docs/assets/candlestick-loop.svg" alt="Animated BTC breakout chart" width="100%" />

<br />

<a href="https://thelouismahdi.github.io/btc-hourly-forecast/">
  <img src="https://img.shields.io/badge/OPEN_LIVE_DASHBOARD-GITHUB_PAGES-6f9b91?style=for-the-badge&logo=githubpages&logoColor=ffffff" alt="Open live dashboard" />
</a>

<br /><br />

<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/hourly_forecast.yml">
  <img src="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/hourly_forecast.yml/badge.svg?branch=main" alt="Hourly BTC forecast status" />
</a>
<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/weekly_retrain.yml">
  <img src="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/weekly_retrain.yml/badge.svg?branch=main" alt="Weekly BTC model retraining status" />
</a>
<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/pages_dashboard.yml">
  <img src="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/pages_dashboard.yml/badge.svg?branch=main" alt="Dashboard deployment status" />
</a>
<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/quality.yml">
  <img src="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/quality.yml/badge.svg?branch=main" alt="Repository quality status" />
</a>

<br /><br />

<img src="https://img.shields.io/badge/version-5.4.0-8f8ab8?style=flat-square" alt="Version 5.4.0" />
<img src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 or newer" />
<img src="https://img.shields.io/badge/training-no_random_sampling-6f9b91?style=flat-square" alt="No random sampling" />
<img src="https://img.shields.io/badge/mode-paper_trading_only-c99078?style=flat-square" alt="Paper trading only" />

</div>

# BTC Adaptive Directional Breakout Trader

This repository is a research-grade, paper-only Bitcoin trading system built around two separate structural events:

- `RESISTANCE_BREAKOUT_LONG`: a confirmed close crossing above resistance;
- `SUPPORT_BREAKDOWN_SHORT`: a confirmed close crossing below support.

The primary output is a persistent LONG or SHORT paper position with an entry, adaptive target, stop-loss and maximum holding time. A secondary public contract forecasts the exact `NEXT_CLOSED_1H_CANDLE` and is scored only after that candle closes.

The project does not claim profitability and does not place live orders.

## One canonical product contract

### Primary: position lifecycle

A fresh structural event may create one paper position. The position remains open across hourly candles until one of these immutable outcomes is recorded:

```text
TARGET
STOP
TIME_EXIT_WIN
TIME_EXIT_LOSS
```

Each position stores:

```text
entry price
target price
initial and current stop
risk budget and notional
stress execution cost
expected value
maximum favorable and adverse excursion
realized net P/L
realized R-multiple
```

Only one active position is managed at a time. Closed outcomes are never recomputed from later prices.

### Secondary: exact next-close forecast

For a source candle with open time `T`:

| Event | Time |
|---|---|
| Source candle opens | `T` |
| Source candle closes | `T + 1h` |
| Forecast is created | after `T + 1h`, before the target closes |
| Target candle opens | `T + 1h` |
| Target candle closes | `T + 2h` |
| Resolution becomes eligible | target close plus settlement delay |

The direction result is one of:

```text
PENDING
DIRECTION_CORRECT
DIRECTION_WRONG
LEGACY_NOT_SCORED
```

Interval scoring is independent:

```text
IN_RANGE
OUT_OF_RANGE
```

A wide interval cannot turn a wrong direction into a correct result. Resolved records are immutable.

## Causal candle context

Every event is represented by exactly three closed candles:

```text
PREVIOUS_2
PREVIOUS_1
EVENT
```

The model receives OHLCV shape, candle body, full upper and lower shadows, close location, range, volume change and three-bar pressure features. Future candles are never predictors; they are used only to create labels and resolve positions.

## Deterministic event mining

Support and resistance are evaluated across:

```text
24h · 48h · 96h · 168h · 336h · 720h
```

A candidate requires a real close-to-close crossing. Remaining above an already broken resistance or below an already broken support is not a new event.

Each event records its unique ID, direction, source, scale, level, structural invalidation, ATR-normalized distance, touch count, age, line quality, market regime and diversity key. Near-duplicates are removed deterministically.

## Separate Long and Short models

Long and Short do not share one event head. Every trade horizon contains independent classifiers and return regressors for each direction.

Long logic emphasizes resistance quality, upward candle strength, upper close location, relative volume and upward structural alignment. Short logic separately emphasizes support quality, downward candle strength, lower close location, relative volume and downward structural alignment.

The compatible model artifact contract is:

```text
schema_version: 5
model_id prefix: directional-breakout-hourly-
```

Older model artifacts are rejected.

## Training contract

```text
sampling: NONE
shuffle: false
synthetic events: 0
oversampling: false
undersampling: false
split: chronological expanding window
```

Training requires at least **2,000 unique Long events** and **2,000 unique Short events**, plus multi-year, multi-quarter, multi-scale, volatility and regime diversity. Real events are never duplicated to satisfy the gate.

The configured history target is ten years of real hourly BTC candles. One provider is selected for each training run; rows from different exchanges are not mixed into one dataset.

## Validation and champion promotion

General prediction uses chronological time-series validation. Long and Short event heads use independent expanding-window Walk-Forward evaluation with an embargo.

A direction-horizon pair is evaluated using event count, AUC, calibration, selected-trade count, hit rate, stress-adjusted expectancy and positive-fold consistency. A challenger replaces the current champion only when the economic validation and sandwiched negative-memory validation agree.

Candidate diagnostics are retained even when promotion is rejected. Production-facing artifacts live on `model-state`, not on `main`.

## Aggressive paper mode

The GitHub runtime uses `AGGRESSIVE_PAPER` exploration. It may ignore selected soft blockers, including incomplete qualification or weak predicted economic edge, to collect paper outcomes. It never ignores hard conditions such as missing market data, stale quote protection, unsupported event type, absent breakout level, absent invalidation, duplicate event or an already active position.

The dashboard therefore separates:

- pipeline and data health;
- economic qualification;
- the current paper position;
- candidate blockers and ignored soft blockers.

An exploratory paper position is not evidence of a profitable or qualified production strategy.

## Adaptive learning and negative memory

The price learner receives weight only when its prequential Brier score, direction accuracy and return error improve over the batch model.

The trade learner updates only after a position resolves. It learns target probability, stop probability and realized R from the frozen entry feature vector, including the causal three-candle context.

The negative-memory layer uses a sandwiched design:

```text
front Bloom memory
learned boundary-risk model
backup Bloom memory
```

Bloom hits are treated as learned risk evidence. In the current paper configuration they adjust or report risk rather than silently changing historical outcomes.

## State isolation

Generated data is intentionally kept outside `main`:

| Branch | Purpose |
|---|---|
| `forecast-state` | latest record, forecast history and position ledger |
| `model-state` | promoted champion, reports and negative memory |
| `adaptive-state` | online learner states and summaries |
| `quality-state` | test and compile diagnostics |
| `training-diagnostics` | challenger training diagnostics |

`main` contains source code, configuration, tests and documentation only.

## Automation

The hourly workflow:

1. restores the champion and persistent state;
2. runs the full quality suite;
3. fetches the latest contiguous closed-candle segment;
4. resolves existing positions;
5. creates one new structural decision when eligible;
6. freezes the secondary next-close contract;
7. persists forecast, position and adaptive states.

The weekly workflow:

1. fetches long historical data;
2. segments real gaps without filling or interpolation;
3. mines deterministic Long and Short events;
4. trains independent direction heads;
5. performs chronological validation;
6. trains sandwiched negative memory;
7. promotes the challenger only when all promotion gates pass.

GitHub Pages is the only canonical dashboard implementation.

## Local setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Fetch and train:

```bash
btc-regime fetch --days 3650
btc-regime news --historical --days 365
btc-regime news
btc-regime train
```

Run one local cycle:

```bash
btc-regime cycle --force
```

Show the canonical dashboard URL:

```bash
btc-regime dashboard
```

Run validation:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts
```

## Repository layout

```text
config/default.yaml                   Canonical strategy and model configuration
scripts/github_weekly_retrain.py      Challenger training and promotion
scripts/github_structural_forecast.py Canonical GitHub hourly entry point
scripts/github_pages_dashboard.py     Public dashboard orchestration
src/btc_ema_trader/candle_context.py  Causal three-candle context
src/btc_ema_trader/directional_events.py
src/btc_ema_trader/market_structure_fast.py
src/btc_ema_trader/model.py
src/btc_ema_trader/negative_memory.py
src/btc_ema_trader/trade_lifecycle.py
src/btc_ema_trader/strict_forecast_contract.py
tests/
```

## Scope

This software is for research and paper trading only. Model qualification, a positive expected value and past paper performance do not guarantee future profit. No exchange credentials or live-order path are included.

---

© 2026 Mahdi Ghahremani · ID: [TheLouisMahdi](https://github.com/TheLouisMahdi)
