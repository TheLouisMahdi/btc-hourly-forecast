<div align="center">

<img src="docs/assets/candlestick-loop.svg" alt="Animated BTC structural breakout chart" width="100%" />

<br />

<a href="https://thelouismahdi.github.io/btc-hourly-forecast/">
  <img src="https://img.shields.io/badge/OPEN_LIVE_DASHBOARD-GITHUB_PAGES-6f9b91?style=for-the-badge&logo=githubpages&logoColor=ffffff" alt="Open live dashboard" />
</a>

<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/hourly_forecast.yml">
  <img src="https://img.shields.io/github/actions/workflow/status/TheLouisMahdi/btc-hourly-forecast/hourly_forecast.yml?branch=main&style=for-the-badge&label=Hourly%20Pipeline&labelColor=47746b" alt="Hourly pipeline status" />
</a>

<a href="https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/workflows/weekly_retrain.yml">
  <img src="https://img.shields.io/github/actions/workflow/status/TheLouisMahdi/btc-hourly-forecast/weekly_retrain.yml?branch=main&style=for-the-badge&label=Structural%20Retraining&labelColor=47746b" alt="Structural retraining status" />
</a>

<br />
<br />

<img src="https://img.shields.io/badge/version-4.0.0-8f8ab8?style=flat-square" alt="Version 4.0.0" />
<img src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 or newer" />
<img src="https://img.shields.io/badge/trade_setup-structural_breakout-6f9b91?style=flat-square" alt="Structural breakout setup" />
<img src="https://img.shields.io/badge/mode-paper_trading_only-c99078?style=flat-square" alt="Paper trading only" />

</div>

## Overview

**BTC Structural Breakout Forecast** is a research and paper-trading system built around one explicit market thesis:

- a confirmed resistance breakout can create a Long setup;
- a confirmed support breakdown can create a Short setup;
- dynamic long-term levels and triangle boundaries are treated as real market structure;
- machine learning estimates whether the break is likely to hold, fail or remain untradeable after costs.

Version 4.0 replaces the previous generic event engine. The trading layer no longer starts from volume impulses, broad momentum events or a direction guess alone. It starts from a causal structural break and lets the model accept or reject that setup.

The public dashboard still publishes one immutable `NEXT_CLOSED_1H_CANDLE` forecast after each fully closed hourly candle. That public direction forecast is separate from the multi-horizon structural trade decision.

## Core design

```text
Closed hourly candles
        ↓
Confirmed causal pivots
        ↓
Multi-scale static and dynamic levels
        ↓
Triangle and compression structure
        ↓
Confirmed resistance breakout or support breakdown
        ↓
Breakout-hold, false-breakout and net-tradeability models
        ↓
Qualified paper-trade decision
```

The system deliberately separates three questions:

1. **What is the next hourly close direction?**
2. **Did a valid structural breakout occur?**
3. **Is that breakout likely to hold and remain profitable after stressed costs?**

A correct answer to the first question does not automatically create a trade.

## Causal market structure

All structure is calculated without using future candles.

### Confirmed pivots

A pivot is available only after its right-side confirmation bars have closed. The original turning point may be earlier, but the model cannot use it until confirmation time.

This avoids retrospective chart drawing that would look accurate historically but could not have existed during live execution.

### Multi-scale levels

The engine analyzes four hourly windows:

| Window | Approximate role |
|---|---|
| `48h` | Local short-term structure |
| `120h` | Swing structure |
| `240h` | Medium-term structure |
| `480h` | Long-term dynamic structure |

For every scale it estimates:

- static rolling support and resistance;
- pivot-derived dynamic trend lines;
- normalized line slope;
- line fit quality;
- touch count;
- channel width in ATR units;
- distance from price to the active level.

Raw absolute level values are excluded from model training. The model learns normalized distances, slopes, touches and quality so it is not tied to one historical BTC price range.

## Triangle detection

The causal pattern engine recognizes:

| Pattern | Structure |
|---|---|
| `SYMMETRICAL` | Falling resistance and rising support |
| `ASCENDING` | Approximately flat resistance and rising support |
| `DESCENDING` | Falling resistance and approximately flat support |

A triangle requires:

- at least two confirmed upper pivots;
- at least two confirmed lower pivots;
- converging boundaries;
- minimum contraction;
- acceptable line fit;
- bounded current width;
- sufficient pattern quality.

The model receives triangle type, contraction, quality, width and estimated apex distance as features.

## Structural events

Only these primary trade events are supported:

| Event | Trade direction |
|---|---|
| `RESISTANCE_BREAKOUT_LONG` | Long |
| `TRIANGLE_BREAKOUT_LONG` | Long |
| `SUPPORT_BREAKDOWN_SHORT` | Short |
| `TRIANGLE_BREAKDOWN_SHORT` | Short |

A breakout candle must satisfy configurable confirmation rules:

- close beyond the structural level by an ATR-normalized buffer;
- limited extension beyond the level;
- minimum candle-body strength;
- acceptable close location inside the candle;
- minimum relative volume;
- event cooldown to prevent duplicate setup counting.

Each event stores:

```text
Event ID
Breakout source
Breakout level
Invalidation level
Event direction
Event score
Triangle type
Structure regime
```

## True and false breakout labels

The model is not trained merely to predict whether price rises after a signal.

For each `1h`, `3h` and `6h` trade horizon, the label asks whether the structural break actually succeeded.

A successful breakout must:

- remain beyond the broken level at the evaluation horizon;
- avoid the structural invalidation level;
- produce positive event-aligned movement;
- satisfy the path-aware target and stop logic;
- remain profitable after execution costs and the required profit buffer.

The training data stores:

```text
breakout_hold_h*
breakout_success_h*
false_breakout_h*
event_continuation_h*
tradeable_h*
event_gross_return_h*
event_net_return_h*
```

This makes false breakouts a first-class failure mode rather than hiding them inside a generic direction label.

## Model objectives

The weekly Batch champion contains separate models for each trade horizon:

- general close direction;
- general close return;
- breakout success probability;
- breakout tradeability probability;
- breakout-aligned return.

Trade horizons are:

```text
1h · 3h · 6h
```

The public price forecast remains a next-candle `1h` contract.

## Training and validation

Version 4.0 performs full retraining from a rolling `365-day` market window.

The configured process uses:

- six chronological Walk-Forward folds;
- a six-hour embargo gap;
- recency weighting without discarding older regimes;
- stronger weights for confirmed structural events;
- calibrated tree and linear classifier blending;
- realistic execution costs;
- stress-adjusted net expectancy;
- event-type and triangle-type diagnostics.

The new model artifact uses schema version `4` and the ID prefix:

```text
structure-breakout-hourly-
```

Older model artifacts are rejected and cannot silently enter the new runtime.

## Qualification

A horizon is not tradable merely because its direction accuracy is above 50%.

It must pass all configured checks, including:

- minimum structural event count;
- breakout-success AUC;
- tradeability AUC;
- probability calibration;
- minimum breakout hold rate;
- maximum false-breakout rate;
- minimum number of selected Out-of-Fold trades;
- minimum OOF hit rate;
- positive stress-adjusted expectancy;
- sufficient positive Walk-Forward folds.

If no horizon qualifies, the system remains in `WAIT`.

## Structural risk management

The trade plan is anchored to the broken structure.

The preferred stop is placed behind the stored structural invalidation level. ATR-based risk remains a bounded fallback, not the primary definition.

A trade can still be blocked by:

```text
MODEL_NOT_QUALIFIED
SELECTED_HORIZON_NOT_QUALIFIED
NO_NEW_STRUCTURE_BREAKOUT
WEAK_BREAKOUT_STRUCTURE
BREAKOUT_LEVEL_UNAVAILABLE
INVALIDATION_LEVEL_UNAVAILABLE
LOW_BREAKOUT_SUCCESS_PROBABILITY
LOW_TRADEABILITY_PROBABILITY
BREAKOUT_HORIZON_DISAGREEMENT
INSUFFICIENT_STRESS_NET_EDGE
```

The system is intentionally conservative and remains paper-trading only.

## Public forecast contract

For a source candle with open time `T`:

| Field | Time |
|---|---|
| Source candle opens | `T` |
| Source candle closes and forecast is created | `T + 1h` |
| Target candle opens | `T + 1h` |
| Target candle closes | `T + 2h` |
| Outcome becomes eligible | after target close plus settlement delay |

Every contract is frozen and immutable after creation.

Direction outcomes:

```text
PENDING
DIRECTION_CORRECT
DIRECTION_WRONG
LEGACY_NOT_SCORED
```

Interval outcomes remain independent:

```text
IN_RANGE
OUT_OF_RANGE
```

A wide interval cannot make an incorrect direction appear correct.

## Adaptive learning

Two persistent adaptive layers remain available:

| Layer | Purpose |
|---|---|
| Trade adaptive state | Breakout success, tradeability and event-return correction |
| Price adaptive state | Next-close direction and return correction |

Both states reset when the Batch champion changes. Online influence begins at zero and receives bounded weight only after chronological prequential evaluation shows that it remains competitive with the Batch model.

The state branches are:

| Branch | Purpose |
|---|---|
| `forecast-state` | Immutable forecast ledger |
| `model-state` | Latest structural Batch champion and reports |
| `adaptive-state` | Persistent adaptive learners |

## Automation

### Structural retraining

The weekly workflow:

1. downloads the configured 365-day hourly history;
2. collects historical and recent news;
3. builds causal market structure;
4. detects structural breakouts and triangle events;
5. creates path-aware success and false-breakout labels;
6. runs Walk-Forward validation;
7. applies structural qualification;
8. publishes the new schema-v4 champion only after successful completion.

### Hourly forecast

The hourly workflow:

1. restores the latest structural champion and adaptive states;
2. runs all repository tests;
3. refreshes closed market candles and news;
4. builds current causal structure;
5. updates adaptive learners from mature labels;
6. creates one immutable next-candle forecast;
7. evaluates any newly closed target candle;
8. persists forecast and adaptive state.

Dashboard deployment is independent from model execution so a temporary data or model failure cannot keep an old interface forever.

## Local setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Fetch one year and train from scratch:

```bash
btc-regime fetch --days 365
btc-regime news --historical --days 365
btc-regime news
btc-regime train
```

Or run the full bootstrap:

```bash
btc-regime bootstrap --days 365
```

Run one cycle:

```bash
btc-regime cycle --force
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Repository layout

```text
.github/workflows/             Quality, retraining, forecast and Pages workflows
config/default.yaml            Structural, model, adaptive and risk configuration
scripts/github_weekly_retrain.py
scripts/github_hourly_forecast.py
scripts/github_dashboard.py
src/btc_ema_trader/
  market_structure.py          Causal pivots, levels, triangles and breakouts
  features.py                  Structural features and breakout labels
  structure_training.py        Walk-Forward training and qualification
  model.py                     Schema-v4 Batch models
  strategy.py                  Structural entry and risk gates
  adaptive.py                  Incremental trade learner
  price_adaptive.py            Incremental next-close learner
  forecast_contract.py         Immutable next-candle contract
tests/                         Causality, structure, forecast and repository tests
```

## Current status and limitations

Version 4.0 is a full model redesign, not evidence of profitability by itself.

The new architecture must complete fresh Walk-Forward training and then accumulate immutable live evidence. Trust should be based on:

- structural event count;
- false-breakout rate;
- breakout hold rate;
- calibrated probabilities;
- stress-adjusted expectancy;
- live results across different volatility regimes.

Until those results are sufficient, the project must be treated as transparent market-model research rather than a proven trading product.

This repository is not financial advice.

---

© 2026 Mahdi Ghahremani · ID: [TheLouisMahdi](https://github.com/TheLouisMahdi)
