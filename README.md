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

<img src="https://img.shields.io/badge/version-5.5.0-8f8ab8?style=flat-square" alt="Version 5.5.0" />
<img src="https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 or newer" />
<img src="https://img.shields.io/badge/training-no_random_sampling-6f9b91?style=flat-square" alt="No random sampling" />
<img src="https://img.shields.io/badge/mode-paper_trading_only-c99078?style=flat-square" alt="Paper trading only" />

</div>

# BTC Adaptive Directional Breakout Trader

This repository is a research-grade, paper-only Bitcoin trading system built around two independent structural events:

- `RESISTANCE_BREAKOUT_LONG`: a confirmed close crossing above resistance;
- `SUPPORT_BREAKDOWN_SHORT`: a confirmed close crossing below support.

The primary output is a persistent LONG or SHORT paper position with an execution quote, adaptive target, stop-loss and maximum holding time. A secondary public contract forecasts the exact `NEXT_CLOSED_1H_CANDLE` and is scored only after that candle closes.

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
entry quote and observation time
target price
initial and current stop
risk score and risk fraction
risk budget, notional and leverage
policy and risk-contract versions
stress execution cost and expected value
maximum favorable and adverse excursion
realized net P/L and R-multiple
```

Only one active position is managed at a time. A new hourly candidate is stored separately and cannot replace the active position plan. Closed outcomes are never recomputed from later prices.

### Execution-time contract

A new paper position uses a fresh observed market quote, not the previous candle close. The source candle close remains frozen as the reference for the secondary forecast.

Hourly OHLC data cannot determine whether a target or stop was touched before a mid-hour entry. The partial entry candle is therefore excluded from barrier resolution. Target, stop, breakeven and trailing calculations begin with the first fully observable hourly candle after entry.

This contract is conservative and auditable, but it exposes one known research limitation: batch event labels use the next hourly open while GitHub paper positions use the quote observed when the workflow runs. The resolved-trade learner evaluates the actual paper entry, but batch qualification is not fully execution-aligned until minute-level historical entry labels are added and a compatible challenger is promoted.

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

A wide interval cannot turn a wrong direction into a correct result. Resolved records are immutable. When the price model is unavailable or the target window is missed, the secondary forecast is `NOT_CREATED`; it cannot invalidate the primary position cycle.

## Causal candle context

Every new event feature row is represented by exactly three closed candles:

```text
PREVIOUS_2
PREVIOUS_1
EVENT
```

The feature pipeline exposes OHLCV shape, candle body, full upper and lower shadows, close location, range, volume change and three-bar pressure. Future candles are never predictors; they are used only to create labels and resolve outcomes.

A champion trained before these fields were introduced can continue to run because inputs are schema-aligned, but it does not use the new context columns until a compatible challenger is retrained, validated and promoted.

## Deterministic event mining

Support and resistance are evaluated across:

```text
24h · 48h · 96h · 168h · 336h · 720h
```

A candidate requires a real close-to-close crossing. Remaining above an already broken resistance or below an already broken support is not a new event.

Each event records its unique ID, direction, source, scale, level, structural invalidation, ATR-normalized distance, touch count, age, line quality, market regime and diversity key. Near-duplicates are removed deterministically.

## Separate Long and Short batch heads

Long and Short do not share one batch event head. Every trade horizon contains independent classifiers and return regressors for each direction.

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

The configured history target is ten years of real hourly BTC candles. One provider is selected for each training run; rows from different exchanges are not mixed into one dataset. Real gaps are segmented and audited rather than filled or interpolated.

## Validation and champion promotion

General prediction uses chronological time-series validation. Long and Short event heads use independent expanding-window Walk-Forward evaluation with an embargo.

A direction-horizon pair is evaluated using event count, AUC, calibration, selected-trade count, hit rate, stress-adjusted expectancy and positive-fold consistency. A challenger replaces the current champion only when economic validation and sandwiched negative-memory validation agree.

Candidate diagnostics are retained when promotion is rejected. Production-facing artifacts live on `model-state`, not on `main`.

## Aggressive risk-scaled position policy

The repository has one canonical entry policy:

```text
AGGRESSIVE_STRUCTURAL_RISK_SCALED
policy_version: 2
risk_contract_version: 2
```

A valid fresh structural breakout seeks a paper position. Qualification, predicted economic edge, confidence, tradeability, regime alignment and soft warnings are combined into a bounded risk score rather than creating a second conservative mode.

Paper-account risk is selected between **0.5% and 3.0%**:

```text
weak or unqualified event with negative edge -> smaller position
strong qualified event with positive edge   -> larger position
```

Soft risk evidence includes incomplete qualification, missing economic policy, weak confidence, weak tradeability, negative stress edge, volatility or news shock, stale model/news evidence, signal frequency and regime uncertainty.

Hard blockers remain fail-safe and non-negotiable:

```text
no fresh structural breakout
unsupported event type
missing breakout or invalidation level
duplicate event
Long/Short direction mismatch
short venue disabled
invalid ATR
unhealthy candles
stale or unavailable execution quote
provider mismatch
an already active position
```

The dashboard exposes the policy version, risk score, allocated risk, qualification state and soft risk evidence. An exploratory paper position is not evidence of a profitable production strategy.

Legacy positions stay in the same ledger and remain identifiable by their missing or version-1 policy metadata. New positions persist `policy_version: 2` and `entry_contract: STRUCTURAL_EVENT_RISK_SCALED`.

## Adaptive learning and negative memory

The generic shared multi-horizon adaptive blend is disabled because it can blend shared online estimates into otherwise independent Long and Short batch heads.

The dedicated price learner remains performance-gated and receives weight only when its prequential Brier score, direction accuracy and return error improve over the batch model.

The trade learner remains enabled. It updates only after a position resolves and learns target probability, stop probability and realized R from the frozen entry feature vector, including the causal three-candle context. Final target-stop economics preserve the risk fraction selected by the entry policy.

The negative-memory layer uses a sandwiched design:

```text
front Bloom memory
learned boundary-risk model
backup Bloom memory
```

Bloom hits are stored as risk evidence. The current `ADAPTIVE_PENALTY_ONLY` runtime reports or penalizes risk and never rewrites historical outcomes.

## State isolation

Generated data is intentionally kept outside `main`:

| Branch | Purpose |
|---|---|
| `forecast-state` | latest record, forecast history and position ledger |
| `model-state` | promoted champion, reports and negative memory |
| `adaptive-state` | price and resolved-trade learner states and summaries |
| `quality-state` | test and compile diagnostics |
| `training-diagnostics` | challenger training diagnostics |

`main` contains source code, configuration, tests and documentation only.

## Automation

The repository-quality workflow validates source, configuration, documentation, tests and GitHub Actions files and publishes the exact result to `quality-state`.

The hourly workflow:

1. restores forecast, position, champion and adaptive state;
2. fetches the latest contiguous closed-candle segment;
3. resolves existing positions using only fully observable post-entry candles;
4. creates one new structural decision when eligible;
5. freezes the execution quote, risk contract and position plan;
6. creates the secondary next-close contract only when its target window is valid;
7. persists forecast, position and adaptive states.

The weekly workflow:

1. runs the full validation suite;
2. fetches long historical data;
3. segments real gaps without filling or interpolation;
4. mines deterministic Long and Short events;
5. trains independent direction heads;
6. performs chronological validation;
7. trains sandwiched negative memory;
8. promotes the challenger only when all promotion gates pass.

GitHub Pages is the only canonical dashboard implementation and is rendered through one orchestration command.

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

Run one local diagnostic cycle:

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
config/default.yaml                    Canonical strategy and model configuration
scripts/github_weekly_retrain.py       Challenger training and promotion
scripts/github_structural_forecast.py  Canonical GitHub hourly entry point
scripts/github_pages_dashboard.py      Public dashboard orchestration
src/btc_ema_trader/active_position_contract.py
src/btc_ema_trader/candle_context.py   Causal three-candle context
src/btc_ema_trader/context_trade_features.py
src/btc_ema_trader/directional_events.py
src/btc_ema_trader/execution_entry.py
src/btc_ema_trader/execution_path.py
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
