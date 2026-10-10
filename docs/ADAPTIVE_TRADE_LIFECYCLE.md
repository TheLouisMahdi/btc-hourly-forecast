# Adaptive target-stop trade lifecycle

Version 5.3 changes the primary paper-trading contract from a one-candle price range to a persistent trade lifecycle.

## Primary objective

A valid model LONG or SHORT direction can open one paper position. Structural events, qualification and meta-model evidence adjust risk but do not have to approve the entry. The position remains open across hourly candles until one of these events occurs:

1. the adaptive take-profit target is touched;
2. the current stop-loss is touched;
3. the maximum holding time expires and the trade exits at the closing price.

The next closed one-hour candle forecast remains available as secondary research output, but it no longer replaces the trade direction, holding horizon or trade economics.

## Initial asymmetric contract

The initial reward-to-risk ratio is `5R`:

- LONG target = entry + 5 × initial stop distance;
- LONG stop = entry − initial stop distance;
- SHORT target = entry − 5 × initial stop distance;
- SHORT stop = entry + initial stop distance.

When structural invalidation is available, it contributes to the stop distance; otherwise an ATR-based stop is used. The stop is capped between configured minimum and maximum percentages.

## Adaptive exits

After enough resolved trades, the online learner estimates:

- probability of target before stop;
- probability of stop before target;
- expected realized R-multiple.

These outputs adapt:

- target multiple between 3R and 8R;
- stop width within configured limits;
- maximum holding time between 12 and 168 hours.

At 2R maximum favorable excursion the stop can move toward stress-cost-adjusted break-even. At 3R a fixed-ATR trailing stop is activated.

## Online feedback

Every closed trade stores the entry feature vector and realized outcome. The online models use `partial_fit` and learn only once from each trade ID. Stop-loss outcomes receive extra learning weight so attractive-looking failure patterns are corrected faster.

The persistent files are:

- `forecast-state/trades.json` — open and resolved paper-trade ledger;
- `adaptive-state/trade_adaptive_state.joblib` — online target, stop and R models;
- `adaptive-state/trade_summary.json` — public summary of live paper evidence.

## Paper-only aggressive mode

The GitHub workflow enables model-first aggressive paper mode. A missing structural event prevents a new position. For confirmed structural events, model qualification, edge threshold, news shock, duplicate-event context, meta-model rejection and negative-memory warnings remain advisory or risk penalties instead of automatic vetoes. Hard blockers remain for unusable direction, invalid prices or ATR, unhealthy candle data, stale or unavailable execution quotes, provider mismatch and unsupported short execution. Only one paper position is managed at a time.

No exchange order is submitted by this repository.

## Model-only entry eligibility

When no confirmed structural event exists, the position action is always `WAIT`. Event-only continuation, tradeability, and expected-return heads have no validated trade label for a model-direction-only row; even a positive event-only qualification cannot authorize these entries. The general UP/DOWN forecast is still published. Confirmed structural events retain the existing risk-scaled advisory policy. To enable model-only entries in the future, build and independently validate an economic model for non-event rows.
