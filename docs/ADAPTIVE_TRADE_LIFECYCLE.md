# Adaptive target-stop trade lifecycle

Version 5.3 changes the primary paper-trading contract from a one-candle price range to a persistent trade lifecycle.

## Primary objective

A qualified structural signal can open one paper position. The position remains open across hourly candles until one of these events occurs:

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

The stop distance combines the structural invalidation distance and an ATR floor. It is capped between configured minimum and maximum percentages.

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

The GitHub workflow enables aggressive paper mode. Model qualification, edge threshold, news shock and negative-memory warnings become soft penalties instead of automatic vetoes. Hard blockers remain for missing structural events, invalid prices, unhealthy candle data, stale quotes, provider mismatch, duplicate events and unsupported short execution.

No exchange order is submitted by this repository.
