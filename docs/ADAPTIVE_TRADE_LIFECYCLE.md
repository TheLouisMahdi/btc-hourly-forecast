# Adaptive target-stop trade lifecycle

Version 5.3 changes the primary paper-trading contract from a one-candle price range to a persistent trade lifecycle.

## Primary objective

A confirmed structural event can open a paper position when the remaining execution safety gates pass. A model-only LONG or SHORT forecast cannot authorize an entry without separately validated economic evidence. For eligible structural events, qualification and meta-model evidence continue to adjust risk under the paper policy. The position remains open across hourly candles until one of these events occurs:

1. the adaptive take-profit target is touched;
2. the current stop-loss is touched;
3. the maximum holding time expires and the trade exits at the closing price.

The next closed one-hour candle forecast remains available as secondary research output, but it no longer replaces the trade direction, holding horizon or trade economics.

## Initial asymmetric contract

The reward-to-risk multiple is configuration-driven (currently `4R` in `config/default.yaml`, bounded by `2.5R` and `6R`):

- LONG target = entry + configured reward multiple × initial stop distance;
- LONG stop = entry − initial stop distance;
- SHORT target = entry − configured reward multiple × initial stop distance;
- SHORT stop = entry + initial stop distance.

When structural invalidation is available, it contributes to the stop distance; otherwise an ATR-based stop is used. The stop is capped between configured minimum and maximum percentages.

## Adaptive exits

After enough resolved trades, the online learner estimates:

- probability of target before stop;
- probability of stop before target;
- expected realized R-multiple.

These outputs adapt:

- target multiple within configured limits (currently 2.5R to 6R);
- stop width within configured limits;
- maximum holding time between 12 and 168 hours.

The current configuration uses 0.75R maximum favorable excursion to consider stress-cost-adjusted break-even and 1.25R to consider a fixed-ATR trailing stop. Changes become active only after a complete candle and only if the proposed stop is on the valid side of its closing price.

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

## Candle-close stop activation safety

Break-even and trailing changes are evaluated after the full hourly candle is complete. A newly proposed LONG stop must remain strictly below that candle's closing price; a SHORT stop must remain strictly above it. When the candle retraces through a proposed level, the existing stop remains in force instead of assuming that an invalid stop order can be placed retroactively. Historical entries and already resolved paper trades are not rewritten.

## Gap-fill timestamps and hourly evidence

A full hourly candle can confirm at its end that its **opening** price crossed a stop or target. For `GAP_THROUGH_STOP` and `GAP_THROUGH_TARGET`, `closed_at` now records that candle's **open**, the first observable fill price time. `exit_evidence_available_at` separately records the candle end when the hourly evidence becomes available; `exit_time_basis` labels `CANDLE_OPEN`, `INTRABAR_END_BOUND`, or `CANDLE_CLOSE`. For intrabar touches, the exact minute is unknown and `closed_at` remains the candle-end upper bound. The partially observed entry candle is still excluded from OHLC barrier tests to prevent pre-entry look-ahead. Old closed trades are not recalculated.

## Historical policy attribution and recorded-entry replay

The review workflow `.github/workflows/paper-policy-attribution.yml` reads the `forecast-state` ledger and 168 recent hourly candles **without modifying** either branch. It publishes a report artifact containing (1) original recorded PnL versus a filtered subset of recorded structural opportunities and (2) a paired exit replay using the original `main` exit engine and the proposed review engine on **identical recorded entries**. Reports retain trade IDs, execution costs, close-only drawdown, OHLC coverage, and per-trade replay differences.

Neither method is a complete policy backtest: filtering past trades cannot recover new opportunities that would become available after a blocked entry; two policies would also retrain their online models differently. Only a limited recent subset has full price coverage, and minute-level intrabar execution remains unavailable. The baseline replay should be checked against the originally recorded ledger before interpreting any candidate delta. Artifacts are created only for research; no orders are placed.
