# Changelog

## Unreleased — Exact trade ledger and causal candle context

- Replaced the final forecast-history table with a position-only LONG/SHORT ledger.
- Added realized P&L for closed positions and mark-to-market P&L for open positions.
- Added entry, target, stop, mark/exit, P&L percentage, R-multiple and outcome columns.
- Added a strict contract version 3 wrapper that refuses forecasts created before the source close or after the target close.
- Bound every secondary forecast to one exact future closed hourly candle and added an explicit settlement timestamp.
- Added a causal three-candle context contract: the event candle plus the two immediately preceding closed candles.
- Added normalized body, range, upper-shadow, lower-shadow, close-location, volume and three-bar interaction features.
- Persisted raw OHLCV and shadow values with each opened paper position for auditability.
- Kept future candles out of model inputs; they remain available only for outcome labels and trade resolution.
- Added regression tests for prefix stability, raw candle context, stale forecast rejection and position-only dashboard output.

## 5.0.0 — Deterministic directional breakout models

- Replaced the shared event classifier with independent Long and Short model heads.
- Added deterministic resistance-breakout and support-breakdown candidate mining.
- Added separate Long and Short crossing, candle, volume, trend, invalidation, target and hold formulas.
- Added six structural scales: 24h, 48h, 96h, 168h, 336h and 720h.
- Added unique event IDs, direction, source, scale, touch count, age, line quality and diversity keys.
- Added deterministic duplicate suppression based on time and ATR-normalized level similarity.
- Added a hard inventory gate requiring at least 2,000 unique Long events and 2,000 unique Short events.
- Added mandatory multi-year, multi-quarter, multi-scale, volatility-regime and market-regime coverage.
- Explicitly prohibited random sampling, shuffling, synthetic events, oversampling and undersampling.
- Extended the configured market-history target to ten years and at least 80,000 hourly candles.
- Added Coinbase BTC-USD spot and Binance BTC-USDT spot historical providers.
- Kept each training run on one provider instead of mixing exchanges inside one dataset.
- Added direction-specific success, false-breakout, neutral and tradeability labels.
- Added 3h, 6h and 12h structural trade horizons while retaining the public next-candle 1h forecast.
- Added independent expanding-window OOF evaluation for Long and Short events.
- Added direction-specific qualification so one direction cannot authorize the other.
- Added schema version 5 and the `directional-breakout-hourly-` artifact prefix.
- Disabled the generic online trade learner until separate online Long and Short heads are implemented and validated.
- Added tests for deterministic mining, duplicate prevention, separate formulas and inventory-gate failure.

## 4.0.0 — Causal structural breakout model

- Replaced the previous generic event engine with a structural breakout architecture.
- Added confirmed causal pivots with explicit right-side confirmation timing.
- Added multi-scale static and dynamic support and resistance across 48h, 120h, 240h and 480h windows.
- Added line slope, fit quality, touch count, level age, ATR-normalized width and distance features.
- Added causal symmetrical, ascending and descending triangle detection.
- Added triangle contraction, line quality, width and apex-distance features.
- Restricted primary trade events to resistance breakouts, support breakdowns and triangle boundary breaks.
- Added ATR-normalized breakout confirmation, maximum extension, candle-body, close-location and volume checks.
- Added structural breakout levels and invalidation levels to every event.
- Rebuilt labels around level hold, false breakout, path-aware target or invalidation and net tradeability.
- Added `1h`, `3h` and `6h` structural trade horizons while retaining the public next-candle one-hour forecast.
- Extended full retraining history from 180 to 365 days.
- Added six-fold chronological Walk-Forward validation with a six-hour embargo.
- Added breakout hold-rate and false-breakout-rate qualification gates.
- Rejected all pre-v4 model artifacts and introduced the `structure-breakout-hourly-` model ID prefix.
- Removed raw absolute support, resistance and EMA levels from model training.
- Anchored stop placement to structural invalidation with a bounded ATR fallback.
- Added causal pivot, triangle, long-breakout, short-breakdown and prefix-stability tests.

## 3.2.0 — Direction-first adaptive price fusion

- Made next-candle direction the primary forecast result.
- Replaced `RANGE` direction output with explicit `UP` or `DOWN` plus confidence strength.
- Added a dedicated incremental price learner for close direction and close return.
- Added independent performance-weighted fusion for direction and price magnitude.
- Added bounded online probabilities and returns to reduce unstable overconfidence.
- Removed recent candle range as the primary interval source.
- Centered price intervals on the fused Batch and Online return estimate.
- Kept interval coverage as a secondary diagnostic separate from direction accuracy.
- Added a persistent `price_adaptive_state.joblib` artifact and price summary.
- Added tests for blend maturity, blend safety and direction-first outcome scoring.
- Rebuilt the public dashboard with a soft modern visual system and calmer colors.
- Added author identity, GitHub ID and copyright details to the dashboard footer.

## 3.1.0 — Next-candle probabilistic forecast contract

- Replaced direction-only public outcomes with an 80% probable close range for the next closed hourly candle.
- Added explicit source open, source close, target open and target close timestamps.
- Added a 90-second settlement delay before outcome resolution.
- Added immutable `IN_RANGE` and `OUT_OF_RANGE` results.
- Added independent direction scoring.
- Excluded legacy direction-only forecasts from interval coverage metrics.
- Added empirical prequential residual calibration from resolved live forecasts.
- Added a conservative model-error and recent-market-range fallback before enough residuals exist.
- Added regression tests for premature resolution and outcome mutation.

## 3.0.0 — Adaptive online learning

- Added a persistent incremental learning layer.
- Added delayed-label synchronization and prequential base-versus-online evaluation.
- Added bounded adaptive blending and degradation fallback.
- Added adaptive state persistence and outcome tests.

## 2.1.0 — Event meta-labeling

- Replaced EMA crossover events with independent regime and event detection.
- Added event continuation, tradeability and event-return models.
- Added strict walk-forward qualification and stress-adjusted strategy gates.
