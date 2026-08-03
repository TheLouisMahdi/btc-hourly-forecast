# Changelog

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
- Updated workflows, CLI defaults, documentation and project metadata for version 4.0.0.

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
- Updated the README and project metadata for version 3.2.0.

## 3.1.0 — Next-candle probabilistic forecast contract

- Replaced direction-only public outcomes with an 80% probable close range for the next closed hourly candle.
- Added explicit source open, source close, target open and target close timestamps.
- Added a 90-second settlement delay before outcome resolution.
- Added immutable `IN_RANGE` and `OUT_OF_RANGE` results.
- Added independent direction scoring.
- Excluded legacy direction-only forecasts from interval coverage metrics.
- Added empirical prequential residual calibration from resolved live forecasts.
- Added a conservative model-error and recent-market-range fallback before enough residuals exist.
- Changed the configured model horizon to the next one-hour candle only.
- Added regression tests for premature resolution and outcome mutation.
- Reworked the static dashboard around probabilistic next-close ranges.
- Rebuilt the README as a concise English project overview.
- Replaced the external header renderer with a repository-owned animated candlestick SVG.
- Added contribution and security policies.
- Added a shared editor configuration.
- Added a dedicated repository-quality workflow.

## 3.0.0 — Adaptive online learning

- Added a persistent incremental learning layer for 1-hour, 2-hour and 3-hour horizons.
- Added delayed-label synchronization using the same label definitions as batch training.
- Added prequential base-versus-online evaluation.
- Added per-horizon shadow, active and suspended states.
- Added bounded adaptive blending and automatic degradation fallback.
- Added the `adaptive-state` GitHub snapshot branch.
- Added adaptive performance metrics to the static dashboard.
- Corrected dashboard outcome scoring to use next-candle entry and horizon close.
- Consolidated dashboard rendering into one English-only implementation.
- Added adaptive, outcome and repository-quality tests.
- Removed obsolete patch files and Persian deployment documentation.

## 2.1.0 — Event meta-labeling

- Replaced EMA crossover events with independent regime and event detection.
- Added event continuation, tradeability and event-return models.
- Added strict walk-forward qualification and stress-adjusted strategy gates.
- Added automated hourly forecasting, weekly retraining and GitHub Pages deployment.
