# Changelog

## 5.4.0 — Canonical position contract and repository cleanup

- Made the persistent LONG/SHORT target-stop lifecycle the primary product contract.
- Replaced the public forecast ledger with a position-only ledger showing open and realized P/L, return and R-multiple.
- Added an exact secondary next-close timing contract and rejected retroactive forecasts.
- Added causal three-candle context: the event candle plus the two immediately preceding closed candles.
- Added full candle-body, range, upper-shadow, lower-shadow, close-location, volume and three-bar pressure features.
- Extended the adaptive trade learner to consume the frozen causal candle context.
- Preserved legacy position outcomes by migrating old feature vectors with neutral context values.
- Made GitHub Pages the only canonical dashboard implementation.
- Removed the obsolete local Gradio/Plotly dashboard and its unused dependencies.
- Removed stale schema-v3 model and report artifacts from `main`; promoted artifacts remain isolated on `model-state`.
- Expanded `.gitignore` for generated runtime, dashboard, model, quality and training state.
- Aligned package documentation with aggressive paper exploration, economic qualification and online outcome learning.
- Added tests for strict forecast timing, causal context, position-only rendering, migration and signed P/L display.

## 5.3.0 — Adaptive target-stop paper positions

- Added a persistent paper-position ledger with one active trade at a time.
- Added adaptive targets that begin at 5R and remain bounded by configuration.
- Added structural stop-loss, breakeven, trailing-stop and maximum-holding-time logic.
- Added stress-cost-aware position sizing, notional, margin, target profit, stop loss and expected value.
- Added incremental target, stop and realized-R learning from resolved positions.
- Added deterministic replay of open position paths across hourly workflow runs.

## 5.2.0 — Sandwiched negative memory

- Added side-specific support and resistance boundary-risk heads.
- Added front and backup Bloom memories for recurring and hard negative fingerprints.
- Added joint economic and negative-memory challenger promotion.
- Added persistent negative-memory artifacts and dashboard diagnostics.

## 5.0.0 — Deterministic directional breakout models

- Replaced the shared event classifier with independent Long and Short model heads.
- Added deterministic resistance-breakout and support-breakdown mining.
- Added six structural scales: 24h, 48h, 96h, 168h, 336h and 720h.
- Added unique event IDs, direction, source, scale, touch count, age, line quality and diversity keys.
- Added a hard inventory gate requiring at least 2,000 unique Long events and 2,000 unique Short events.
- Prohibited random sampling, shuffling, synthetic events, oversampling and undersampling.
- Extended the historical target to ten years and at least 80,000 hourly candles.
- Added direction-specific path labels and 3h, 6h and 12h trade horizons.
- Added independent expanding-window evaluation and direction-specific qualification.
- Added schema version 5 and the `directional-breakout-hourly-` artifact prefix.

## 4.0.0 — Causal structural breakout architecture

- Added confirmed causal pivots, multi-scale support and resistance, line quality, touch count and age.
- Added causal triangle detection and structural invalidation levels.
- Rebuilt labels around level hold, false breakout, path-aware target or invalidation and net tradeability.
- Removed absolute price levels from model inputs and anchored stops to structure.

## 3.0.0 — Adaptive probabilistic forecasting

- Added an immutable next-candle forecast contract with direction and interval scoring.
- Added incremental price learning with prequential evaluation and bounded blending.
- Added persistent state, GitHub Pages deployment and chronological outcome tracking.
