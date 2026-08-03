# Changelog

## Unreleased

- Rebuilt the README as a concise English project overview.
- Replaced the external header renderer with a repository-owned animated candlestick SVG.
- Added contribution and security policies.
- Added a shared editor configuration.
- Added a dedicated repository-quality workflow for source and documentation changes.
- Extended repository tests to enforce English-only text, required professional files and local README assets.

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
