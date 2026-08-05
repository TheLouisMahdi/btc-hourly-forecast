# Changelog

## 5.5.0 — Structural risk policy and execution integrity

- Added one canonical `AGGRESSIVE_STRUCTURAL_RISK_SCALED` paper-position policy.
- Converted qualification, predicted edge, confidence, tradeability, regime alignment and soft warnings into bounded paper-account risk between 0.5% and 3.0%.
- Preserved hard blockers for invalid structure, unhealthy data, stale or unavailable quotes, direction mismatch, duplicate events and an active position.
- Disabled the shared multi-horizon adaptive blend while retaining the performance-gated price learner and resolved-position learner.
- Centralized position economics in one gap-aware module so modeled stop, execution and gap risk remain within the selected risk budget.
- Preserved policy, risk, execution and alignment metadata throughout active-position management.
- Bound new paper entries to fresh observed quotes and excluded the partial entry candle from barrier evaluation.
- Documented the current difference between batch labels at `NEXT_HOURLY_OPEN` and runtime entries at `LIVE_QUOTE_AT_SIGNAL_RUN`.
- Isolated unavailable or late secondary next-close forecasts from the primary target-stop lifecycle.
- Converted policy tests to the canonical `unittest` runner and removed the unused pytest fixture layer.
- Removed obsolete local-dashboard settings and retained GitHub Pages as the only dashboard implementation.

## 5.4.0 — Canonical position contract and repository cleanup

- Made the persistent LONG/SHORT target-stop lifecycle the primary product contract.
- Replaced the public forecast ledger with a position-only ledger showing open and realized P/L, return and R-multiple.
- Separated the active managed position from the newest hourly candidate plan.
- Added exact secondary next-close timing and rejected retroactive forecasts.
- Added causal three-candle context using the event candle and two previous closed candles.
- Added candle body, range, shadows, close location, volume and three-bar pressure features.
- Preserved immutable forecast outcomes while refreshing runtime metadata on same-candle reruns.
- Made GitHub Pages the only canonical dashboard and reduced deployment to one render command.
- Removed the obsolete local UI, command alias, unused dependencies and stale generated artifacts from `main`.
- Made `config/default.yaml` the only source of strategy, model and risk parameters.
- Added timing, context, position, workflow and repository-consistency tests.

## 5.3.0 — Adaptive target-stop paper positions

- Added a persistent position ledger with one active paper position at a time.
- Added bounded adaptive targets, structural stops, breakeven, trailing and time exits.
- Added cost-aware sizing, margin, target profit, stop loss, expected value and resolved-R learning.

## 5.2.0 — Sandwiched negative memory

- Added side-specific support and resistance risk heads.
- Added front and backup Bloom memories and joint promotion validation.

## 5.0.0 — Deterministic directional breakout models

- Added independent Long and Short model heads and deterministic event mining.
- Added six structural scales, unique event IDs, diversity gates and chronological evaluation.
- Required at least 2,000 unique events per direction without synthetic sampling.
- Added schema version 5 and the `directional-breakout-hourly-` artifact prefix.

## 4.0.0 — Causal structural breakout architecture

- Added causal pivots, support, resistance, triangles, invalidation and path-aware labels.

## 3.0.0 — Adaptive probabilistic forecasting

- Added immutable next-candle direction and interval outcomes with persistent adaptive state.
