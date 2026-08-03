# Changelog

## 2.0.0 — Regime/Event Patch

- Removed EMA20/EMA50 crossover logic from model and strategy.
- Added KAMA/ADX regime detection.
- Added independent Donchian breakout, squeeze-release, pullback-resume and volume-impulse events.
- Added unique Event IDs and one-trade-per-event protection.
- Changed labels to next-candle-open execution.
- Added 1h/2h/3h MFE, MAE, triple-barrier and tradeability labels.
- Added calibrated direction and tradeability model heads.
- Added per-horizon qualification and horizon selection by expected net edge.
- Added maker/taker/slippage cost breakdown and corrected realized PnL accounting.
- Added news availability-time protection using first-seen timestamps.
- Migrated dashboard to regime/event views while preserving the previous tab structure.
- Added automatic SQLite migration for existing v1 databases.
- Added seven automated tests.

## 1.0.0

- Initial EMA crossover version.

## 2.1.0

- Replaced all-candle event trading objective with event-only meta-labeling.
- Added continuation classifier, direction-aware tradeability classifier, and event-aligned return regressor.
- Fixed calibration/base-model distribution mismatch.
- Made event direction the only source of LONG/SHORT direction; generic UP/DOWN is dashboard-only.
- Qualification now uses event continuation skill, event tradeability, OOF trades, hit rate, expectancy, and fold consistency.
- Bumped artifact schema to 3; v2.0 models require retraining.
