# File Manifest v2

- `features.py` — adaptive trend, regime, independent events, path labels and hourly news features
- `model.py` — calibrated direction/tradeability models and horizon scoring
- `training.py` — walk-forward evaluation, cost-adjusted metrics and qualification
- `strategy.py` — fail-safe trade gate and risk sizing
- `costs.py` — single source of truth for fees and slippage
- `runtime.py` — next-closed-candle live loop
- `storage.py` — SQLite schema migration, event signals and corrected paper resolution
- `dashboard.py` — managerial dashboard for regime/event model
