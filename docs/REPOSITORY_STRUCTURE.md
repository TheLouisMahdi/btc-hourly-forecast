# Repository structure

The repository keeps product logic in the installable package and limits root-level automation to small, explicit entry points.

```text
.
├── .github/workflows/       # Small operational workflows
│   ├── quality.yml          # Main-branch + daily validation, with manual fallback
│   ├── forecast.yml         # Hourly forecast/trade-assistant cycle + manual run
│   ├── dashboard.yml        # Auto deploy after a successful forecast + manual run
│   └── retrain.yml          # Heavy challenger training, strictly manual
├── config/
│   └── default.yaml         # Canonical strategy and runtime configuration
├── docs/                    # Product, architecture and visual assets
├── scripts/                 # GitHub adapters and operational entry points
├── src/btc_ema_trader/      # Canonical application and model logic
├── tests/                   # Unit, contract and repository consistency tests
├── pyproject.toml           # Package metadata and dependencies
└── requirements-github.txt  # Reproducible GitHub Actions environment
```

## Boundaries

- `src/btc_ema_trader/` owns forecasting, feature engineering, market data, risk, trade lifecycle, pattern memory, precision meta filtering and model behavior.
- `config/default.yaml` is the only source of strategy and trade-assistant parameters.
- `scripts/` adapts package logic to GitHub Actions; it must not redefine model or risk formulas.
- `.github/workflows/` contains orchestration only. The continuous automation contract is intentionally small: hourly forecasting, forecast-driven dashboard deployment and repository validation. Retraining has no automatic trigger.
- Generated models, reports and live state are stored on dedicated state branches, not on `main`.

## Trade-assistant layers

The primary public contract is the exact next closed one-hour candle direction forecast. Position suggestions are secondary and require a qualified precision meta head before a new paper position may open.

Key modules:

- `price_adaptive.py`: guarded online support for the primary one-hour forecast.
- `pattern_memory.py`: static false-breakout Bloom memory with exact counts plus persistent live next-candle mistake memory.
- `meta_filter.py`: precision-first position TAKE/SKIP layer, explicit false-breakout probability and horizon-aligned MFE/MAE exit profiles.
- `trade_assistant_bootstrap.py`: GitHub runtime overlay that leaves the serialized incumbent model untouched and preserves existing open-position contracts.
- `negative_memory.py`: structural boundary memory and bounded risk penalties.

See `docs/TRADE_ASSISTANT_ARCHITECTURE.md` for the full data, qualification and promotion contract.

## State branches

- `model-state`: champion model plus static negative memory, precision meta filter and static false-pattern memory.
- `adaptive-state`: online price/trade learners plus live next-candle pattern memory.
- `forecast-state`: immutable forecast history, paper-position ledger and chart candle state.

## Continuous workflow schedule

### Hourly BTC forecast

- Automatic schedule: every hour at `HH:12 UTC` using `12 * * * *`.
- Manual execution remains available.
- Each run restores the latest model and adaptive state, fetches fresh closed-market data, resolves previous outcomes, runs the trade-assistant cycle and persists the new forecast/adaptive state.
- The workflow may calculate a retraining recommendation for visibility, but it cannot dispatch retraining.

### Deploy BTC dashboard

- No independent cron is used.
- The dashboard starts automatically after `Hourly BTC forecast` completes successfully through `workflow_run`.
- Failed or cancelled forecast runs do not trigger a deployment.
- Manual deployment remains available for maintenance.

### Repository quality

- Runs automatically after every push to `main`.
- Also runs once per day at `03:37 UTC` using `37 3 * * *` as a safety check for dependency/configuration drift.
- Manual execution remains available.
- Validation writes diagnostics to `quality-state`; that state-branch update does not retrigger quality because the push trigger is restricted to `main`.

### On-demand BTC model retraining

- Strictly `workflow_dispatch` only.
- No cron, push, workflow-run dependency or dispatch from the hourly forecast exists.
- A human explicitly starts the heavy challenger-training workflow when retraining is desired.

## Normal operation

1. `Hourly BTC forecast` runs at `HH:12 UTC`.
2. A successful forecast automatically triggers `Deploy BTC dashboard`.
3. `Repository quality` continuously validates changes to `main` and also performs the daily safety run.
4. `On-demand BTC model retraining` runs only when explicitly started by a person.

A one-time manual retrain is required before the new precision meta filter can authorize future positions. Until a qualified meta artifact exists, the one-hour forecast remains available and new positions stay experimentally blocked.

The dashboard has one public renderer: `scripts/render_dashboard.py`. The older `github_*_dashboard.py` files are deterministic internal components, not workflow entry points.

`github_structural_forecast.py` remains only as a backward-compatible shim. New automation must call `github_hourly_forecast.py` directly.
