# Repository structure

The repository keeps product logic in the installable package and limits root-level automation to small, explicit entry points.

```text
.
├── .github/workflows/       # Manual operational workflows only
│   ├── quality.yml          # Tests and compile validation
│   ├── forecast.yml         # One hourly forecast/trade cycle
│   ├── dashboard.yml        # Render and deploy GitHub Pages
│   └── retrain.yml          # Heavy challenger training on demand
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

- `src/btc_ema_trader/` owns forecasting, feature engineering, market data, risk, trade lifecycle and model behavior.
- `config/default.yaml` is the only source of strategy parameters.
- `scripts/` adapts package logic to GitHub Actions; it must not redefine model or risk formulas.
- `.github/workflows/` contains orchestration only. Every workflow is manual while maintenance mode is active.
- Generated models, reports and live state are stored on dedicated state branches, not on `main`.

## Canonical operations

Run workflows in this order after a maintenance change:

1. `Repository quality`
2. `Hourly BTC forecast` with `allow_retrain=false`
3. `Deploy BTC dashboard`
4. `On-demand BTC model retraining` only when the retraining policy or a deliberate manual review requires it

The dashboard has one public renderer: `scripts/render_dashboard.py`. The older `github_*_dashboard.py` files are deterministic internal components, not workflow entry points.

`github_structural_forecast.py` remains only as a backward-compatible shim. New automation must call `github_hourly_forecast.py` directly.
