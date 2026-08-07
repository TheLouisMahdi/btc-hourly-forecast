# Repository structure

The repository keeps product logic in the installable package and limits root-level automation to small, explicit entry points.

```text
.
├── .github/workflows/       # Manual operational workflows only
│   ├── quality.yml          # Tests and compile validation
│   ├── forecast.yml         # One forecast/trade-assistant cycle
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

- `src/btc_ema_trader/` owns forecasting, feature engineering, market data, risk, trade lifecycle, pattern memory, precision meta filtering and model behavior.
- `config/default.yaml` is the only source of strategy and trade-assistant parameters.
- `scripts/` adapts package logic to GitHub Actions; it must not redefine model or risk formulas.
- `.github/workflows/` contains orchestration only. Every workflow is manual while maintenance mode is active.
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

## Canonical operations

Run workflows in this order after a maintenance change:

1. `Repository quality`
2. `Hourly BTC forecast` with `allow_retrain=false`
3. `Deploy BTC dashboard`
4. `On-demand BTC model retraining` only when the retraining policy or a deliberate manual review requires it

A one-time on-demand retrain is required before the new precision meta filter can authorize future positions. Until a qualified meta artifact exists, the one-hour forecast remains available and new positions stay experimentally blocked.

The dashboard has one public renderer: `scripts/render_dashboard.py`. The older `github_*_dashboard.py` files are deterministic internal components, not workflow entry points.

`github_structural_forecast.py` remains only as a backward-compatible shim. New automation must call `github_hourly_forecast.py` directly.
