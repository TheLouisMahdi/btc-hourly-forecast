# Operational scripts

Only the following files are workflow entry points:

| Entry point | Purpose |
|---|---|
| `github_hourly_forecast.py` | Fetch fresh closed candles, run one canonical forecast/trade cycle and persist adaptive state. |
| `render_dashboard.py` | Render the full dashboard in a fixed component order. |
| `github_weekly_retrain.py` | Train and validate a challenger model when the on-demand workflow is requested. |
| `github_retrain_policy.py` | Decide whether heavyweight retraining is justified. |

## Shared adapters

- `github_common.py`: GitHub-specific paths, state restoration helpers and JSON serialization.
- `github_dashboard.py`: resolve immutable next-candle outcomes and build the base HTML surface.
- `push_snapshot_branch.sh`: publish generated state to dedicated branches.

## Dashboard components

`github_pages_dashboard.py`, `github_boundary_dashboard.py`, `github_trade_dashboard.py`, `github_timing_dashboard.py`, `github_visual_dashboard.py`, `github_uncertainty_dashboard.py` and `github_resilience_dashboard.py` are internal rendering stages. Workflows must call only `render_dashboard.py`.

## Compatibility

`github_structural_forecast.py` is retained for older integrations. New code must use `github_hourly_forecast.py`.
