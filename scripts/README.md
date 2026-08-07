# Operational scripts

Only the following files are workflow entry points:

| Entry point | Purpose |
|---|---|
| `github_hourly_forecast.py` | Fetch fresh closed candles, run one canonical forecast/trade-assistant cycle and persist adaptive state. |
| `render_dashboard.py` | Render the full dashboard in a fixed component order. |
| `github_retrain_assistant.py` | Run canonical challenger training, then build and validate precision-meta and false-pattern artifacts for promotion. |
| `github_retrain_policy.py` | Decide whether heavyweight retraining is justified. |

`github_weekly_retrain.py` remains the canonical base-training implementation and is called by `github_retrain_assistant.py`; workflows must use the assistant-aware wrapper so a challenger cannot be promoted without the precision-meta gate.

## Shared adapters

- `github_common.py`: GitHub-specific paths, state restoration helpers, JSON serialization and installation of the runtime-only trade-assistant overlay.
- `github_dashboard.py`: resolve immutable next-candle outcomes and build the base HTML surface.
- `github_assistant_dashboard.py`: display the primary 1h / secondary precision-gated position contract and fake-pattern memory state.
- `push_snapshot_branch.sh`: publish generated state to dedicated branches.

## Dashboard components

`github_pages_dashboard.py`, `github_boundary_dashboard.py`, `github_trade_dashboard.py`, `github_timing_dashboard.py`, `github_visual_dashboard.py`, `github_uncertainty_dashboard.py`, `github_resilience_dashboard.py`, `github_assistant_dashboard.py` and `github_chart_dashboard.py` are deterministic rendering stages. Workflows must call only `render_dashboard.py`.

## Compatibility

`github_structural_forecast.py` is retained for older integrations. New code must use `github_hourly_forecast.py`.

The incumbent serialized model stays backward-compatible. The precision meta filter and static false-pattern memory are independent model-state artifacts; live next-candle mistake memory is independent adaptive state.
