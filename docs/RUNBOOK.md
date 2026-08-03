# Runbook v2

## Apply patch

1. Close the dashboard.
2. Back up `data/` and `artifacts/`.
3. Replace project files with the patch.
4. Run `setup.bat` once to migrate the database and install v2.
5. Run `start_retrain.bat`.

## Normal live run

Run `start_live.bat`. The session clock resets and the first eligible suggestion occurs only after the next fully closed one-hour candle.

## Verify

Run:

```bat
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m btc_ema_trader status
```

The model ID must begin with `regime-hourly-` and `model_schema_version` must be `2`.
