# Contributing

Thank you for improving BTC Adaptive Directional Breakout Trader. Changes must preserve chronological integrity, state compatibility, immutable outcomes and paper-only execution.

## Development setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the full validation suite before submitting changes:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts
```

## Repository standards

- Write code, comments, documentation, commit messages and user-facing text in English.
- Keep generated runtime, model, dashboard and adaptive state out of `main`.
- Do not commit credentials, API keys, databases, logs or generated model artifacts.
- Treat GitHub Pages as the only canonical dashboard.
- Keep live-order and exchange-credential paths out of this repository.
- Add or update tests for every behavior change.
- Prefer focused changes over unrelated refactors.

## Timing and leakage checklist

Before changing features, labels, training, forecasting or adaptive learning, verify that:

1. every predictor is available when the source candle closes;
2. candle context contains only the event candle and earlier closed candles;
3. future candles are used only for labels and outcome resolution;
4. the exact next-close forecast is created after the source close and before the target close;
5. the position entry timestamp and price source are stored explicitly;
6. validation splits are chronological and the embargo covers the longest label horizon;
7. online updates occur only after the frozen prediction or position outcome is recorded;
8. resolved forecast and position outcomes are immutable;
9. runtime and training use the same feature contract;
10. legacy state migration is tested before changing any schema version.

## Trading contract checklist

The primary contract is a persistent LONG or SHORT paper position resolved by target, stop or time exit. The secondary `NEXT_CLOSED_1H_CANDLE` forecast must not replace or mutate the position contract.

When an active position exists:

- do not expose a new candidate plan as the active plan;
- do not open another position;
- report the active entry, target, current stop and mark-to-market P/L;
- retain candidate diagnostics separately when useful.

Aggressive paper mode may relax documented soft gates for exploration, but hard market-data, timing, structure, duplication and active-position checks must remain enforced.

## Pull requests

A pull request should explain:

- what changed and why;
- which contract or schema is affected;
- how leakage and timing were checked;
- how generated state remains isolated;
- which tests were run;
- whether model, forecast, trade or adaptive state migration is required.

Do not include generated model or state artifacts unless the change explicitly requires a reviewed test fixture.
