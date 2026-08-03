# Contributing

Thank you for improving BTC Hourly Forecast. Changes should preserve the project’s research focus, chronological integrity and fail-safe behavior.

## Development setup

Use Python 3.11 or newer.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Run the full test suite before submitting changes:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts
```

## Repository standards

- Write all code, comments, documentation, commit messages and user-facing text in English.
- Keep runtime state out of the `main` branch.
- Do not commit credentials, API keys, local databases, logs or generated state files.
- Preserve one-hour candle timing and the next-candle entry convention.
- Never train on labels that were unavailable at prediction time.
- Keep paper-trading safeguards enabled.
- Add or update tests for every behavior change.
- Prefer small, focused changes over broad unrelated refactors.

## Data-leakage checklist

Before changing features, labels, training or adaptive learning, verify that:

1. every feature is available at the source candle close;
2. future prices are used only for labels;
3. predictions are recorded before online updates;
4. validation splits are chronological;
5. the validation gap covers the longest forecast horizon;
6. dashboard outcome scoring matches the training contract.

## Pull requests

A pull request should explain:

- what changed;
- why the change is needed;
- how leakage and trading safety were considered;
- which tests were run;
- whether configuration or state compatibility changed.

Do not include generated model or state artifacts unless the change explicitly requires a reviewed fixture.
