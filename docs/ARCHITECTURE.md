# Architecture v2

## Pipeline

`Market data -> Leakage-safe hourly features -> Regime engine -> Independent event detector -> Direction/tradeability models -> Per-horizon qualification -> Fail-safe decision -> Paper resolution`

## Regime engine

- KAMA efficiency-adaptive trend
- ADX and DI spread
- ATR and volatility percentile
- Donchian location and channel width
- Bollinger width percentile

## Events

Events are emitted only on the confirming closed candle and receive a unique ID. Same-direction events are deduplicated for the configured cooldown.

- `DONCHIAN_BREAKOUT`
- `SQUEEZE_RELEASE`
- `PULLBACK_RESUME`
- `VOLUME_IMPULSE`

## Targets

At event candle `t`, the reference entry is `open[t+1]`. For horizon `h`, the target close is `close[t+h]`. Labels include raw direction, tradeability after cost buffer, MFE, MAE and conservative triple-barrier outcome.

## Models

Each horizon contains:

- blended gradient-boosting and logistic direction classifier;
- absolute-error return regressor;
- event-only tradeability classifier;
- temporal Platt-style probability calibrator when enough calibration data exists.

## Decision separation

The forecast is always binary UP/DOWN. The action can be WAIT when a new event is absent, the horizon is unqualified, costs consume the expected edge, data is stale or a risk limit is active.
