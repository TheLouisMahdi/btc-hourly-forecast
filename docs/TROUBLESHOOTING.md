# Troubleshooting

## All providers failed

Possible causes: DNS, VPN, regional blocking, firewall or a temporary exchange outage. The program tries Binance Futures, Bybit Linear and OKX Swap in order. Retry with a specific source:

```text
.venv\Scripts\python -m btc_ema_trader fetch --days 180 --provider bybit_linear
```

## News collection warning

Training can continue with `news_available=0`, but the dashboard reports missing/stale news. Retry:

```text
start_news_refresh.bat
```

## Model remains blocked

This is not a software error. It means walk-forward qualification did not prove enough edge. Forecasts remain UP/DOWN, but trade actions stay WAIT by design.

## Dashboard opens but no signal appears

The live session intentionally waits for the next complete one-hour candle after process start. Check the countdown in the top cards.

## Port 7860 is occupied

Change `live.dashboard_port` in `config/default.yaml`.
