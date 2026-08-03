# Research and implementation basis

Primary technical references used for the v2 design:

- Official exchange REST/WebSocket market-data documentation for Binance Futures, Bybit V5 and OKX V5.
- scikit-learn `TimeSeriesSplit` for ordered expanding validation with an embargo gap.
- scikit-learn probability-calibration documentation and reliability-diagram methodology.
- Triple-barrier and meta-labeling literature for path-aware event outcomes.
- Recent hourly BTC research emphasizing that target design, regime conditioning and transaction costs can matter more than adding a deeper model.

Design consequences:

- market APIs are paginated, retried and provider-locked;
- time-series folds never train on future observations;
- event labels use the open of the next candle;
- news becomes available only after both publication and first collection;
- direction and tradeability are modeled separately;
- expected edge is tested after maker/taker fees, slippage and stress costs;
- a model can forecast UP/DOWN while the trade gate remains WAIT.
