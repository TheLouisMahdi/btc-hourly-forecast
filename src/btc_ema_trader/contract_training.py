from __future__ import annotations

import pandas as pd

from .config import Settings
from .features import build_feature_set
from .forecast_contract import attach_close_based_general_labels
from .storage import Database
from .training import train_feature_set


def train_from_database(
    settings: Settings,
    database: Database,
    provider: str | None = None,
) -> dict[str, object]:
    market_cfg = settings.section("market")
    symbol = str(market_cfg.get("symbol", "BTCUSDT"))
    if provider is None:
        candidates = database.providers(symbol)
        if not candidates:
            raise ValueError(
                "No candle history found. Run: btc-regime fetch --days 180"
            )
        provider = str(candidates[0]["provider"])

    candles = database.load_candles(
        provider=provider,
        symbol=symbol,
    )
    history_days = float(market_cfg.get("history_days", 180))
    cutoff = candles["open_time"].max() - pd.Timedelta(
        days=history_days
    )
    candles = candles[
        candles["open_time"] >= cutoff
    ].reset_index(drop=True)
    news = database.load_news(
        start=candles["open_time"].min(),
        end=candles["open_time"].max()
        + pd.Timedelta(hours=1),
    )
    feature_set = build_feature_set(
        candles,
        news,
        settings,
        include_labels=True,
    )
    feature_set.frame = attach_close_based_general_labels(
        feature_set.frame,
        feature_set.horizons,
    )
    return train_feature_set(
        settings,
        feature_set,
        provider=provider,
        symbol=symbol,
    )
