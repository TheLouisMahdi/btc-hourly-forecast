from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from .config import Settings
from .storage import Database

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Quote:
    provider: str
    symbol: str
    price: float
    timestamp: pd.Timestamp


class MarketDataClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.section("market")
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "btc-directional-breakout/5.0"}
        )

    def fetch_history(
        self,
        days: float | None = None,
        provider: str | None = None,
    ) -> tuple[str, pd.DataFrame]:
        days = float(days or self.cfg.get("history_days", 3650))
        requested = provider or str(
            self.cfg.get("provider", "auto")
        )
        providers = (
            [requested]
            if requested != "auto"
            else list(self.cfg.get("provider_order", []))
        )
        errors: dict[str, str] = {}
        for name in providers:
            try:
                frame = self._fetch_provider(name, days)
                self._validate(frame, days)
                return name, frame
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "Market provider %s failed: %s",
                    name,
                    exc,
                )
        raise RuntimeError(f"All market providers failed: {errors}")

    def refresh_recent(
        self,
        provider: str,
        days: float = 7.0,
    ) -> pd.DataFrame:
        frame = self._fetch_provider(provider, days)
        self._validate(
            frame,
            min(days, 2.0),
            enforce_minimum_rows=False,
        )
        return frame

    def live_quote(
        self,
        provider_hint: str | None = None,
    ) -> Quote:
        order = [provider_hint] if provider_hint else []
        order += [
            provider
            for provider in self.cfg.get("provider_order", [])
            if provider not in order
        ]
        errors: dict[str, str] = {}
        for provider in order:
            try:
                if provider == "coinbase_spot":
                    product = self.cfg.get("coinbase_product", "BTC-USD")
                    url = (
                        self.cfg["coinbase_base_url"].rstrip("/")
                        + f"/products/{product}/ticker"
                    )
                    payload = self._get_json(url, {})
                    timestamp = pd.to_datetime(
                        payload.get("time"),
                        utc=True,
                        errors="coerce",
                    )
                    if pd.isna(timestamp):
                        timestamp = pd.Timestamp.now(tz="UTC")
                    return Quote(
                        provider,
                        self.cfg.get("symbol", "BTCUSDT"),
                        float(payload["price"]),
                        timestamp,
                    )
                if provider == "binance_spot":
                    url = (
                        self.cfg["binance_spot_base_url"].rstrip("/")
                        + "/api/v3/ticker/price"
                    )
                    payload = self._get_json(
                        url,
                        {"symbol": self.cfg.get("symbol", "BTCUSDT")},
                    )
                    return Quote(
                        provider,
                        self.cfg.get("symbol", "BTCUSDT"),
                        float(payload["price"]),
                        pd.Timestamp.now(tz="UTC"),
                    )
                if provider == "binance_futures":
                    url = (
                        self.cfg["binance_futures_base_url"].rstrip("/")
                        + "/fapi/v1/ticker/price"
                    )
                    payload = self._get_json(
                        url,
                        {"symbol": self.cfg.get("symbol", "BTCUSDT")},
                    )
                    return Quote(
                        provider,
                        self.cfg.get("symbol", "BTCUSDT"),
                        float(payload["price"]),
                        pd.Timestamp.now(tz="UTC"),
                    )
                if provider == "bybit_linear":
                    url = (
                        self.cfg["bybit_base_url"].rstrip("/")
                        + "/v5/market/tickers"
                    )
                    payload = self._get_json(
                        url,
                        {
                            "category": "linear",
                            "symbol": self.cfg.get("symbol", "BTCUSDT"),
                        },
                    )
                    item = payload["result"]["list"][0]
                    return Quote(
                        provider,
                        self.cfg.get("symbol", "BTCUSDT"),
                        float(item["lastPrice"]),
                        pd.to_datetime(
                            int(payload.get("time", 0)),
                            unit="ms",
                            utc=True,
                        ),
                    )
                if provider == "okx_swap":
                    url = (
                        self.cfg["okx_base_url"].rstrip("/")
                        + "/api/v5/market/ticker"
                    )
                    payload = self._get_json(
                        url,
                        {
                            "instId": self.cfg.get(
                                "okx_instrument",
                                "BTC-USDT-SWAP",
                            )
                        },
                    )
                    item = payload["data"][0]
                    return Quote(
                        provider,
                        self.cfg.get("symbol", "BTCUSDT"),
                        float(item["last"]),
                        pd.to_datetime(
                            int(item["ts"]),
                            unit="ms",
                            utc=True,
                        ),
                    )
            except Exception as exc:
                errors[str(provider)] = str(exc)
        raise RuntimeError(f"Live quote unavailable: {errors}")

    def _fetch_provider(
        self,
        provider: str,
        days: float,
    ) -> pd.DataFrame:
        if provider == "coinbase_spot":
            return self._fetch_coinbase(days)
        if provider == "binance_spot":
            return self._fetch_binance(days, futures=False)
        if provider == "binance_futures":
            return self._fetch_binance(days, futures=True)
        if provider == "bybit_linear":
            return self._fetch_bybit(days)
        if provider == "okx_swap":
            return self._fetch_okx(days)
        raise ValueError(f"Unsupported provider: {provider}")

    def _fetch_coinbase(self, days: float) -> pd.DataFrame:
        end = pd.Timestamp.now(tz="UTC").floor("h")
        start = end - pd.Timedelta(days=days)
        product = self.cfg.get("coinbase_product", "BTC-USD")
        url = (
            self.cfg["coinbase_base_url"].rstrip("/")
            + f"/products/{product}/candles"
        )
        cursor = start
        rows: list[list[Any]] = []
        window = pd.Timedelta(hours=290)
        while cursor < end:
            batch_end = min(cursor + window, end)
            batch = self._get_json(
                url,
                {
                    "granularity": 3600,
                    "start": cursor.isoformat(),
                    "end": batch_end.isoformat(),
                },
            )
            if isinstance(batch, list):
                rows.extend(batch)
            cursor = batch_end
            time.sleep(0.14)
        now = pd.Timestamp.now(tz="UTC")
        data: list[dict[str, Any]] = []
        for row in rows:
            if len(row) < 6:
                continue
            open_time = pd.to_datetime(
                int(row[0]),
                unit="s",
                utc=True,
            )
            if not (start <= open_time < end):
                continue
            data.append(
                {
                    "provider": "coinbase_spot",
                    "symbol": self.cfg.get("symbol", "BTCUSDT"),
                    "open_time": open_time,
                    "open": float(row[3]),
                    "high": float(row[2]),
                    "low": float(row[1]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_volume": None,
                    "trades": None,
                    "closed": True,
                    "fetched_at": now,
                }
            )
        return self._normalize(data)

    def _fetch_binance(
        self,
        days: float,
        futures: bool,
    ) -> pd.DataFrame:
        end = pd.Timestamp.now(tz="UTC").floor("h")
        start = end - pd.Timedelta(days=days)
        if futures:
            url = (
                self.cfg["binance_futures_base_url"].rstrip("/")
                + "/fapi/v1/klines"
            )
            provider = "binance_futures"
        else:
            url = (
                self.cfg["binance_spot_base_url"].rstrip("/")
                + "/api/v3/klines"
            )
            provider = "binance_spot"
        cursor_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        rows: list[list[Any]] = []
        while cursor_ms < end_ms:
            batch = self._get_json(
                url,
                {
                    "symbol": self.cfg.get("symbol", "BTCUSDT"),
                    "interval": "1h",
                    "startTime": cursor_ms,
                    "endTime": end_ms,
                    "limit": 1000 if not futures else 1500,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_ms = int(batch[-1][0]) + 3_600_000
            if next_ms <= cursor_ms:
                break
            cursor_ms = next_ms
            time.sleep(0.05)
        now = pd.Timestamp.now(tz="UTC")
        data: list[dict[str, Any]] = []
        for row in rows:
            open_time = pd.to_datetime(
                int(row[0]),
                unit="ms",
                utc=True,
            )
            close_time = pd.to_datetime(
                int(row[6]),
                unit="ms",
                utc=True,
            )
            if close_time >= now or not (start <= open_time < end):
                continue
            data.append(
                {
                    "provider": provider,
                    "symbol": self.cfg.get("symbol", "BTCUSDT"),
                    "open_time": open_time,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_volume": float(row[7]),
                    "trades": float(row[8]),
                    "closed": True,
                    "fetched_at": now,
                }
            )
        return self._normalize(data)

    def _fetch_bybit(self, days: float) -> pd.DataFrame:
        end = pd.Timestamp.now(tz="UTC").floor("h")
        start = end - pd.Timedelta(days=days)
        url = (
            self.cfg["bybit_base_url"].rstrip("/")
            + "/v5/market/kline"
        )
        cursor_end = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)
        raw: list[list[str]] = []
        while cursor_end > start_ms:
            payload = self._get_json(
                url,
                {
                    "category": "linear",
                    "symbol": self.cfg.get("symbol", "BTCUSDT"),
                    "interval": "60",
                    "start": start_ms,
                    "end": cursor_end,
                    "limit": 1000,
                },
            )
            if int(payload.get("retCode", 0)) != 0:
                raise RuntimeError(f"Bybit API error: {payload}")
            batch = payload.get("result", {}).get("list", [])
            if not batch:
                break
            raw.extend(batch)
            oldest = min(int(item[0]) for item in batch)
            if oldest <= start_ms or oldest >= cursor_end:
                break
            cursor_end = oldest - 1
            time.sleep(0.08)
        now = pd.Timestamp.now(tz="UTC")
        data: list[dict[str, Any]] = []
        for row in raw:
            open_time = pd.to_datetime(
                int(row[0]),
                unit="ms",
                utc=True,
            )
            if not (start <= open_time < end):
                continue
            data.append(
                {
                    "provider": "bybit_linear",
                    "symbol": self.cfg.get("symbol", "BTCUSDT"),
                    "open_time": open_time,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_volume": float(row[6]),
                    "trades": None,
                    "closed": True,
                    "fetched_at": now,
                }
            )
        return self._normalize(data)

    def _fetch_okx(self, days: float) -> pd.DataFrame:
        end = pd.Timestamp.now(tz="UTC").floor("h")
        start = end - pd.Timedelta(days=days)
        url = (
            self.cfg["okx_base_url"].rstrip("/")
            + "/api/v5/market/history-candles"
        )
        after: int | None = None
        raw: list[list[str]] = []
        required_pages = int(days * 24 / 100) + 12
        maximum_pages = max(100, required_pages)
        for _ in range(maximum_pages):
            params: dict[str, Any] = {
                "instId": self.cfg.get(
                    "okx_instrument",
                    "BTC-USDT-SWAP",
                ),
                "bar": "1H",
                "limit": "100",
            }
            if after is not None:
                params["after"] = str(after)
            payload = self._get_json(url, params)
            batch = payload.get("data", [])
            if not batch:
                break
            raw.extend(batch)
            oldest = min(int(item[0]) for item in batch)
            if pd.to_datetime(oldest, unit="ms", utc=True) <= start:
                break
            if after == oldest:
                break
            after = oldest
            time.sleep(0.22)
        now = pd.Timestamp.now(tz="UTC")
        data: list[dict[str, Any]] = []
        for row in raw:
            open_time = pd.to_datetime(
                int(row[0]),
                unit="ms",
                utc=True,
            )
            if not (start <= open_time < end) or str(row[8]) != "1":
                continue
            data.append(
                {
                    "provider": "okx_swap",
                    "symbol": self.cfg.get("symbol", "BTCUSDT"),
                    "open_time": open_time,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "quote_volume": float(row[7]),
                    "trades": None,
                    "closed": True,
                    "fetched_at": now,
                }
            )
        return self._normalize(data)

    def _normalize(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()
        return (
            pd.DataFrame(data)
            .drop_duplicates("open_time", keep="last")
            .sort_values("open_time")
            .reset_index(drop=True)
        )

    def _get_json(
        self,
        url: str,
        params: dict[str, Any],
    ) -> Any:
        retries = int(self.cfg.get("max_retries", 5))
        timeout = float(
            self.cfg.get("request_timeout_seconds", 25)
        )
        backoff = float(
            self.cfg.get("retry_backoff_seconds", 1.5)
        )
        last_exception: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if (
                    isinstance(payload, dict)
                    and payload.get("code") not in (None, "0", 0)
                ):
                    raise RuntimeError(f"API error: {payload}")
                return payload
            except Exception as exc:
                last_exception = exc
                if attempt + 1 < retries:
                    time.sleep(backoff * (2**attempt))
        raise RuntimeError(
            f"Request failed {url}: {last_exception}"
        )

    def _validate(
        self,
        frame: pd.DataFrame,
        days: float,
        enforce_minimum_rows: bool = True,
    ) -> None:
        if frame.empty:
            raise ValueError("Provider returned no closed hourly candles")
        if frame["open_time"].duplicated().any():
            raise ValueError("Duplicate candle timestamps")
        times = pd.DatetimeIndex(frame["open_time"])
        gaps = (
            times.to_series()
            .diff()
            .dropna()
            .dt.total_seconds()
            .div(3600)
        )
        maximum_gap = float(gaps.max()) if not gaps.empty else 0.0
        if maximum_gap > float(
            self.cfg.get("maximum_gap_hours", 3)
        ):
            raise ValueError(
                "Candle continuity failed; maximum gap is "
                f"{maximum_gap:.1f} hours"
            )
        expected = int(days * 24 * 0.90)
        configured_minimum = int(
            self.cfg.get("minimum_history_rows", 80000)
        )
        minimum = min(expected, configured_minimum)
        if enforce_minimum_rows and len(frame) < minimum:
            raise ValueError(
                f"Only {len(frame)} candles received; "
                f"expected at least {minimum}"
            )


def fetch_and_store(
    settings: Settings,
    database: Database,
    days: float | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    client = MarketDataClient(settings)
    selected, frame = client.fetch_history(
        days=days,
        provider=provider,
    )
    database.upsert_candles(frame)
    return {
        "provider": selected,
        "rows": len(frame),
        "first": frame["open_time"].min().isoformat(),
        "last": frame["open_time"].max().isoformat(),
    }
