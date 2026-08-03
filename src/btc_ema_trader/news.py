from __future__ import annotations

import hashlib
from datetime import datetime, timezone
import logging
import math
import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
import requests

from .config import Settings
from .storage import Database

LOGGER = logging.getLogger(__name__)

POSITIVE = {
    "adoption": 1.2, "approve": 1.4, "approved": 1.5, "approval": 1.4, "bull": 1.0,
    "bullish": 1.4, "breakout": 1.1, "buy": 0.8, "gain": 0.8, "gains": 0.8,
    "growth": 0.8, "institutional": 0.7, "inflow": 1.0, "launch": 0.6, "record": 0.6,
    "recover": 0.9, "recovery": 0.9, "rally": 1.2, "surge": 1.2, "upgrade": 0.8,
    "win": 0.7, "positive": 0.8, "support": 0.5, "accumulate": 1.0,
}
NEGATIVE = {
    "ban": 1.4, "bear": 1.0, "bearish": 1.4, "crash": 1.7, "decline": 0.8,
    "drop": 0.9, "dump": 1.4, "exploit": 1.6, "fear": 0.9, "fraud": 1.4,
    "hack": 1.6, "hacked": 1.7, "lawsuit": 1.0, "liquidation": 1.0, "loss": 0.8,
    "outflow": 1.0, "reject": 1.3, "rejected": 1.4, "risk": 0.5, "sell": 0.8,
    "slump": 1.0, "warning": 0.8, "negative": 0.8, "probe": 0.6, "investigation": 0.8,
}
RELEVANT = {
    "bitcoin": 2.0, "btc": 2.0, "crypto": 1.0, "cryptocurrency": 1.0,
    "etf": 1.0, "sec": 0.8, "fed": 0.7, "inflation": 0.6, "rate": 0.4,
    "binance": 0.8, "coinbase": 0.8, "okx": 0.7, "mining": 0.7, "halving": 1.0,
}
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


class NewsCollector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.section("news")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "btc-ema-hourly-trader/1.0"})

    def collect_recent(self) -> list[dict[str, Any]]:
        now = pd.Timestamp.now(tz="UTC")
        start = now - pd.Timedelta(hours=float(self.cfg.get("recent_hours", 72)))
        articles: list[dict[str, Any]] = []
        if self.cfg.get("gdelt_enabled", True):
            try:
                articles.extend(self._gdelt_range(start, now, sort="HybridRel"))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Recent GDELT collection failed: %s", exc)
        if self.cfg.get("rss_enabled", True):
            for feed in self.cfg.get("rss_feeds", []):
                try:
                    articles.extend(self._rss_feed(str(feed)))
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("RSS collection failed for %s: %s", feed, exc)
        return deduplicate_articles(articles)

    def backfill(self, days: float | None = None) -> list[dict[str, Any]]:
        days = float(days or self.cfg.get("historical_days", 180))
        end = pd.Timestamp.now(tz="UTC").floor("h")
        start = end - pd.Timedelta(days=days)
        chunk_days = float(self.cfg.get("gdelt_chunk_days", 3))
        cursor = start
        articles: list[dict[str, Any]] = []
        while cursor < end:
            chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end)
            try:
                articles.extend(self._gdelt_range(cursor, chunk_end, sort="DateDesc"))
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("GDELT backfill chunk %s..%s failed: %s", cursor, chunk_end, exc)
            cursor = chunk_end
            time.sleep(0.15)
        return deduplicate_articles(articles)

    def _gdelt_range(self, start: pd.Timestamp, end: pd.Timestamp, sort: str) -> list[dict[str, Any]]:
        query = str(self.cfg.get("gdelt_query", "bitcoin OR btc"))
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": int(self.cfg.get("gdelt_max_records", 250)),
            "format": "json",
            "sort": sort,
            "startdatetime": pd.Timestamp(start).strftime("%Y%m%d%H%M%S"),
            "enddatetime": pd.Timestamp(end).strftime("%Y%m%d%H%M%S"),
        }
        url = "https://api.gdeltproject.org/api/v2/doc/doc?" + "&".join(
            f"{key}={quote_plus(str(value))}" for key, value in params.items()
        )
        response = self.session.get(url, timeout=float(self.cfg.get("request_timeout_seconds", 20)))
        response.raise_for_status()
        payload = response.json()
        now = pd.Timestamp.now(tz="UTC")
        results = []
        for item in payload.get("articles", []):
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            published = _parse_gdelt_date(item.get("seendate")) or now
            sentiment, relevance = score_title(title)
            url_value = str(item.get("url") or "")
            results.append({
                "article_id": _article_id(url_value, title, published),
                "published_at": published,
                "first_seen_at": now,
                "source": str(item.get("domain") or item.get("sourcecountry") or "GDELT"),
                "title": title,
                "url": url_value,
                "sentiment": sentiment,
                "relevance": relevance,
                "metadata": {"origin": "gdelt", "language": item.get("language"), "sourcecountry": item.get("sourcecountry")},
            })
        return results

    def _rss_feed(self, feed_url: str) -> list[dict[str, Any]]:
        response = self.session.get(feed_url, timeout=float(self.cfg.get("request_timeout_seconds", 20)))
        response.raise_for_status()
        root = ET.fromstring(response.content)
        now = pd.Timestamp.now(tz="UTC")
        results = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            raw_date = item.findtext("pubDate") or item.findtext("date")
            if not title:
                continue
            try:
                parsed = pd.Timestamp(parsedate_to_datetime(raw_date)) if raw_date else now
                published = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
            except Exception:
                published = now
            sentiment, relevance = score_title(title)
            results.append({
                "article_id": _article_id(link, title, published), "published_at": published,
                "first_seen_at": now, "source": _domain(feed_url), "title": title, "url": link,
                "sentiment": sentiment, "relevance": relevance, "metadata": {"origin": "rss"},
            })
        return results


def collect_and_store(settings: Settings, database: Database, historical: bool = False, days: float | None = None) -> dict[str, Any]:
    if not settings.section("news").get("enabled", True):
        return {"enabled": False, "articles": 0}
    collector = NewsCollector(settings)
    articles = collector.backfill(days) if historical else collector.collect_recent()
    stored = database.upsert_news(articles)
    return {"enabled": True, "articles": len(articles), "stored": stored, "historical": historical}


def score_title(title: str) -> tuple[float, float]:
    tokens = [token.lower() for token in TOKEN_RE.findall(title)]
    positive = sum(POSITIVE.get(token, 0.0) for token in tokens)
    negative = sum(NEGATIVE.get(token, 0.0) for token in tokens)
    relevance = sum(RELEVANT.get(token, 0.0) for token in tokens)
    raw = positive - negative
    sentiment = math.tanh(raw / 2.4)
    relevance_score = min(1.0, relevance / 3.0)
    return float(sentiment), float(relevance_score)


def deduplicate_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for item in articles:
        article_id = str(item["article_id"])
        current = best.get(article_id)
        if current is None or float(item.get("relevance", 0)) > float(current.get("relevance", 0)):
            best[article_id] = item
    return sorted(best.values(), key=lambda x: pd.Timestamp(x["published_at"]))


def _parse_gdelt_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt), tz="UTC")
        except Exception:
            pass
    try:
        return pd.Timestamp(text, tz="UTC")
    except Exception:
        return None


def _article_id(url: str, title: str, published: pd.Timestamp) -> str:
    raw = f"{url}|{title}|{pd.Timestamp(published).floor('min').isoformat()}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:32]


def _domain(url: str) -> str:
    return url.split("//", 1)[-1].split("/", 1)[0].lower()
