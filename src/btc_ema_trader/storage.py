from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import Settings


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS candles (
    provider TEXT NOT NULL,
    symbol TEXT NOT NULL,
    open_time TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    quote_volume REAL,
    trades REAL,
    closed INTEGER NOT NULL DEFAULT 1,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (provider, symbol, open_time)
);
CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(symbol, open_time);

CREATE TABLE IF NOT EXISTS news (
    article_id TEXT PRIMARY KEY,
    published_at TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    source TEXT,
    title TEXT NOT NULL,
    url TEXT,
    sentiment REAL NOT NULL,
    relevance REAL NOT NULL,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_time ON news(published_at);

CREATE TABLE IF NOT EXISTS signals (
    signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    candle_time TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    price REAL NOT NULL,
    ema20 REAL NOT NULL,
    ema50 REAL NOT NULL,
    cross_direction INTEGER NOT NULL,
    hours_since_cross REAL,
    event_id TEXT,
    event_type TEXT,
    event_direction INTEGER,
    regime TEXT,
    trigger_score REAL,
    forecast_direction TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    tradeability_probability REAL,
    expected_return REAL NOT NULL,
    expected_net_edge_bps REAL NOT NULL,
    selected_horizon INTEGER NOT NULL,
    probabilities_json TEXT NOT NULL,
    tradeability_json TEXT,
    returns_json TEXT NOT NULL,
    trade_plan_json TEXT NOT NULL,
    blockers_json TEXT NOT NULL,
    model_id TEXT NOT NULL,
    qualification_passed INTEGER NOT NULL,
    actual_cost_bps REAL,
    entry_price REAL,
    exit_price REAL,
    resolved INTEGER NOT NULL DEFAULT 0,
    realized_return REAL,
    outcome TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(candle_time);
CREATE INDEX IF NOT EXISTS idx_signals_event ON signals(event_id);

CREATE TABLE IF NOT EXISTS runtime_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT
);
"""


MIGRATION_COLUMNS: dict[str, str] = {
    "event_id": "TEXT",
    "event_type": "TEXT",
    "event_direction": "INTEGER",
    "regime": "TEXT",
    "trigger_score": "REAL",
    "tradeability_probability": "REAL",
    "tradeability_json": "TEXT",
    "actual_cost_bps": "REAL",
    "entry_price": "REAL",
    "exit_price": "REAL",
}


class Database:
    def __init__(self, settings: Settings):
        self.path: Path = settings.path("database")
        self.settings = settings

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
            for name, sql_type in MIGRATION_COLUMNS.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE signals ADD COLUMN {name} {sql_type}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_event ON signals(event_id)")

    def upsert_candles(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        required = {"provider", "symbol", "open_time", "open", "high", "low", "close", "volume", "fetched_at"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing candle columns: {sorted(missing)}")
        rows = []
        for row in frame.to_dict(orient="records"):
            rows.append(
                (
                    str(row["provider"]), str(row["symbol"]), pd.Timestamp(row["open_time"]).isoformat(),
                    float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]),
                    float(row["volume"]), _float_or_none(row.get("quote_volume")), _float_or_none(row.get("trades")),
                    int(bool(row.get("closed", True))), pd.Timestamp(row["fetched_at"]).isoformat(),
                )
            )
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO candles(provider,symbol,open_time,open,high,low,close,volume,quote_volume,trades,closed,fetched_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(provider,symbol,open_time) DO UPDATE SET
                  open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close,
                  volume=excluded.volume, quote_volume=excluded.quote_volume, trades=excluded.trades,
                  closed=excluded.closed, fetched_at=excluded.fetched_at
                """,
                rows,
            )
        return len(rows)

    def load_candles(self, provider: str | None = None, symbol: str = "BTCUSDT", limit: int | None = None) -> pd.DataFrame:
        clauses = ["symbol = ?", "closed = 1"]
        params: list[Any] = [symbol]
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        sql = "SELECT * FROM candles WHERE " + " AND ".join(clauses) + " ORDER BY open_time"
        if limit:
            sql = f"SELECT * FROM ({sql} DESC LIMIT ?) ORDER BY open_time"
            params.append(int(limit))
        with self.connect() as conn:
            frame = pd.read_sql_query(sql, conn, params=params)
        if not frame.empty:
            frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
            frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], utc=True)
        return frame

    def providers(self, symbol: str = "BTCUSDT") -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT provider, COUNT(*) AS rows, MIN(open_time) AS first_time, MAX(open_time) AS last_time "
                "FROM candles WHERE symbol=? AND closed=1 GROUP BY provider ORDER BY rows DESC",
                (symbol,),
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_news(self, articles: Iterable[dict[str, Any]]) -> int:
        rows = []
        for item in articles:
            rows.append(
                (
                    str(item["article_id"]), pd.Timestamp(item["published_at"]).isoformat(),
                    pd.Timestamp(item["first_seen_at"]).isoformat(), str(item.get("source") or ""),
                    str(item["title"]), str(item.get("url") or ""), float(item.get("sentiment", 0.0)),
                    float(item.get("relevance", 0.0)), json.dumps(item.get("metadata", {}), ensure_ascii=False),
                )
            )
        if not rows:
            return 0
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO news(article_id,published_at,first_seen_at,source,title,url,sentiment,relevance,metadata_json)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(article_id) DO UPDATE SET
                  published_at=excluded.published_at, source=excluded.source, title=excluded.title,
                  url=excluded.url, sentiment=excluded.sentiment, relevance=excluded.relevance,
                  metadata_json=excluded.metadata_json
                """,
                rows,
            )
        return len(rows)

    def load_news(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
        clauses, params = [], []
        if start is not None:
            clauses.append("published_at >= ?")
            params.append(pd.Timestamp(start).isoformat())
        if end is not None:
            clauses.append("published_at <= ?")
            params.append(pd.Timestamp(end).isoformat())
        sql = "SELECT * FROM news"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY published_at"
        with self.connect() as conn:
            frame = pd.read_sql_query(sql, conn, params=params)
        if not frame.empty:
            frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
            frame["first_seen_at"] = pd.to_datetime(frame["first_seen_at"], utc=True)
        return frame

    def save_signal(self, payload: dict[str, Any]) -> int:
        # Legacy columns remain populated so an existing v1 database can be migrated in-place.
        legacy_fast = float(payload.get("trend_kama", payload.get("price", 0.0)))
        legacy_slow = float(payload.get("donchian_mid", payload.get("price", 0.0)))
        event_direction = int(payload.get("event_direction", 0))
        bars_since_event = payload.get("bars_since_event")
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals(
                  candle_time,created_at,provider,price,ema20,ema50,cross_direction,hours_since_cross,
                  event_id,event_type,event_direction,regime,trigger_score,
                  forecast_direction,action,confidence,tradeability_probability,expected_return,
                  expected_net_edge_bps,selected_horizon,probabilities_json,tradeability_json,returns_json,
                  trade_plan_json,blockers_json,model_id,qualification_passed,actual_cost_bps
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(candle_time) DO UPDATE SET
                  created_at=excluded.created_at,provider=excluded.provider,price=excluded.price,
                  ema20=excluded.ema20,ema50=excluded.ema50,cross_direction=excluded.cross_direction,
                  hours_since_cross=excluded.hours_since_cross,event_id=excluded.event_id,event_type=excluded.event_type,
                  event_direction=excluded.event_direction,regime=excluded.regime,trigger_score=excluded.trigger_score,
                  forecast_direction=excluded.forecast_direction,action=excluded.action,confidence=excluded.confidence,
                  tradeability_probability=excluded.tradeability_probability,expected_return=excluded.expected_return,
                  expected_net_edge_bps=excluded.expected_net_edge_bps,selected_horizon=excluded.selected_horizon,
                  probabilities_json=excluded.probabilities_json,tradeability_json=excluded.tradeability_json,
                  returns_json=excluded.returns_json,trade_plan_json=excluded.trade_plan_json,
                  blockers_json=excluded.blockers_json,model_id=excluded.model_id,
                  qualification_passed=excluded.qualification_passed,actual_cost_bps=excluded.actual_cost_bps
                """,
                (
                    pd.Timestamp(payload["candle_time"]).isoformat(), pd.Timestamp(payload["created_at"]).isoformat(),
                    payload["provider"], payload["price"], legacy_fast, legacy_slow, event_direction,
                    bars_since_event, payload.get("event_id"), payload.get("event_type", "NONE"), event_direction,
                    payload.get("regime", "UNKNOWN"), payload.get("trigger_score", 0.0),
                    payload["forecast_direction"], payload["action"], payload["confidence"],
                    payload.get("tradeability_probability"), payload["expected_return"],
                    payload["expected_net_edge_bps"], payload["selected_horizon"],
                    json.dumps(payload["probabilities"]), json.dumps(payload.get("tradeability", {})),
                    json.dumps(payload["returns"]), json.dumps(payload["trade_plan"]),
                    json.dumps(payload["blockers"]), payload["model_id"],
                    int(bool(payload["qualification_passed"])), payload.get("actual_cost_bps"),
                ),
            )
            return int(cursor.lastrowid or 0)

    def recent_signals(self, limit: int = 100) -> pd.DataFrame:
        with self.connect() as conn:
            frame = pd.read_sql_query(
                "SELECT * FROM signals ORDER BY candle_time DESC LIMIT ?", conn, params=(int(limit),)
            )
        for col in ("candle_time", "created_at"):
            if col in frame and not frame.empty:
                frame[col] = pd.to_datetime(frame[col], utc=True)
        return frame

    def has_action_for_event(self, event_id: str | None) -> bool:
        if not event_id:
            return False
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM signals WHERE event_id=? AND action IN ('LONG','SHORT') LIMIT 1", (event_id,)
            ).fetchone()
        return row is not None

    def log_event(self, level: str, event_type: str, message: str, details: dict[str, Any] | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO runtime_events(created_at,level,event_type,message,details_json) VALUES(?,?,?,?,?)",
                (pd.Timestamp.now(tz="UTC").isoformat(), level, event_type, message, json.dumps(details or {})),
            )

    def recent_events(self, limit: int = 50) -> pd.DataFrame:
        with self.connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM runtime_events ORDER BY event_id DESC LIMIT ?", conn, params=(int(limit),)
            )

    def resolve_due_signals(self, candles: pd.DataFrame) -> int:
        if candles.empty:
            return 0
        indexed = candles.set_index("open_time")
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT signal_id,candle_time,forecast_direction,selected_horizon,actual_cost_bps "
                "FROM signals WHERE resolved=0 AND action IN ('LONG','SHORT')"
            ).fetchall()
            resolved = 0
            for row in rows:
                signal_open = _utc(row["candle_time"])
                entry_open_time = signal_open + pd.Timedelta(hours=1)
                exit_open_time = signal_open + pd.Timedelta(hours=int(row["selected_horizon"]))
                if entry_open_time not in indexed.index or exit_open_time not in indexed.index:
                    continue
                entry_price = float(indexed.loc[entry_open_time, "open"])
                exit_price = float(indexed.loc[exit_open_time, "close"])
                gross = exit_price / entry_price - 1
                signed = gross if row["forecast_direction"] == "UP" else -gross
                default_cost = float(self.settings.section("strategy").get("fallback_round_trip_cost_bps", 11.0))
                cost_bps = float(row["actual_cost_bps"] if row["actual_cost_bps"] is not None else default_cost)
                net = signed - cost_bps / 10_000
                outcome = "WIN" if net > 0 else "LOSS"
                conn.execute(
                    "UPDATE signals SET resolved=1,entry_price=?,exit_price=?,realized_return=?,outcome=? WHERE signal_id=?",
                    (entry_price, exit_price, net, outcome, int(row["signal_id"])),
                )
                resolved += 1
        return resolved

    def signal_stats_today(self) -> tuple[int, float | None]:
        now = pd.Timestamp.now(tz="UTC")
        start = now.floor("D").isoformat()
        with self.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE created_at >= ? AND action IN ('LONG','SHORT')", (start,)
            ).fetchone()[0]
            row = conn.execute(
                "SELECT MAX(created_at) FROM signals WHERE action IN ('LONG','SHORT')"
            ).fetchone()
        if not row or not row[0]:
            return int(count), None
        last = _utc(row[0])
        return int(count), max(0.0, (now - last).total_seconds() / 3600)


def _float_or_none(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
