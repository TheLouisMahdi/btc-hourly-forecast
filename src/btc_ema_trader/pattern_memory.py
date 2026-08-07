from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .config import Settings
from .negative_memory_core import BloomFilter

STATIC_PATTERN_SCHEMA_VERSION = 1
LIVE_PATTERN_SCHEMA_VERSION = 1
LONG = 1
SHORT = -1

CONTEXT_NEUTRAL = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    dtype=float,
)


@dataclass
class PatternHead:
    bloom: BloomFilter
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def assess(self, key: str) -> dict[str, Any]:
        item = self.stats.get(key, {})
        count = int(item.get("count", 0))
        bad = int(item.get("bad", 0))
        bad_rate = bad / count if count else 0.0
        return {
            "fingerprint": key,
            "bloom_hit": bool(count and self.bloom.contains(key)),
            "count": count,
            "bad": bad,
            "bad_rate": float(bad_rate),
        }


@dataclass
class StaticPatternBundle:
    schema_version: int
    model_id: str
    created_at: str
    heads: dict[str, dict[int, PatternHead]]
    report: dict[str, Any]

    def assess_record(
        self,
        record: dict[str, Any],
        *,
        horizon: int,
    ) -> dict[str, Any]:
        direction = event_direction(record)
        if direction not in {LONG, SHORT}:
            return {
                "available": False,
                "direction": "NONE",
                "horizon": int(horizon),
                "bloom_hit": False,
                "count": 0,
                "bad_rate": 0.0,
            }
        name = "LONG" if direction == LONG else "SHORT"
        key = event_pattern_key_from_record(record, direction, int(horizon))
        head = self.heads.get(name, {}).get(int(horizon))
        if head is None:
            return {
                "available": False,
                "direction": name,
                "horizon": int(horizon),
                "fingerprint": key,
                "bloom_hit": False,
                "count": 0,
                "bad_rate": 0.0,
            }
        return {
            "available": True,
            "direction": name,
            "direction_code": direction,
            "horizon": int(horizon),
            **head.assess(key),
        }


@dataclass
class LivePatternState:
    schema_version: int
    created_at: str
    updated_at: str
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    learned_forecasts: set[str] = field(default_factory=set)


class LiveCandlePatternMemory:
    """Online memory of resolved next-candle mistakes.

    Exact counts guard every decision; the memory never flips a direction by
    itself and only shrinks confidence on repeatedly bad candle contexts.
    """

    def __init__(self, settings: Settings, path: Path) -> None:
        self.settings = settings
        self.cfg = settings.section("trade_assistant")
        self.path = path
        self.state = self._load_or_create()

    def _load_or_create(self) -> LivePatternState:
        if self.path.exists():
            try:
                state = joblib.load(self.path)
                if (
                    isinstance(state, LivePatternState)
                    and state.schema_version == LIVE_PATTERN_SCHEMA_VERSION
                ):
                    return state
            except Exception:
                pass
        now = pd.Timestamp.now(tz="UTC").isoformat()
        return LivePatternState(
            schema_version=LIVE_PATTERN_SCHEMA_VERSION,
            created_at=now,
            updated_at=now,
        )

    def synchronize(self, history: Iterable[dict[str, Any]]) -> int:
        learned = 0
        for item in history:
            result = str(item.get("direction_result") or "")
            if result not in {"DIRECTION_CORRECT", "DIRECTION_WRONG"}:
                continue
            forecast_id = _forecast_id(item)
            if not forecast_id or forecast_id in self.state.learned_forecasts:
                continue
            direction = forecast_direction(item)
            key = forecast_pattern_key(item, direction)
            bucket = self.state.stats.setdefault(
                key,
                {"count": 0, "wrong": 0, "last_seen": None},
            )
            bucket["count"] = int(bucket.get("count", 0)) + 1
            if result == "DIRECTION_WRONG":
                bucket["wrong"] = int(bucket.get("wrong", 0)) + 1
            bucket["last_seen"] = str(
                item.get("resolved_at") or item.get("run_finished_at") or ""
            )
            self.state.learned_forecasts.add(forecast_id)
            learned += 1
        if learned:
            self.state.updated_at = pd.Timestamp.now(tz="UTC").isoformat()
            self.save()
        return learned

    def assess(
        self,
        record: dict[str, Any],
        *,
        direction: str,
    ) -> dict[str, Any]:
        key = forecast_pattern_key(record, direction)
        bucket = self.state.stats.get(key, {})
        count = int(bucket.get("count", 0))
        wrong = int(bucket.get("wrong", 0))
        wrong_rate = wrong / count if count else 0.0
        minimum_count = int(self.cfg.get("live_pattern_minimum_count", 3))
        threshold = float(self.cfg.get("live_pattern_bad_rate", 0.67))
        return {
            "fingerprint": key,
            "count": count,
            "wrong": wrong,
            "wrong_rate": float(wrong_rate),
            "bad_pattern": bool(
                count >= minimum_count and wrong_rate >= threshold
            ),
            "minimum_count": minimum_count,
            "bad_rate_threshold": threshold,
        }

    def summary(self) -> dict[str, Any]:
        bad = 0
        threshold = float(self.cfg.get("live_pattern_bad_rate", 0.67))
        minimum = int(self.cfg.get("live_pattern_minimum_count", 3))
        for item in self.state.stats.values():
            count = int(item.get("count", 0))
            wrong = int(item.get("wrong", 0))
            if count >= minimum and wrong / max(count, 1) >= threshold:
                bad += 1
        return {
            "schema_version": self.state.schema_version,
            "patterns": len(self.state.stats),
            "bad_patterns": bad,
            "learned_forecasts": len(self.state.learned_forecasts),
            "updated_at": self.state.updated_at,
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        joblib.dump(self.state, temporary)
        temporary.replace(self.path)


def build_static_pattern_bundle(
    frame: pd.DataFrame,
    settings: Settings,
    *,
    model_id: str,
) -> tuple[StaticPatternBundle, dict[str, Any]]:
    cfg = settings.section("trade_assistant")
    data = frame.copy().sort_values("open_time").reset_index(drop=True)
    trade_horizons = [
        int(value)
        for value in settings.section("model").get(
            "trade_horizons_hours", [3, 6, 12]
        )
    ]
    horizons = sorted(set([1, *trade_horizons]))
    heads: dict[str, dict[int, PatternHead]] = {"LONG": {}, "SHORT": {}}
    report_heads: dict[str, dict[str, Any]] = {"LONG": {}, "SHORT": {}}
    minimum_count = int(cfg.get("static_pattern_minimum_count", 2))
    minimum_bad_rate = float(cfg.get("static_pattern_bad_rate", 0.75))
    fpr = float(cfg.get("bloom_false_positive_rate", 0.005))

    for direction, name in ((LONG, "LONG"), (SHORT, "SHORT")):
        direction_rows = data.loc[data["event_direction"] == direction]
        for horizon in horizons:
            false_column = f"false_breakout_h{horizon}"
            if false_column not in data:
                rows = direction_rows.iloc[0:0]
            else:
                rows = direction_rows.dropna(subset=[false_column])
            stats: dict[str, dict[str, float]] = {}
            for index, row in rows.iterrows():
                key = event_pattern_key_from_frame(
                    data,
                    int(index),
                    direction,
                    horizon,
                )
                item = stats.setdefault(key, {"count": 0.0, "bad": 0.0})
                item["count"] += 1.0
                item["bad"] += float(
                    _number(row.get(false_column), 0.0) >= 0.5
                )
            bad_keys = [
                key
                for key, item in stats.items()
                if int(item["count"]) >= minimum_count
                and item["bad"] / max(item["count"], 1.0) >= minimum_bad_rate
            ]
            bloom = BloomFilter.create(max(1, len(bad_keys)), fpr)
            for key in bad_keys:
                bloom.add(key)
            heads[name][horizon] = PatternHead(bloom=bloom, stats=stats)
            report_heads[name][str(horizon)] = {
                "patterns": len(stats),
                "bad_patterns": len(bad_keys),
                "minimum_count": minimum_count,
                "minimum_bad_rate": minimum_bad_rate,
            }

    report = {
        "schema_version": STATIC_PATTERN_SCHEMA_VERSION,
        "model_id": model_id,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": "CROSS_LAYER_FALSE_BREAKOUT_BLOOM_WITH_EXACT_COUNTS",
        "horizons": horizons,
        "heads": report_heads,
    }
    bundle = StaticPatternBundle(
        schema_version=STATIC_PATTERN_SCHEMA_VERSION,
        model_id=model_id,
        created_at=report["created_at"],
        heads=heads,
        report=report,
    )
    return bundle, report


def save_static_pattern_bundle(bundle: StaticPatternBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(bundle, temporary)
    temporary.replace(path)


def load_static_pattern_bundle(path: Path) -> StaticPatternBundle:
    value = joblib.load(path)
    if not isinstance(value, StaticPatternBundle):
        raise TypeError(f"Unexpected static pattern artifact: {type(value)!r}")
    if value.schema_version != STATIC_PATTERN_SCHEMA_VERSION:
        raise RuntimeError("Unsupported static pattern artifact schema")
    return value


def adjust_forecast_with_pattern_memory(
    prediction: dict[str, Any],
    *,
    record: dict[str, Any],
    static: dict[str, Any] | None,
    live: dict[str, Any] | None,
    settings: Settings,
) -> dict[str, Any]:
    output = dict(prediction)
    probability = float(np.clip(
        _number(output.get("fused_probability_up"), 0.5),
        0.05,
        0.95,
    ))
    predicted_direction = "UP" if probability >= 0.5 else "DOWN"
    cfg = settings.section("trade_assistant")

    static_strength = 0.0
    if isinstance(static, dict) and static.get("available", False):
        event_side = (
            "UP"
            if int(static.get("direction_code", 0)) == LONG
            else "DOWN"
        )
        if (
            predicted_direction == event_side
            and static.get("bloom_hit", False)
            and int(static.get("count", 0))
            >= int(cfg.get("static_pattern_minimum_count", 2))
            and _number(static.get("bad_rate"), 0.0)
            >= float(cfg.get("static_pattern_bad_rate", 0.75))
        ):
            static_strength = float(
                np.clip(
                    (_number(static.get("bad_rate"), 0.0) - 0.5)
                    * float(cfg.get("static_forecast_penalty_gain", 0.25)),
                    0.0,
                    float(cfg.get("maximum_static_forecast_penalty", 0.08)),
                )
            )

    live_strength = 0.0
    if isinstance(live, dict) and live.get("bad_pattern", False):
        live_strength = float(
            np.clip(
                (_number(live.get("wrong_rate"), 0.0) - 0.5)
                * float(cfg.get("live_forecast_penalty_gain", 0.30)),
                0.0,
                float(cfg.get("maximum_live_forecast_penalty", 0.10)),
            )
        )

    shrink = float(np.clip(static_strength + live_strength, 0.0, 0.15))
    adjusted = 0.5 + (probability - 0.5) * (1.0 - shrink)
    output["fused_probability_up"] = float(np.clip(adjusted, 0.05, 0.95))
    output["pattern_memory_adjustment"] = {
        "applied": bool(shrink > 0),
        "confidence_shrink_fraction": shrink,
        "static": static,
        "live": live,
        "direction_preserved": (
            (probability >= 0.5) == (adjusted >= 0.5)
        ),
    }
    return output


def event_pattern_key_from_record(
    record: dict[str, Any],
    direction: int,
    horizon: int,
) -> str:
    context = context_vector_from_record(record)
    return _event_key(
        direction=direction,
        horizon=horizon,
        event_scale=_number(record.get("event_scale_hours"), 0.0),
        event_score=_number(
            record.get("trigger_score", record.get("event_score")),
            0.0,
        ),
        context=context,
    )


def event_pattern_key_from_frame(
    frame: pd.DataFrame,
    index: int,
    direction: int,
    horizon: int,
) -> str:
    row = frame.iloc[index]
    return _event_key(
        direction=direction,
        horizon=horizon,
        event_scale=_number(row.get("event_scale_hours"), 0.0),
        event_score=_number(row.get("event_score"), 0.0),
        context=context_vector_from_frame(frame, index),
    )


def forecast_pattern_key(
    record: dict[str, Any],
    direction: str | None = None,
) -> str:
    chosen = str(direction or forecast_direction(record)).upper()
    context = context_vector_from_record(record)
    values = (
        "FORECAST",
        chosen,
        str(record.get("regime") or "UNKNOWN"),
        *(_bucket(value, 0.25, -3.0, 3.0) for value in (
            context[0], context[1], context[5], context[7],
            context[9], context[10],
        )),
        *(_bucket(value, 0.20, 0.0, 1.0) for value in (
            context[2], context[3], context[4], context[6], context[8],
        )),
        _bucket(context[11], 0.20, -1.0, 1.0),
    )
    return _hash(values)


def context_vector_from_record(record: dict[str, Any]) -> np.ndarray:
    context = record.get("event_candle_context")
    context = context if isinstance(context, dict) else {}
    bars = context.get("bars")
    bars = bars if isinstance(bars, list) else []
    by_role = {
        str(item.get("role")): item
        for item in bars
        if isinstance(item, dict)
    }
    return _context_vector(
        by_role.get("PREVIOUS_2", {}),
        by_role.get("PREVIOUS_1", {}),
        by_role.get("EVENT", {}),
    )


def context_vector_from_frame(frame: pd.DataFrame, index: int) -> np.ndarray:
    if index < 2:
        return CONTEXT_NEUTRAL.copy()
    bars: list[dict[str, float]] = []
    for offset in (-2, -1, 0):
        row = frame.iloc[index + offset]
        open_price = _number(row.get("open"), 0.0)
        high = _number(row.get("high"), open_price)
        low = _number(row.get("low"), open_price)
        close = _number(row.get("close"), open_price)
        span = max(high - low, 0.0)
        bars.append(
            {
                "open": open_price,
                "close": close,
                "volume": _number(row.get("volume"), 0.0),
                "body_percent": (
                    (close - open_price) / open_price
                    if open_price > 0
                    else 0.0
                ),
                "range_percent": (
                    span / open_price if open_price > 0 else 0.0
                ),
                "range": span,
                "upper_wick": max(high - max(open_price, close), 0.0),
                "lower_wick": max(min(open_price, close) - low, 0.0),
                "close_location": (
                    (close - low) / span if span > 0 else 0.5
                ),
            }
        )
    return _context_vector(bars[0], bars[1], bars[2])


def event_direction(record: dict[str, Any]) -> int:
    raw = int(_number(record.get("event_direction"), 0.0))
    if raw in {LONG, SHORT}:
        return raw
    action = str(
        record.get("candidate_action")
        or record.get("action")
        or ""
    ).upper()
    return LONG if action == "LONG" else SHORT if action == "SHORT" else 0


def forecast_direction(record: dict[str, Any]) -> str:
    contract = record.get("next_candle_forecast")
    contract = contract if isinstance(contract, dict) else {}
    direction = str(
        contract.get("direction")
        or record.get("next_candle_direction")
        or record.get("forecast_direction")
        or "NONE"
    ).upper()
    return direction if direction in {"UP", "DOWN"} else "NONE"


def _event_key(
    *,
    direction: int,
    horizon: int,
    event_scale: float,
    event_score: float,
    context: np.ndarray,
) -> str:
    values = (
        "EVENT",
        int(direction),
        int(horizon),
        _bucket(event_scale, 120.0, 0.0, 720.0),
        _bucket(event_score, 0.10, 0.0, 1.0),
        *(_bucket(value, 0.25, -3.0, 3.0) for value in (
            context[0], context[1], context[5], context[7],
            context[9], context[10],
        )),
        *(_bucket(value, 0.20, 0.0, 1.0) for value in (
            context[2], context[3], context[4], context[6], context[8],
        )),
        _bucket(context[11], 0.20, -1.0, 1.0),
    )
    return _hash(values)


def _context_vector(
    previous_2: dict[str, Any],
    previous_1: dict[str, Any],
    event: dict[str, Any],
) -> np.ndarray:
    event_range = max(_number(event.get("range"), 0.0), 0.0)
    upper_share = (
        max(_number(event.get("upper_wick"), 0.0), 0.0) / event_range
        if event_range > 0
        else 0.0
    )
    lower_share = (
        max(_number(event.get("lower_wick"), 0.0), 0.0) / event_range
        if event_range > 0
        else 0.0
    )
    first_open = _number(previous_2.get("open"), 0.0)
    event_close = _number(event.get("close"), 0.0)
    three_bar_return = (
        event_close / first_open - 1.0
        if first_open > 0 and event_close > 0
        else 0.0
    )
    previous_volumes = [
        _number(item.get("volume"), 0.0)
        for item in (previous_2, previous_1)
        if _number(item.get("volume"), 0.0) > 0
    ]
    previous_volume = (
        sum(previous_volumes) / len(previous_volumes)
        if previous_volumes
        else 0.0
    )
    event_volume = _number(event.get("volume"), 0.0)
    volume_log_ratio = (
        math.log1p(event_volume) - math.log1p(previous_volume)
        if event_volume > 0 and previous_volume > 0
        else 0.0
    )
    upper_total = sum(
        max(_number(item.get("upper_wick"), 0.0), 0.0)
        for item in (previous_2, previous_1, event)
    )
    lower_total = sum(
        max(_number(item.get("lower_wick"), 0.0), 0.0)
        for item in (previous_2, previous_1, event)
    )
    range_total = sum(
        max(_number(item.get("range"), 0.0), 0.0)
        for item in (previous_2, previous_1, event)
    )
    wick_pressure = (
        (lower_total - upper_total) / range_total
        if range_total > 0
        else 0.0
    )
    return np.asarray(
        [
            _scaled(event.get("body_percent"), 0.02),
            _scaled(event.get("range_percent"), 0.03),
            float(np.clip(upper_share, 0.0, 1.0)),
            float(np.clip(lower_share, 0.0, 1.0)),
            float(np.clip(
                _number(event.get("close_location"), 0.5), 0.0, 1.0
            )),
            _scaled(previous_1.get("body_percent"), 0.02),
            float(np.clip(
                _number(previous_1.get("close_location"), 0.5), 0.0, 1.0
            )),
            _scaled(previous_2.get("body_percent"), 0.02),
            float(np.clip(
                _number(previous_2.get("close_location"), 0.5), 0.0, 1.0
            )),
            float(np.clip(three_bar_return / 0.04, -3.0, 3.0)),
            float(np.clip(volume_log_ratio / 2.0, -3.0, 3.0)),
            float(np.clip(wick_pressure, -1.0, 1.0)),
        ],
        dtype=float,
    )


def _forecast_id(item: dict[str, Any]) -> str:
    contract = item.get("next_candle_forecast")
    contract = contract if isinstance(contract, dict) else {}
    return str(
        contract.get("source_open_time")
        or item.get("candle_time")
        or item.get("run_finished_at")
        or ""
    )


def _scaled(value: Any, scale: float) -> float:
    return float(np.clip(_number(value, 0.0) / max(scale, 1e-9), -3.0, 3.0))


def _bucket(value: Any, step: float, low: float, high: float) -> int:
    number = float(np.clip(_number(value, 0.0), low, high))
    return int(round((number - low) / max(step, 1e-9)))


def _hash(values: Iterable[Any]) -> str:
    payload = "|".join(str(value) for value in values)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)
