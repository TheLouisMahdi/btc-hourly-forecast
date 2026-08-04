from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

SUPPORT = "SUPPORT"
RESISTANCE = "RESISTANCE"

@dataclass
class BloomFilter:
    bit_count: int
    hash_count: int
    bits: bytearray

    @classmethod
    def create(
        cls,
        capacity: int,
        false_positive_rate: float = 0.005,
    ) -> "BloomFilter":
        n = max(1, int(capacity))
        p = float(np.clip(false_positive_rate, 1e-6, 0.25))
        m = max(64, int(math.ceil(-n * math.log(p) / math.log(2) ** 2)))
        k = max(1, min(16, int(round(m / n * math.log(2)))))
        return cls(m, k, bytearray((m + 7) // 8))

    def add(self, key: str) -> None:
        for position in self._positions(key):
            self.bits[position // 8] |= 1 << (position % 8)

    def contains(self, key: str) -> bool:
        return all(
            self.bits[position // 8] & (1 << (position % 8))
            for position in self._positions(key)
        )

    def _positions(self, key: str) -> Iterable[int]:
        payload = key.encode("utf-8")
        for seed in range(self.hash_count):
            person = f"btc{seed:013d}".encode("ascii")
            digest = hashlib.blake2b(
                payload, digest_size=8, person=person
            ).digest()
            yield int.from_bytes(digest, "little") % self.bit_count

def boundary_context(
    row: pd.Series,
    event_direction: int = 0,
) -> dict[str, Any] | None:
    approach = _num(row.get("return_6"), 0.0) or 0.0
    if event_direction > 0:
        side = RESISTANCE
    elif event_direction < 0:
        side = SUPPORT
    else:
        choices: list[tuple[float, str]] = []
        support_distance = _num(row.get("distance_to_support_atr"))
        resistance_distance = _num(row.get("distance_to_resistance_atr"))
        if support_distance is not None and support_distance >= -0.25 and approach <= 0:
            choices.append((support_distance, SUPPORT))
        if resistance_distance is not None and resistance_distance >= -0.25 and approach >= 0:
            choices.append((resistance_distance, RESISTANCE))
        if not choices:
            return None
        side = min(choices)[1]
    prefix = "support" if side == SUPPORT else "resistance"
    level = _num(row.get(f"structure_{prefix}"))
    distance = _num(row.get(f"distance_to_{prefix}_atr"))
    if level is None or distance is None:
        return None
    return {
        "boundary_side": side,
        "boundary_side_code": -1 if side == SUPPORT else 1,
        "boundary_level": level,
        "boundary_distance_atr": distance,
        "boundary_strength": _num(row.get(f"{prefix}_strength"), 0.0) or 0.0,
        "boundary_age_bars": _num(row.get(f"{prefix}_age_bars"), 0.0) or 0.0,
        "boundary_approach_return_6": approach,
    }

def fingerprint(row: pd.Series, side: str, horizon: int) -> str:
    values = (
        side,
        int(horizon),
        _bucket(row.get("boundary_distance_atr"), 0.10, -0.5, 3.0),
        _bucket(row.get("boundary_strength"), 0.50, 0.0, 12.0),
        _log_bucket(row.get("boundary_age_bars")),
        _bucket(row.get("return_1"), 0.0025, -0.05, 0.05),
        _bucket(row.get("return_3"), 0.004, -0.10, 0.10),
        _bucket(row.get("return_6"), 0.006, -0.15, 0.15),
        _bucket(row.get("volume_z_24"), 0.50, -4.0, 6.0),
        _bucket(row.get("volume_z_72"), 0.50, -4.0, 6.0),
        _bucket(row.get("atr_percentile_168"), 0.10, 0.0, 1.0),
        _bucket(row.get("rsi_centered"), 0.10, -1.0, 1.0),
        _bucket(row.get("adx"), 5.0, 0.0, 80.0),
        int(_num(row.get("regime_code"), 0.0) or 0),
        int(_num(row.get("triangle_code"), 0.0) or 0),
    )
    return "|".join(str(value) for value in values)

def _outside(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "boundary_side": "NONE",
        "horizons": {},
        "selected_horizon": None,
        "negative_memory_veto": False,
        "qualified": False,
    }

def _unavailable_head(key: str) -> dict[str, Any]:
    return {
        "available": False,
        "qualified": False,
        "p_break": 0.5,
        "p_unprofitable": 1.0,
        "front_memory_hit": False,
        "backup_memory_hit": False,
        "negative_memory_veto": True,
        "policy": {},
        "fingerprint": key,
    }

def _first(values: np.ndarray) -> int | None:
    positions = np.flatnonzero(values)
    return int(positions[0]) if len(positions) else None

def _per_horizon(value: Any, horizon: int, default: float) -> float:
    if isinstance(value, dict):
        return float(value.get(horizon, value.get(str(horizon), default)))
    return float(default if value is None else value)

def _bucket(value: Any, step: float, low: float, high: float) -> int:
    number = float(np.clip(_num(value, 0.0) or 0.0, low, high))
    return int(round((number - low) / step))

def _log_bucket(value: Any) -> int:
    return int(min(12, math.floor(math.log2(max(0.0, _num(value, 0.0) or 0.0) + 1))))

def _num(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default
