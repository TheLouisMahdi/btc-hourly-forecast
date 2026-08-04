from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .negative_memory_core import (
    BloomFilter, boundary_context, fingerprint, _num, _outside, _unavailable_head,
)

@dataclass
class BoundaryHead:
    side: str
    horizon: int
    features: list[str]
    break_model: Any | None
    bad_model: Any | None
    front: BloomFilter
    backup: BloomFilter
    policy: dict[str, float]
    report: dict[str, Any]

    def predict(self, row: pd.DataFrame) -> dict[str, Any]:
        key = fingerprint(row.iloc[-1], self.side, self.horizon)
        if self.break_model is None or self.bad_model is None:
            return _unavailable_head(key)
        x = row.reindex(columns=self.features)
        p_break = float(self.break_model.predict_proba(x)[-1, 1])
        p_bad = float(self.bad_model.predict_proba(x)[-1, 1])
        front_hit = self.front.contains(key)
        backup_hit = self.backup.contains(key)
        minimum_break = float(self.policy["minimum_break_probability"])
        maximum_bad = float(self.policy["maximum_bad_probability"])
        return {
            "available": True,
            "qualified": bool(self.report.get("qualified", False)),
            "p_break": p_break,
            "p_unprofitable": p_bad,
            "front_memory_hit": front_hit,
            "backup_memory_hit": backup_hit,
            "negative_memory_veto": bool(
                front_hit
                or backup_hit
                or p_break < minimum_break
                or p_bad > maximum_bad
            ),
            "policy": dict(self.policy),
            "fingerprint": key,
        }

@dataclass
class SandwichedBoundaryMemory:
    model_id: str
    heads: dict[str, dict[int, BoundaryHead]]
    report: dict[str, Any]
    minimum_distance_atr: float = -0.25
    maximum_distance_atr: float = 0.85
    schema_version: int = 1

    def predict_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return _outside("EMPTY_FRAME")
        latest = frame.iloc[-1]
        event_direction = int(_num(latest.get("event_direction"), 0.0) or 0)
        context = boundary_context(latest, event_direction)
        if (
            context is None
            or float(context["boundary_distance_atr"])
            < self.minimum_distance_atr
            or float(context["boundary_distance_atr"])
            > self.maximum_distance_atr
        ):
            return _outside("OUTSIDE_BOUNDARY_ZONE")
        prepared = latest.to_frame().T.copy()
        for key, value in context.items():
            prepared[key] = value
        side = str(context["boundary_side"])
        outputs = {
            str(horizon): head.predict(prepared)
            for horizon, head in sorted(self.heads.get(side, {}).items())
        }
        if not outputs:
            return {"status": "UNAVAILABLE", **context, "horizons": {}}
        selected = max(
            outputs,
            key=lambda key: outputs[key]["p_break"]
            * (1.0 - outputs[key]["p_unprofitable"]),
        )
        value = outputs[selected]
        return {
            "status": "READY",
            **context,
            "horizons": outputs,
            "selected_horizon": int(selected),
            "p_break": value["p_break"],
            "p_unprofitable": value["p_unprofitable"],
            "negative_memory_veto": value["negative_memory_veto"],
            "qualified": value["qualified"],
        }
