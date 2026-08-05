from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import Settings
from .negative_memory_core import (
    SUPPORT,
    RESISTANCE,
    BloomFilter,
    boundary_context,
    fingerprint,
    _num,
    _outside,
)
from .negative_memory_dataset import mine_boundary_encounters
from .negative_memory_model import BoundaryHead, SandwichedBoundaryMemory
from .negative_memory_training import train_sandwiched_boundary_memory

__all__ = [
    "SUPPORT",
    "RESISTANCE",
    "BloomFilter",
    "BoundaryHead",
    "SandwichedBoundaryMemory",
    "train_sandwiched_boundary_memory",
    "mine_boundary_encounters",
    "save_boundary_memory",
    "load_boundary_memory",
    "install_runtime_guard",
    "boundary_context",
    "fingerprint",
]


def save_boundary_memory(memory: SandwichedBoundaryMemory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(memory, temporary)
    temporary.replace(path)


def load_boundary_memory(path: Path) -> SandwichedBoundaryMemory:
    value = joblib.load(path)
    if not isinstance(value, SandwichedBoundaryMemory):
        raise TypeError(f"Unexpected negative-memory artifact: {type(value)!r}")
    if int(getattr(value, "schema_version", 0)) != 1:
        raise RuntimeError("Unsupported sandwiched negative-memory schema")
    return value


def install_runtime_guard(
    memory: SandwichedBoundaryMemory | None,
    *,
    require_for_trade: bool = True,
) -> None:
    """Install memory as a configured hard veto or paper-risk penalty."""
    from . import runtime as runtime_module
    from .model import HourlyModelBundle

    if not getattr(HourlyModelBundle.predict_frame, "_negative_memory_guard", False):
        original_predict = HourlyModelBundle.predict_frame

        def guarded_predict(self, frame: pd.DataFrame) -> dict[str, Any]:
            output = original_predict(self, frame)
            active = getattr(guarded_predict, "_active_memory", None)
            if active is None:
                boundary = _outside("UNAVAILABLE")
            elif active.model_id != getattr(self, "model_id", None):
                boundary = _outside("MODEL_ID_MISMATCH")
            else:
                boundary = active.predict_frame(frame)
            output["boundary_memory"] = boundary
            return output

        guarded_predict._negative_memory_guard = True
        guarded_predict._active_memory = memory
        HourlyModelBundle.predict_frame = guarded_predict
    else:
        HourlyModelBundle.predict_frame._active_memory = memory

    if getattr(runtime_module.make_decision, "_negative_memory_guard", False):
        return
    original_decision = runtime_module.make_decision

    def guarded_decision(
        latest_row: pd.Series,
        prediction: dict[str, Any],
        bundle: Any,
        settings: Settings,
        **kwargs: Any,
    ) -> Any:
        decision = original_decision(
            latest_row,
            prediction,
            bundle,
            settings,
            **kwargs,
        )
        boundary = prediction.get("boundary_memory")
        boundary = (
            boundary
            if isinstance(boundary, dict)
            else _outside("UNAVAILABLE")
        )
        direction = int(_num(latest_row.get("event_direction"), 0.0) or 0)
        flags: list[str] = []
        if direction:
            expected = RESISTANCE if direction > 0 else SUPPORT
            if boundary.get("status") != "READY":
                if require_for_trade:
                    flags.append("BOUNDARY_NEGATIVE_MEMORY_UNAVAILABLE")
            elif boundary.get("boundary_side") != expected:
                flags.append("BOUNDARY_CONTEXT_MISMATCH")
            else:
                item = boundary.get("horizons", {}).get(
                    str(decision.selected_horizon)
                )
                if not isinstance(item, dict):
                    flags.append("BOUNDARY_HORIZON_UNAVAILABLE")
                else:
                    policy = item.get("policy", {})
                    if not item.get("qualified", False):
                        flags.append("BOUNDARY_HEAD_NOT_QUALIFIED")
                    if item.get("front_memory_hit", False):
                        flags.append("KNOWN_BAD_PATTERN_FRONT_BLOOM")
                    if item.get("backup_memory_hit", False):
                        flags.append("HARD_NEGATIVE_BACKUP_BLOOM")
                    if float(item.get("p_break", 0.0)) < float(
                        policy.get("minimum_break_probability", 0.60)
                    ):
                        flags.append("LOW_LEVEL_BREAK_PROBABILITY")
                    if float(item.get("p_unprofitable", 1.0)) > float(
                        policy.get("maximum_bad_probability", 0.45)
                    ):
                        flags.append("HIGH_UNPROFITABLE_PATTERN_RISK")

        flags = list(dict.fromkeys(flags))
        plan = dict(decision.trade_plan)
        plan["boundary_memory"] = boundary
        plan["boundary_memory_flags"] = flags

        memory_cfg = settings.section("negative_memory")
        mode = str(
            memory_cfg.get("runtime_mode", "HARD_VETO")
        ).upper()
        if mode == "ADAPTIVE_PENALTY_ONLY":
            plan = _apply_boundary_risk_penalty(
                plan,
                flags,
                settings,
            )
            plan["negative_memory_mode"] = mode
            return replace(decision, trade_plan=plan)

        blockers = list(dict.fromkeys(list(decision.blockers) + flags))
        plan["negative_memory_mode"] = "HARD_VETO"
        return replace(
            decision,
            action="WAIT" if blockers else decision.action,
            blockers=blockers,
            trade_plan=plan,
        )

    guarded_decision._negative_memory_guard = True
    runtime_module.make_decision = guarded_decision


def _apply_boundary_risk_penalty(
    plan: dict[str, Any],
    flags: list[str],
    settings: Settings,
) -> dict[str, Any]:
    """Convert negative-memory evidence into a bounded paper-risk reduction."""
    output = dict(plan)
    factors = {
        "BOUNDARY_NEGATIVE_MEMORY_UNAVAILABLE": 0.90,
        "BOUNDARY_CONTEXT_MISMATCH": 0.80,
        "BOUNDARY_HORIZON_UNAVAILABLE": 0.88,
        "BOUNDARY_HEAD_NOT_QUALIFIED": 0.90,
        "KNOWN_BAD_PATTERN_FRONT_BLOOM": 0.75,
        "HARD_NEGATIVE_BACKUP_BLOOM": 0.65,
        "LOW_LEVEL_BREAK_PROBABILITY": 0.85,
        "HIGH_UNPROFITABLE_PATTERN_RISK": 0.80,
    }
    multiplier = 1.0
    applied: dict[str, float] = {}
    for flag in flags:
        factor = factors.get(flag)
        if factor is None:
            continue
        multiplier *= factor
        applied[flag] = factor

    strategy = settings.section("strategy")
    minimum = float(
        strategy.get("minimum_risk_per_trade_fraction", 0.005)
    )
    maximum = float(
        strategy.get("maximum_risk_per_trade_fraction", 0.03)
    )
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    original_fraction = float(
        output.get(
            "risk_fraction",
            strategy.get("risk_per_trade_fraction", 0.0125),
        )
    )
    adjusted_fraction = float(
        np.clip(original_fraction * multiplier, minimum, maximum)
    )
    account = float(strategy.get("account_equity_usd", 1000.0))

    assessment = output.get("risk_assessment")
    assessment = dict(assessment) if isinstance(assessment, dict) else {}
    assessment["negative_memory_multiplier"] = float(multiplier)
    assessment["negative_memory_penalties"] = applied
    assessment["pre_memory_risk_fraction"] = original_fraction
    assessment["risk_fraction"] = adjusted_fraction

    soft_flags = list(output.get("soft_risk_flags", []))
    output["soft_risk_flags"] = list(dict.fromkeys(soft_flags + flags))
    output["ignored_soft_blockers"] = list(output["soft_risk_flags"])
    output["risk_assessment"] = assessment
    output["risk_fraction"] = adjusted_fraction
    output["risk_budget_usd"] = account * adjusted_fraction
    output["negative_memory_risk_multiplier"] = float(multiplier)
    output["negative_memory_penalties"] = applied
    return output
