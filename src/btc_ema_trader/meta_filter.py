from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import Settings
from .pattern_memory import (
    LONG,
    SHORT,
    StaticPatternBundle,
    context_vector_from_frame,
    context_vector_from_record,
    event_direction,
)

META_FILTER_SCHEMA_VERSION = 1
META_FEATURES = (
    "primary_continuation",
    "primary_tradeability",
    "primary_predicted_return_scaled",
    "general_direction_confidence",
    "event_score",
    "event_scale_scaled",
    "event_body_scaled",
    "event_range_scaled",
    "event_upper_wick_share",
    "event_lower_wick_share",
    "event_close_location",
    "previous_1_body_scaled",
    "previous_1_close_location",
    "previous_2_body_scaled",
    "previous_2_close_location",
    "three_bar_net_return_scaled",
    "event_volume_vs_previous_log",
    "three_bar_wick_pressure",
)


@dataclass
class MetaHead:
    direction: int
    horizon: int
    take_model: Any | None
    false_model: Any | None
    minimum_take_probability: float
    maximum_false_probability: float
    qualified: bool
    report: dict[str, Any]

    def predict(self, vector: np.ndarray) -> dict[str, Any]:
        if (
            self.take_model is None
            or self.false_model is None
            or vector.shape != (len(META_FEATURES),)
            or not np.isfinite(vector).all()
        ):
            return {
                "available": False,
                "qualified": False,
                "p_take": 0.0,
                "p_false": 1.0,
                "selected": False,
                "reason": "META_HEAD_UNAVAILABLE",
            }
        matrix = vector.reshape(1, -1)
        p_take = float(
            np.clip(self.take_model.predict_proba(matrix)[0, 1], 0.0, 1.0)
        )
        p_false = float(
            np.clip(self.false_model.predict_proba(matrix)[0, 1], 0.0, 1.0)
        )
        selected = bool(
            self.qualified
            and p_take >= self.minimum_take_probability
            and p_false <= self.maximum_false_probability
        )
        return {
            "available": True,
            "qualified": bool(self.qualified),
            "p_take": p_take,
            "p_false": p_false,
            "minimum_take_probability": float(self.minimum_take_probability),
            "maximum_false_probability": float(self.maximum_false_probability),
            "selected": selected,
            "reason": (
                "META_GATE_ACCEPTED"
                if selected
                else "META_HEAD_NOT_QUALIFIED"
                if not self.qualified
                else "META_PRECISION_GATE_REJECTED"
            ),
        }


@dataclass
class PrecisionMetaFilter:
    schema_version: int
    model_id: str
    created_at: str
    heads: dict[str, dict[int, MetaHead]]
    exit_profiles: dict[str, dict[int, dict[str, float]]]
    report: dict[str, Any]

    def assess(
        self,
        record: dict[str, Any],
        patterns: StaticPatternBundle | None,
    ) -> dict[str, Any]:
        direction = event_direction(record)
        horizon = int(_number(record.get("selected_horizon"), 0.0))
        if direction not in {LONG, SHORT} or horizon <= 0:
            return {
                "status": "NO_EVENT",
                "qualified": False,
                "selected": False,
                "reason": "NO_DIRECTIONAL_EVENT",
            }
        name = "LONG" if direction == LONG else "SHORT"
        head = self.heads.get(name, {}).get(horizon)
        pattern = (
            patterns.assess_record(record, horizon=horizon)
            if patterns is not None
            else {
                "available": False,
                "bloom_hit": False,
                "count": 0,
                "bad_rate": 0.0,
            }
        )
        vector = meta_vector_from_record(record, direction, horizon)
        head_result = (
            head.predict(vector)
            if head is not None
            else {
                "available": False,
                "qualified": False,
                "p_take": 0.0,
                "p_false": 1.0,
                "selected": False,
                "reason": "META_HEAD_UNAVAILABLE",
            }
        )
        memory_reject = bool(
            pattern.get("available", False)
            and pattern.get("bloom_hit", False)
            and int(pattern.get("count", 0)) >= 2
            and _number(pattern.get("bad_rate"), 0.0) >= 0.75
        )
        selected = bool(head_result.get("selected", False) and not memory_reject)
        reason = (
            "KNOWN_FAKE_BREAKOUT_PATTERN"
            if memory_reject
            else head_result.get("reason", "META_HEAD_UNAVAILABLE")
        )
        return {
            "status": "READY",
            "model_id": self.model_id,
            "direction": name,
            "horizon": horizon,
            "features": dict(zip(META_FEATURES, vector.tolist())),
            "pattern_memory": pattern,
            **head_result,
            "selected": selected,
            "reason": reason,
            "exit_profile": self.exit_profiles.get(name, {}).get(horizon),
        }


def train_precision_meta_filter(
    frame: pd.DataFrame,
    oof_records: pd.DataFrame,
    settings: Settings,
    *,
    model_id: str,
) -> tuple[PrecisionMetaFilter, dict[str, Any]]:
    data = frame.copy().sort_values("open_time").reset_index(drop=True)
    event_index = {
        str(row["event_id"]): int(index)
        for index, row in data.iterrows()
        if row.get("event_id") not in (None, "")
    }
    event_oof = oof_records.loc[
        oof_records["record_type"].astype(str) == "EVENT"
    ].copy()
    general_one_hour = oof_records.loc[
        (oof_records["record_type"].astype(str) == "GENERAL")
        & (pd.to_numeric(oof_records["horizon"], errors="coerce") == 1)
    ].copy()
    general_p1 = {
        _time_key(row["open_time"]): _number(row.get("p_up"), 0.5)
        for _, row in general_one_hour.iterrows()
    }
    trade_horizons = sorted(
        {
            int(value)
            for value in settings.section("model").get(
                "trade_horizons_hours", [3, 6, 12]
            )
        }
    )
    heads: dict[str, dict[int, MetaHead]] = {"LONG": {}, "SHORT": {}}
    reports: dict[str, dict[str, Any]] = {"LONG": {}, "SHORT": {}}
    exit_profiles: dict[str, dict[int, dict[str, float]]] = {
        "LONG": {},
        "SHORT": {},
    }

    for direction, name in ((LONG, "LONG"), (SHORT, "SHORT")):
        for horizon in trade_horizons:
            subset = event_oof.loc[
                (
                    pd.to_numeric(
                        event_oof["event_direction"], errors="coerce"
                    )
                    == direction
                )
                & (
                    pd.to_numeric(event_oof["horizon"], errors="coerce")
                    == horizon
                )
            ].copy()
            subset = subset.sort_values("open_time").reset_index(drop=True)
            head, report = _train_head(
                data=data,
                rows=subset,
                event_index=event_index,
                direction=direction,
                horizon=horizon,
                settings=settings,
                general_p1=general_p1,
            )
            heads[name][horizon] = head
            reports[name][str(horizon)] = report
            exit_profiles[name][horizon] = build_exit_profile(
                data=data,
                direction=direction,
                horizon=horizon,
                settings=settings,
            )

    qualified_heads = sum(
        int(head.qualified)
        for values in heads.values()
        for head in values.values()
    )
    report = {
        "schema_version": META_FILTER_SCHEMA_VERSION,
        "model_id": model_id,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "method": "OOF_PRECISION_META_LABELING_WITH_FALSE_BREAKOUT_HEAD",
        "qualified_heads": qualified_heads,
        "passed": qualified_heads > 0,
        "heads": reports,
        "exit_profiles": exit_profiles,
    }
    artifact = PrecisionMetaFilter(
        schema_version=META_FILTER_SCHEMA_VERSION,
        model_id=model_id,
        created_at=report["created_at"],
        heads=heads,
        exit_profiles=exit_profiles,
        report=report,
    )
    return artifact, report


def save_precision_meta_filter(
    artifact: PrecisionMetaFilter,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(path)


def load_precision_meta_filter(path: Path) -> PrecisionMetaFilter:
    value = joblib.load(path)
    if not isinstance(value, PrecisionMetaFilter):
        raise TypeError(f"Unexpected meta-filter artifact: {type(value)!r}")
    if value.schema_version != META_FILTER_SCHEMA_VERSION:
        raise RuntimeError("Unsupported meta-filter artifact schema")
    return value


def apply_precision_gate(
    record: dict[str, Any],
    plan: dict[str, Any],
    assessment: dict[str, Any],
    settings: Settings,
) -> tuple[dict[str, Any], list[str]]:
    output = dict(plan)
    output["trade_assistant"] = assessment
    action = str(record.get("action") or "").upper()
    if action not in {"LONG", "SHORT"}:
        return output, []

    cfg = settings.section("trade_assistant")
    require_qualified = bool(
        cfg.get("require_qualified_meta_for_position", True)
    )
    blockers: list[str] = []
    if assessment.get("status") != "READY":
        if require_qualified:
            blockers.append("POSITION_META_FILTER_UNAVAILABLE")
    elif not assessment.get("qualified", False):
        if require_qualified:
            blockers.append("POSITION_META_NOT_QUALIFIED")
    elif not assessment.get("selected", False):
        blockers.append(str(
            assessment.get("reason") or "META_PRECISION_GATE_REJECTED"
        ))

    if blockers:
        output["status"] = "BLOCKED"
        output["position_quality_status"] = "EXPERIMENTAL_BLOCKED"
        return output, blockers

    profile = assessment.get("exit_profile")
    if isinstance(profile, dict):
        output = apply_exit_profile(record, output, profile)
    output["position_quality_status"] = "META_QUALIFIED"
    return output, []


def apply_exit_profile(
    record: dict[str, Any],
    plan: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    output = dict(plan)
    direction = str(record.get("action") or "").upper()
    if direction not in {"LONG", "SHORT"}:
        return output
    entry = _number(
        output.get("entry_reference", record.get("price")),
        0.0,
    )
    atr = _number(output.get("entry_atr"), 0.0)
    if entry <= 0 or atr <= 0:
        return output

    stop_atr = max(_number(profile.get("stop_atr"), 0.6), 0.10)
    target_atr = max(_number(profile.get("target_atr"), 0.8), 0.10)
    stop_distance = stop_atr * atr
    target_distance = target_atr * atr
    stop_price = (
        entry - stop_distance
        if direction == "LONG"
        else entry + stop_distance
    )
    target_price = (
        entry + target_distance
        if direction == "LONG"
        else entry - target_distance
    )
    output.update(
        {
            "contract_type": "HORIZON_ALIGNED_META_TARGET_STOP",
            "stop_price": float(stop_price),
            "initial_stop_price": float(stop_price),
            "target_price": float(target_price),
            "stop_percent": float(stop_distance / entry),
            "target_percent": float(target_distance / entry),
            "risk_reward": float(
                target_distance / max(stop_distance, 1e-9)
            ),
            "maximum_holding_hours": int(
                round(_number(profile.get("horizon_hours"), 3.0))
            ),
            "breakeven_trigger_r": _number(
                profile.get("breakeven_trigger_r"),
                1.0,
            ),
            "trailing_trigger_r": _number(
                profile.get("trailing_trigger_r"),
                1.5,
            ),
            "exit_profile_source": "LOCKED_HISTORICAL_MFE_MAE",
            "exit_profile": dict(profile),
        }
    )
    return output


def build_exit_profile(
    *,
    data: pd.DataFrame,
    direction: int,
    horizon: int,
    settings: Settings,
) -> dict[str, float]:
    cfg_name = "long_breakout" if direction == LONG else "short_breakdown"
    cfg = settings.section(cfg_name)
    rows = data.loc[data["event_direction"] == direction].copy()
    success_col = f"breakout_success_h{horizon}"
    mfe_col = f"event_mfe_atr_h{horizon}"
    mae_col = f"event_mae_atr_h{horizon}"
    if success_col in rows:
        rows = rows.loc[
            pd.to_numeric(rows[success_col], errors="coerce") >= 0.5
        ]
    else:
        rows = rows.iloc[0:0]
    mfe = pd.to_numeric(rows.get(mfe_col), errors="coerce").dropna()
    mae = pd.to_numeric(rows.get(mae_col), errors="coerce").dropna()

    target_map = cfg.get("target_atr_by_horizon", {})
    target_default = float(
        target_map.get(horizon, target_map.get(str(horizon), 1.0))
    )
    q25_mfe = float(mfe.quantile(0.25)) if len(mfe) else target_default
    q50_mfe = (
        float(mfe.quantile(0.50))
        if len(mfe)
        else target_default * 1.25
    )
    q75_mae = (
        float(mae.quantile(0.75))
        if len(mae)
        else float(cfg.get("invalidation_atr", 0.6))
    )
    target_atr = float(
        np.clip(min(target_default, q25_mfe), 0.35, 3.0)
    )
    stop_atr = float(
        np.clip(
            max(q75_mae, float(cfg.get("invalidation_atr", 0.6))),
            0.35,
            2.5,
        )
    )
    breakeven = float(
        np.clip(0.55 * q50_mfe / max(stop_atr, 1e-9), 0.75, 1.50)
    )
    trailing = float(
        np.clip(0.85 * q50_mfe / max(stop_atr, 1e-9), 1.00, 2.50)
    )
    return {
        "horizon_hours": float(horizon),
        "target_atr": target_atr,
        "stop_atr": stop_atr,
        "risk_reward": float(
            np.clip(target_atr / max(stop_atr, 1e-9), 0.5, 4.0)
        ),
        "breakeven_trigger_r": breakeven,
        "trailing_trigger_r": max(trailing, breakeven + 0.15),
        "successful_mfe_q25_atr": q25_mfe,
        "successful_mfe_q50_atr": q50_mfe,
        "successful_mae_q75_atr": q75_mae,
        "successful_samples": float(len(rows)),
    }


def meta_vector_from_record(
    record: dict[str, Any],
    direction: int,
    horizon: int,
) -> np.ndarray:
    base = record.get("base_model")
    base = base if isinstance(base, dict) else {}
    continuation = _mapping_value(
        base.get("continuation"),
        horizon,
        0.5,
    )
    tradeability = _mapping_value(
        base.get("tradeability"),
        horizon,
        _number(record.get("tradeability_probability"), 0.5),
    )
    event_return = _mapping_value(
        base.get("event_returns"),
        horizon,
        _number(record.get("expected_return"), 0.0),
    )
    probabilities = base.get("probabilities")
    p1 = _mapping_value(probabilities, 1, 0.5)
    context = context_vector_from_record(record)
    return _assemble_meta_vector(
        continuation=continuation,
        tradeability=tradeability,
        predicted_return=event_return,
        p1=p1,
        event_score=_number(
            record.get("trigger_score", record.get("event_score")),
            0.0,
        ),
        event_scale=_number(record.get("event_scale_hours"), 0.0),
        context=context,
    )


def _meta_vector_from_training(
    data: pd.DataFrame,
    index: int,
    row: pd.Series,
    *,
    p1: float,
) -> np.ndarray:
    source = data.iloc[index]
    context = context_vector_from_frame(data, index)
    return _assemble_meta_vector(
        continuation=_number(row.get("p_continuation"), 0.5),
        tradeability=_number(row.get("p_tradeable"), 0.5),
        predicted_return=_number(
            row.get("predicted_event_gross_return"), 0.0
        ),
        p1=p1,
        event_score=_number(source.get("event_score"), 0.0),
        event_scale=_number(source.get("event_scale_hours"), 0.0),
        context=context,
    )


def _assemble_meta_vector(
    *,
    continuation: float,
    tradeability: float,
    predicted_return: float,
    p1: float,
    event_score: float,
    event_scale: float,
    context: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        [
            float(np.clip(continuation, 0.0, 1.0)),
            float(np.clip(tradeability, 0.0, 1.0)),
            float(np.clip(predicted_return / 0.02, -3.0, 3.0)),
            float(np.clip(abs(p1 - 0.5) * 2.0, 0.0, 1.0)),
            float(np.clip(event_score, 0.0, 1.0)),
            float(np.clip(event_scale / 720.0, 0.0, 1.0)),
            *context.tolist(),
        ],
        dtype=float,
    )
    return np.nan_to_num(values, nan=0.0, posinf=3.0, neginf=-3.0)


def _train_head(
    *,
    data: pd.DataFrame,
    rows: pd.DataFrame,
    event_index: dict[str, int],
    direction: int,
    horizon: int,
    settings: Settings,
    general_p1: dict[str, float],
) -> tuple[MetaHead, dict[str, Any]]:
    cfg = settings.section("trade_assistant")
    minimum = int(cfg.get("minimum_meta_samples", 600))
    vectors: list[np.ndarray] = []
    take_labels: list[int] = []
    false_labels: list[int] = []
    net_returns: list[float] = []

    false_col = f"false_breakout_h{horizon}"
    for _, row in rows.iterrows():
        index = event_index.get(str(row.get("event_id") or ""))
        if (
            index is None
            or false_col not in data
            or pd.isna(data.iloc[index].get(false_col))
        ):
            continue
        false_label = int(
            _number(data.iloc[index].get(false_col), 0.0) >= 0.5
        )
        actual_tradeable = int(
            _number(row.get("actual_tradeable"), 0.0) >= 0.5
        )
        actual_net = _number(row.get("actual_event_net_return"), 0.0)
        source_key = _time_key(data.iloc[index]["open_time"])
        vectors.append(
            _meta_vector_from_training(
                data,
                index,
                row,
                p1=general_p1.get(source_key, 0.5),
            )
        )
        false_labels.append(false_label)
        take_labels.append(
            int(
                actual_tradeable == 1
                and false_label == 0
                and actual_net > 0
            )
        )
        net_returns.append(actual_net)

    if len(vectors) < minimum:
        report = {
            "qualified": False,
            "samples": len(vectors),
            "blockers": [
                f"insufficient usable OOF meta samples; {minimum} required"
            ],
        }
        return _empty_head(direction, horizon, report), report

    X = np.vstack(vectors)
    y_take = np.asarray(take_labels, dtype=int)
    y_false = np.asarray(false_labels, dtype=int)
    actual_net = np.asarray(net_returns, dtype=float)
    if len(np.unique(y_take)) < 2 or len(np.unique(y_false)) < 2:
        report = {
            "qualified": False,
            "samples": len(X),
            "blockers": ["meta labels require both classes"],
        }
        return _empty_head(direction, horizon, report), report

    train_end = max(200, int(len(X) * 0.70))
    calibration_end = min(
        max(train_end + 100, int(len(X) * 0.85)),
        len(X) - 50,
    )
    if calibration_end <= train_end or len(X) - calibration_end < 50:
        report = {
            "qualified": False,
            "samples": len(X),
            "blockers": [
                "insufficient chronological calibration/holdout rows"
            ],
        }
        return _empty_head(direction, horizon, report), report

    take_model = _classifier(cfg)
    false_model = _classifier(cfg)
    take_model.fit(X[:train_end], y_take[:train_end])
    false_model.fit(X[:train_end], y_false[:train_end])

    policy = _choose_policy(
        p_take=take_model.predict_proba(
            X[train_end:calibration_end]
        )[:, 1],
        p_false=false_model.predict_proba(
            X[train_end:calibration_end]
        )[:, 1],
        y_take=y_take[train_end:calibration_end],
        y_false=y_false[train_end:calibration_end],
        actual_net=actual_net[train_end:calibration_end],
        settings=settings,
    )
    hold_take = take_model.predict_proba(X[calibration_end:])[:, 1]
    hold_false = false_model.predict_proba(X[calibration_end:])[:, 1]
    selected = (
        (hold_take >= policy["minimum_take_probability"])
        & (hold_false <= policy["maximum_false_probability"])
    )
    count = int(selected.sum())
    precision = (
        float(np.mean(y_take[calibration_end:][selected]))
        if count
        else 0.0
    )
    fake_rate = (
        float(np.mean(y_false[calibration_end:][selected]))
        if count
        else 1.0
    )
    mean_net = (
        float(np.mean(actual_net[calibration_end:][selected]))
        if count
        else 0.0
    )

    blockers: list[str] = []
    if count < int(cfg.get("minimum_meta_holdout_selected", 12)):
        blockers.append("insufficient locked holdout selections")
    if precision < float(cfg.get("minimum_meta_holdout_precision", 0.55)):
        blockers.append("holdout precision below minimum")
    if fake_rate > float(
        cfg.get("maximum_meta_holdout_fake_rate", 0.35)
    ):
        blockers.append("holdout fake-breakout acceptance too high")
    if mean_net <= float(
        cfg.get("minimum_meta_holdout_mean_net_return", 0.0)
    ):
        blockers.append("holdout mean net return is not positive")

    report = {
        "qualified": not blockers,
        "samples": len(X),
        "train_samples": train_end,
        "calibration_samples": calibration_end - train_end,
        "holdout_samples": len(X) - calibration_end,
        "selected": count,
        "precision": precision,
        "fake_rate": fake_rate,
        "mean_net_return": mean_net,
        "policy": policy,
        "blockers": blockers,
    }
    return (
        MetaHead(
            direction=direction,
            horizon=horizon,
            take_model=take_model,
            false_model=false_model,
            minimum_take_probability=policy[
                "minimum_take_probability"
            ],
            maximum_false_probability=policy[
                "maximum_false_probability"
            ],
            qualified=not blockers,
            report=report,
        ),
        report,
    )


def _choose_policy(
    *,
    p_take: np.ndarray,
    p_false: np.ndarray,
    y_take: np.ndarray,
    y_false: np.ndarray,
    actual_net: np.ndarray,
    settings: Settings,
) -> dict[str, float]:
    cfg = settings.section("trade_assistant")
    minimum = int(cfg.get("minimum_meta_calibration_selected", 20))
    best: tuple[tuple[float, float, int], dict[str, float]] | None = None
    for take_threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        for false_threshold in (0.50, 0.45, 0.40, 0.35, 0.30, 0.25):
            selected = (
                (p_take >= take_threshold)
                & (p_false <= false_threshold)
            )
            count = int(selected.sum())
            if count < minimum:
                continue
            precision = float(np.mean(y_take[selected]))
            fake_rate = float(np.mean(y_false[selected]))
            mean_net = float(np.mean(actual_net[selected]))
            if mean_net <= 0:
                continue
            score = (
                4.0 * precision
                - 2.5 * fake_rate
                + min(mean_net * 10_000.0, 50.0) / 50.0
            )
            key = (score, precision, count)
            policy = {
                "minimum_take_probability": float(take_threshold),
                "maximum_false_probability": float(false_threshold),
            }
            if best is None or key > best[0]:
                best = (key, policy)
    return (
        best[1]
        if best is not None
        else {
            "minimum_take_probability": float(
                cfg.get(
                    "fallback_minimum_take_probability",
                    0.72,
                )
            ),
            "maximum_false_probability": float(
                cfg.get(
                    "fallback_maximum_false_probability",
                    0.30,
                )
            ),
        }
    )


def _classifier(cfg: dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(cfg.get("learning_rate", 0.035)),
        max_iter=int(cfg.get("max_iter", 180)),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 9)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 35)),
        l2_regularization=float(cfg.get("l2_regularization", 5.0)),
        class_weight="balanced",
        early_stopping=False,
        random_state=int(cfg.get("random_state", 20260807)),
    )


def _empty_head(
    direction: int,
    horizon: int,
    report: dict[str, Any],
) -> MetaHead:
    return MetaHead(
        direction=direction,
        horizon=horizon,
        take_model=None,
        false_model=None,
        minimum_take_probability=1.0,
        maximum_false_probability=0.0,
        qualified=False,
        report=report,
    )


def _mapping_value(
    value: Any,
    key: int,
    default: float,
) -> float:
    if isinstance(value, dict):
        return _number(
            value.get(key, value.get(str(key), default)),
            default,
        )
    return float(default)


def _time_key(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)
