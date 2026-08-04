from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import Settings
from .negative_memory_core import SUPPORT, RESISTANCE, BloomFilter
from .negative_memory_dataset import mine_boundary_encounters
from .negative_memory_model import BoundaryHead, SandwichedBoundaryMemory

def train_sandwiched_boundary_memory(
    frame: pd.DataFrame,
    settings: Settings,
    horizons: Iterable[int],
    feature_columns: Iterable[str],
    model_id: str,
) -> tuple[SandwichedBoundaryMemory, dict[str, Any], pd.DataFrame]:
    cfg = settings.section("negative_memory")
    trade_horizons = sorted({int(value) for value in horizons if int(value) > 1})
    encounters = mine_boundary_encounters(frame, settings, trade_horizons)
    features = _feature_columns(feature_columns)
    heads: dict[str, dict[int, BoundaryHead]] = {
        SUPPORT: {}, RESISTANCE: {}
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "SANDWICHED_NEGATIVE_MEMORY_WITH_HARD_NEGATIVE_MINING",
        "encounters": int(len(encounters)),
        "features": features,
        "sides": {SUPPORT: {}, RESISTANCE: {}},
    }
    oof: list[pd.DataFrame] = []
    for side in (SUPPORT, RESISTANCE):
        for horizon in trade_horizons:
            rows = encounters.loc[
                (encounters["boundary_side"] == side)
                & (encounters["horizon"] == horizon)
            ].sort_values("open_time").reset_index(drop=True)
            head, item, records = _train_head(
                rows, side, horizon, features, cfg
            )
            heads[side][horizon] = head
            report["sides"][side][str(horizon)] = item
            if not records.empty:
                oof.append(records)
    report["qualified_heads"] = sum(
        int(head.report.get("qualified", False))
        for side_heads in heads.values()
        for head in side_heads.values()
    )
    report["passed"] = report["qualified_heads"] > 0
    memory = SandwichedBoundaryMemory(
        model_id=model_id,
        heads=heads,
        report=report,
        minimum_distance_atr=float(
            cfg.get("minimum_boundary_distance_atr", -0.25)
        ),
        maximum_distance_atr=float(
            cfg.get("maximum_boundary_distance_atr", 0.85)
        ),
    )
    return memory, report, (
        pd.concat(oof, ignore_index=True) if oof else pd.DataFrame()
    )

def _train_head(
    rows: pd.DataFrame,
    side: str,
    horizon: int,
    features: list[str],
    cfg: dict[str, Any],
) -> tuple[BoundaryHead, dict[str, Any], pd.DataFrame]:
    minimum = int(cfg.get("minimum_samples_per_head", 500))
    empty = BloomFilter.create(1)
    if (
        len(rows) < minimum
        or rows.get("break_label", pd.Series(dtype=int)).nunique() < 2
        or rows.get("bad_label", pd.Series(dtype=int)).nunique() < 2
    ):
        item = {
            "qualified": False,
            "samples": int(len(rows)),
            "blockers": [f"insufficient two-class encounters; {minimum} required"],
        }
        return (
            BoundaryHead(
                side, horizon, features, None, None, empty,
                BloomFilter.create(1), {}, item,
            ),
            item,
            pd.DataFrame(),
        )
    first = max(200, int(len(rows) * 0.70))
    second = min(max(first + 100, int(len(rows) * 0.85)), len(rows) - 50)
    train, calibration, holdout = (
        rows.iloc[:first].copy(),
        rows.iloc[first:second].copy(),
        rows.iloc[second:].copy(),
    )
    x_train = train.reindex(columns=features)
    break_model, bad_model = _classifier(cfg), _classifier(cfg)
    weights = np.where(train["bad_label"].to_numpy() == 1, 1.5, 1.0)
    break_model.fit(x_train, train["break_label"], sample_weight=weights)
    bad_model.fit(x_train, train["bad_label"], sample_weight=weights)
    hard = (
        (train["bad_label"].to_numpy() == 1)
        & (bad_model.predict_proba(x_train)[:, 1] < 0.50)
    )
    bad_model = _classifier(cfg)
    bad_model.fit(
        x_train,
        train["bad_label"],
        sample_weight=weights
        * np.where(hard, float(cfg.get("hard_negative_weight", 3.0)), 1.0),
    )
    calibration = _probabilities(calibration, features, break_model, bad_model)
    policy = _policy(calibration, cfg)
    grouped = train.groupby("boundary_fingerprint")["bad_label"].agg(
        ["count", "mean"]
    )
    front_keys = grouped.loc[
        (grouped["count"] >= int(cfg.get("front_memory_minimum_count", 2)))
        & (grouped["mean"] >= float(cfg.get("front_memory_bad_rate", 0.80)))
    ].index.astype(str).tolist()
    fpr = float(cfg.get("bloom_false_positive_rate", 0.005))
    front = BloomFilter.create(max(1, len(front_keys)), fpr)
    for key in front_keys:
        front.add(key)
    accepted = (
        (calibration["p_break"] >= policy["minimum_break_probability"])
        & (calibration["p_bad"] <= policy["maximum_bad_probability"])
    )
    backup_keys = calibration.loc[
        (calibration["bad_label"] == 1)
        & accepted
        & ~calibration["boundary_fingerprint"].map(front.contains),
        "boundary_fingerprint",
    ].drop_duplicates().astype(str).tolist()
    backup = BloomFilter.create(max(1, len(backup_keys)), fpr)
    for key in backup_keys:
        backup.add(key)
    holdout = _probabilities(holdout, features, break_model, bad_model)
    holdout["front_memory_hit"] = holdout["boundary_fingerprint"].map(
        front.contains
    )
    holdout["backup_memory_hit"] = holdout["boundary_fingerprint"].map(
        backup.contains
    )
    holdout["selected"] = (
        ~holdout["front_memory_hit"]
        & ~holdout["backup_memory_hit"]
        & (holdout["p_break"] >= policy["minimum_break_probability"])
        & (holdout["p_bad"] <= policy["maximum_bad_probability"])
    )
    selected = holdout.loc[holdout["selected"]]
    count = int(len(selected))
    mean_net = float(selected["stress_net_return"].mean()) if count else 0.0
    profitable = float(selected["profitable_label"].mean()) if count else 0.0
    bad_rate = float(selected["bad_label"].mean()) if count else 1.0
    blockers: list[str] = []
    if count < int(cfg.get("minimum_holdout_selected", 12)):
        blockers.append("insufficient locked holdout selections")
    if mean_net <= float(cfg.get("minimum_holdout_mean_net_return", 0.0)):
        blockers.append("holdout mean stress-net return is not positive")
    if profitable < float(cfg.get("minimum_holdout_profitable_rate", 0.52)):
        blockers.append("holdout profitable-break rate is too low")
    if bad_rate > float(cfg.get("maximum_holdout_bad_acceptance", 0.48)):
        blockers.append("too many unprofitable patterns are accepted")
    item = {
        "qualified": not blockers,
        "samples": int(len(rows)),
        "hard_negatives": int(hard.sum()),
        "front_memory_keys": len(front_keys),
        "backup_memory_keys": len(backup_keys),
        "policy": policy,
        "holdout_selected": count,
        "holdout_mean_stress_net_return": mean_net,
        "holdout_profitable_rate": profitable,
        "holdout_bad_acceptance": bad_rate,
        "blockers": blockers,
    }
    records = holdout[
        [
            "open_time", "boundary_side", "horizon", "break_label",
            "profitable_label", "bad_label", "stress_net_return",
            "p_break", "p_bad", "front_memory_hit",
            "backup_memory_hit", "selected",
        ]
    ].copy()
    records.insert(0, "record_type", "BOUNDARY_MEMORY")
    return (
        BoundaryHead(
            side, horizon, features, break_model, bad_model,
            front, backup, policy, item,
        ),
        item,
        records,
    )

def _feature_columns(columns: Iterable[str]) -> list[str]:
    preferred = {
        "atr_pct", "atr_z_72", "atr_percentile_168", "adx",
        "plus_di", "minus_di", "di_spread", "rsi_centered",
        "return_1", "return_3", "return_6", "return_12", "return_24",
        "realized_vol_6", "realized_vol_12", "realized_vol_24",
        "body_atr", "close_location", "upper_wick_pct", "lower_wick_pct",
        "volume_change_1", "volume_z_24", "volume_z_72",
        "volume_trend_24_72", "price_vs_kama", "price_vs_ema_24",
        "price_vs_ema_72", "price_vs_ema_168", "regime_code",
        "triangle_code", "triangle_quality", "triangle_contraction",
        "news_weighted_sent_6h", "news_relevance_6h",
        "news_negative_share_6h",
    }
    selected = [
        column for column in columns
        if column in preferred
        or (
            column.startswith("structure_")
            and column.endswith(("_touches", "_r2", "_slope_atr", "_width_atr"))
        )
    ]
    selected.extend(
        column for column in (
            "boundary_side_code", "boundary_distance_atr",
            "boundary_strength", "boundary_age_bars",
            "boundary_approach_return_6",
        ) if column not in selected
    )
    return selected

def _classifier(cfg: dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=float(cfg.get("learning_rate", 0.035)),
        max_iter=int(cfg.get("max_iter", 220)),
        max_leaf_nodes=int(cfg.get("max_leaf_nodes", 11)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 35)),
        l2_regularization=float(cfg.get("l2_regularization", 4.0)),
        class_weight="balanced",
        early_stopping=False,
        random_state=int(cfg.get("random_state", 20260804)),
    )

def _probabilities(
    rows: pd.DataFrame,
    features: list[str],
    break_model: Any,
    bad_model: Any,
) -> pd.DataFrame:
    output = rows.copy()
    x = output.reindex(columns=features)
    output["p_break"] = break_model.predict_proba(x)[:, 1]
    output["p_bad"] = bad_model.predict_proba(x)[:, 1]
    return output

def _policy(rows: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float]:
    best: tuple[float, dict[str, float]] | None = None
    minimum = int(cfg.get("minimum_calibration_selected", 30))
    for break_threshold in (0.52, 0.56, 0.60, 0.64, 0.68):
        for bad_threshold in (0.30, 0.38, 0.46, 0.54, 0.62):
            selected = rows.loc[
                (rows["p_break"] >= break_threshold)
                & (rows["p_bad"] <= bad_threshold)
            ]
            if len(selected) < minimum:
                continue
            score = (
                float(selected["stress_net_return"].mean()) * 10_000
                + 2 * float(selected["profitable_label"].mean())
                - 2.5 * float(selected["bad_label"].mean())
                + min(2.0, math.log1p(len(selected)) / 3)
            )
            policy = {
                "minimum_break_probability": break_threshold,
                "maximum_bad_probability": bad_threshold,
            }
            if best is None or score > best[0]:
                best = (score, policy)
    return best[1] if best else {
        "minimum_break_probability": float(
            cfg.get("fallback_minimum_break_probability", 0.62)
        ),
        "maximum_bad_probability": float(
            cfg.get("fallback_maximum_bad_probability", 0.42)
        ),
    }
