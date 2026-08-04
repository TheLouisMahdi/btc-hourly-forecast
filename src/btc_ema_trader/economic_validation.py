from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .costs import execution_cost_breakdown

DIRECTIONS = ("LONG", "SHORT")
HORIZONS = (3, 6, 12)
DEV_FRACTION = 0.70
MIN_DEV_TRADES = 30
MIN_HOLDOUT_TRADES = 12
BOOTSTRAPS = 800
SEED = 20260804


@dataclass(frozen=True)
class CalibrationMap:
    coefficient: float = 1.0
    intercept: float = 0.0

    def transform(self, probability: np.ndarray | float) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), 1e-5, 1 - 1e-5)
        z = np.log(p / (1 - p))
        return np.clip(
            1 / (1 + np.exp(-(self.intercept + self.coefficient * z))),
            1e-4,
            1 - 1e-4,
        )


def apply_calibration(value: float, mapping: dict[str, Any] | None) -> float:
    if not isinstance(mapping, dict):
        return float(np.clip(value, 0, 1))
    try:
        calibration = CalibrationMap(
            float(mapping["coefficient"]), float(mapping["intercept"])
        )
    except (KeyError, TypeError, ValueError):
        return float(np.clip(value, 0, 1))
    return float(calibration.transform(value))


def evaluate_and_patch_candidate(
    report_path: Path,
    oof_path: Path,
    model_path: Path,
    output_path: Path,
    incumbent_economic_report: Path | None = None,
) -> dict[str, Any]:
    report = _read_json(report_path)
    result = evaluate_oof_economics(report, pd.read_csv(oof_path))
    incumbent = (
        _read_json(incumbent_economic_report)
        if incumbent_economic_report and incumbent_economic_report.exists()
        else None
    )
    result["promotion"] = compare_with_incumbent(result, incumbent)
    _write_json(output_path, result)
    bundle = joblib.load(model_path)
    qualification = dict(getattr(bundle, "qualification", {}) or {})
    qualification.update(result["qualification"])
    bundle.qualification = qualification
    joblib.dump(bundle, model_path)
    return result


def evaluate_oof_economics(
    report: dict[str, Any], oof: pd.DataFrame
) -> dict[str, Any]:
    strategy = _strategy(report)
    costs = execution_cost_breakdown(strategy)
    stress_bps = float(costs["stress_cost_bps"]) + max(
        4.0, float(strategy.get("economic_execution_uncertainty_bps", 4.0))
    )
    model_cfg = _snapshot(report, "model")
    horizons = tuple(
        int(v) for v in model_cfg.get("trade_horizons_hours", HORIZONS)
    )
    if "record_type" not in oof:
        return _empty(stress_bps, "OOF records do not identify event rows")
    events = oof.loc[oof["record_type"] == "EVENT"].copy()
    if events.empty:
        return _empty(stress_bps, "No chronological event OOF records")
    events["open_time"] = pd.to_datetime(events["open_time"], utc=True)

    qualified = {name: [] for name in DIRECTIONS}
    policies = {name: {} for name in DIRECTIONS}
    details = {name: {} for name in DIRECTIONS}
    blockers: list[str] = []
    scores: list[float] = []
    for direction in DIRECTIONS:
        for horizon in horizons:
            subset = events.loc[
                (events["direction_name"] == direction)
                & (pd.to_numeric(events["horizon"], errors="coerce") == horizon)
            ].sort_values("open_time")
            item = _evaluate_pair(subset.reset_index(drop=True), horizon, stress_bps)
            details[direction][str(horizon)] = item
            if item["passed"]:
                qualified[direction].append(horizon)
                policies[direction][str(horizon)] = item["policy"]
                scores.append(float(item["holdout_objective_bps"]))
            else:
                blockers.extend(
                    f"{direction} h{horizon}: {reason}"
                    for reason in item["blockers"]
                )
    passed = any(qualified.values())
    qualification = {
        "passed": passed,
        "qualified_horizons": sorted(
            {h for values in qualified.values() for h in values}
        ),
        "qualified_directions": qualified,
        "per_direction": details,
        "blockers": blockers,
        "economic_policy": policies,
        "economic_stress_cost_bps": stress_bps,
        "economic_validation_version": 1,
    }
    return {
        "schema_version": 1,
        "method": "LOCKED_CHRONOLOGICAL_DEVELOPMENT_HOLDOUT",
        "development_fraction": DEV_FRACTION,
        "economic_stress_cost_bps": stress_bps,
        "qualification": qualification,
        "aggregate_holdout_objective_bps": (
            float(np.mean(scores)) if scores else None
        ),
        "passed_pairs": sum(len(v) for v in qualified.values()),
    }


def compare_with_incumbent(
    candidate: dict[str, Any], incumbent: dict[str, Any] | None
) -> dict[str, Any]:
    candidate_score = _number(candidate.get("aggregate_holdout_objective_bps"))
    incumbent_score = _number(
        incumbent.get("aggregate_holdout_objective_bps") if incumbent else None
    )
    candidate_passed = bool(
        candidate.get("qualification", {}).get("passed", False)
    )
    incumbent_passed = bool(
        incumbent and incumbent.get("qualification", {}).get("passed", False)
    )
    if not candidate_passed or candidate_score is None:
        return _promotion(
            "KEEP_INCUMBENT",
            "Candidate did not pass locked economic holdout gates",
            candidate_score,
            incumbent_score,
        )
    if not incumbent_passed or incumbent_score is None:
        return _promotion(
            "PROMOTE",
            "Candidate passed and no qualified incumbent exists",
            candidate_score,
            incumbent_score,
        )
    margin = max(0.5, abs(incumbent_score) * 0.05)
    promote = candidate_score >= incumbent_score + margin
    result = _promotion(
        "PROMOTE" if promote else "KEEP_INCUMBENT",
        (
            "Candidate exceeded the incumbent conservative score"
            if promote
            else "Candidate did not exceed the incumbent promotion margin"
        ),
        candidate_score,
        incumbent_score,
    )
    result["required_improvement_bps"] = margin
    return result


def _evaluate_pair(
    rows: pd.DataFrame, horizon: int, stress_bps: float
) -> dict[str, Any]:
    if len(rows) < 200:
        return _failed(len(rows), f"only {len(rows)} OOF events; 200 required")
    split = min(max(int(len(rows) * DEV_FRACTION), 120), len(rows) - 60)
    dev, holdout = rows.iloc[:split].copy(), rows.iloc[split:].copy()
    success_cal = _fit_calibration(dev["p_continuation"], dev["actual_continuation"])
    trade_cal = _fit_calibration(dev["p_tradeable"], dev["actual_tradeable"])
    dev = _prepare(dev, success_cal, trade_cal, stress_bps)
    holdout = _prepare(holdout, success_cal, trade_cal, stress_bps)
    policy, dev_stats = _choose_policy(dev, horizon, success_cal, trade_cal)
    if policy is None:
        return _failed(
            len(rows),
            "no cost-aware policy produced enough positive development trades",
        )
    selected = _select(holdout, policy, horizon)
    hold_stats = _statistics(
        selected.get("actual_event_net_return", pd.Series(dtype=float)),
        selected.get("actual_continuation", pd.Series(dtype=float)),
        SEED + horizon,
    )
    blockers = _holdout_blockers(hold_stats)
    return {
        "passed": not blockers,
        "event_samples": len(rows),
        "development_samples": len(dev),
        "holdout_samples": len(holdout),
        "policy": policy,
        "development": dev_stats,
        "holdout": hold_stats,
        "development_objective_bps": _objective(dev_stats),
        "holdout_objective_bps": _objective(hold_stats),
        "blockers": blockers,
    }


def _choose_policy(
    dev: pd.DataFrame,
    horizon: int,
    success_cal: CalibrationMap,
    trade_cal: CalibrationMap,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for success in (0.54, 0.58, 0.62, 0.66):
        for tradeable in (0.54, 0.58, 0.62, 0.66):
            for edge in (16.0, 20.0, 25.0, 32.0):
                for score in (0.34, 0.46, 0.58):
                    policy = {
                        "success_probability": success,
                        "tradeability_probability": tradeable,
                        "minimum_predicted_stress_edge_bps": edge,
                        "minimum_event_score": score,
                        "success_calibration": asdict(success_cal),
                        "tradeability_calibration": asdict(trade_cal),
                    }
                    selected = _select(dev, policy, horizon)
                    if len(selected) < MIN_DEV_TRADES:
                        continue
                    stats = _statistics(
                        selected["actual_event_net_return"],
                        selected["actual_continuation"],
                        SEED + len(selected),
                        bootstrap=False,
                    )
                    if stats["mean_net_return"] <= 0 or stats["profit_factor"] <= 1:
                        continue
                    objective = _objective(stats)
                    if best is None or objective > best[0]:
                        best = (float(objective), policy, stats)
    if best is None:
        return None, None
    selected = _select(dev, best[1], horizon)
    return best[1], _statistics(
        selected["actual_event_net_return"],
        selected["actual_continuation"],
        SEED + len(selected),
    )


def _prepare(
    rows: pd.DataFrame,
    success_cal: CalibrationMap,
    trade_cal: CalibrationMap,
    stress_bps: float,
) -> pd.DataFrame:
    output = rows.copy()
    output["calibrated_success"] = success_cal.transform(output["p_continuation"])
    output["calibrated_tradeability"] = trade_cal.transform(output["p_tradeable"])
    output["predicted_stress_edge_bps"] = (
        pd.to_numeric(output["predicted_event_gross_return"], errors="coerce")
        * 10_000
        - stress_bps
    )
    output["event_score"] = pd.to_numeric(
        output["event_score"], errors="coerce"
    ).fillna(0)
    return output


def _select(
    rows: pd.DataFrame, policy: dict[str, Any], horizon: int
) -> pd.DataFrame:
    selected = rows.loc[
        (rows["calibrated_success"] >= policy["success_probability"])
        & (
            rows["calibrated_tradeability"]
            >= policy["tradeability_probability"]
        )
        & (
            rows["predicted_stress_edge_bps"]
            >= policy["minimum_predicted_stress_edge_bps"]
        )
        & (rows["event_score"] >= policy["minimum_event_score"])
    ].copy()
    if selected.empty:
        return selected
    selected["rank"] = (
        selected["calibrated_success"]
        * selected["calibrated_tradeability"]
        * np.maximum(selected["predicted_stress_edge_bps"], 0)
        * (0.5 + selected["event_score"])
    )
    accepted: list[int] = []
    last_exit: pd.Timestamp | None = None
    for timestamp, group in selected.groupby("open_time", sort=True):
        row = group.sort_values("rank", ascending=False).iloc[0]
        current = pd.Timestamp(timestamp)
        if last_exit is not None and current < last_exit:
            continue
        accepted.append(int(row.name))
        last_exit = current + pd.Timedelta(hours=max(1, horizon))
    return selected.loc[accepted].sort_values("open_time").reset_index(drop=True)


def _statistics(
    returns: Any, success: Any, seed: int, bootstrap: bool = True
) -> dict[str, Any]:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "selected": 0,
            "hit_rate": 0.0,
            "mean_net_return": 0.0,
            "median_net_return": 0.0,
            "cumulative_net_return": 0.0,
            "profit_factor": 0.0,
            "maximum_drawdown": 0.0,
            "lower_mean_return_95": 0.0,
            "upper_mean_return_95": 0.0,
            "positive_half_fraction": 0.0,
        }
    successes = np.asarray(success, dtype=float)[: len(values)]
    equity = np.cumprod(1 + values)
    peaks = np.maximum.accumulate(equity)
    drawdown = 1 - equity / np.maximum(peaks, 1e-12)
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    profit_factor = gains / losses if losses > 0 else (99.0 if gains > 0 else 0.0)
    if bootstrap:
        lower, upper = _bootstrap(values, seed)
    else:
        se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        lower, upper = values.mean() - 1.96 * se, values.mean() + 1.96 * se
    halves = [part for part in np.array_split(values, 2) if len(part)]
    return {
        "selected": len(values),
        "hit_rate": float(np.mean(successes > 0.5)) if len(successes) else 0.0,
        "mean_net_return": float(values.mean()),
        "median_net_return": float(np.median(values)),
        "cumulative_net_return": float(equity[-1] - 1),
        "profit_factor": float(profit_factor),
        "maximum_drawdown": float(drawdown.max(initial=0)),
        "lower_mean_return_95": float(lower),
        "upper_mean_return_95": float(upper),
        "positive_half_fraction": float(np.mean([part.mean() > 0 for part in halves])),
    }


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values.mean())
        return value, value
    rng = np.random.default_rng(seed)
    block = max(2, min(12, round(math.sqrt(len(values)))))
    means = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        sample: list[float] = []
        while len(sample) < len(values):
            start = int(rng.integers(0, len(values)))
            positions = (start + np.arange(block)) % len(values)
            sample.extend(values[positions].tolist())
        means[index] = np.mean(sample[: len(values)])
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _holdout_blockers(stats: dict[str, Any]) -> list[str]:
    blockers = []
    if stats["selected"] < MIN_HOLDOUT_TRADES:
        blockers.append(
            f"only {stats['selected']} locked holdout trades; "
            f"{MIN_HOLDOUT_TRADES} required"
        )
    if stats["mean_net_return"] <= 0:
        blockers.append("locked holdout mean net expectancy is not positive")
    if stats["profit_factor"] < 1.10:
        blockers.append("locked holdout profit factor is below 1.10")
    if stats["lower_mean_return_95"] < -0.0002:
        blockers.append("95% lower mean-return bound is below -2 bps")
    if stats["maximum_drawdown"] > 0.10:
        blockers.append("locked holdout maximum drawdown exceeds 10%")
    if stats["positive_half_fraction"] < 1.0:
        blockers.append("edge is not positive in both holdout halves")
    return blockers


def _objective(stats: dict[str, Any] | None) -> float | None:
    if not stats or not stats["selected"]:
        return None
    return float(
        stats["lower_mean_return_95"] * 10_000
        + 0.35 * stats["mean_net_return"] * 10_000
        + min(3.0, math.log1p(stats["selected"]) / 2)
        + min(3.0, max(0.0, stats["profit_factor"] - 1) * 2)
        - stats["maximum_drawdown"] * 100
    )


def _fit_calibration(probability: Any, target: Any) -> CalibrationMap:
    p = np.clip(np.asarray(probability, dtype=float), 1e-5, 1 - 1e-5)
    y = np.asarray(target, dtype=int)
    if len(p) < 100 or len(np.unique(y)) != 2:
        return CalibrationMap()
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
    return CalibrationMap(float(model.coef_[0, 0]), float(model.intercept_[0]))


def _strategy(report: dict[str, Any]) -> dict[str, Any]:
    strategy = _snapshot(report, "strategy")
    if strategy:
        return strategy
    costs = report.get("execution_costs", {})
    if not isinstance(costs, dict):
        costs = {}
    base = float(costs.get("base_cost_bps", 11.0))
    stress = float(costs.get("stress_cost_bps", 16.5))
    return {
        "entry_order_style": "maker",
        "exit_order_style": "taker",
        "maker_fee_bps": costs.get("entry_fee_bps", 2.0),
        "taker_fee_bps": costs.get("exit_fee_bps", 5.0),
        "entry_slippage_bps": costs.get("entry_slippage_bps", 1.5),
        "exit_slippage_bps": costs.get("exit_slippage_bps", 2.5),
        "stress_cost_multiplier": stress / max(base, 1e-9),
    }


def _snapshot(report: dict[str, Any], name: str) -> dict[str, Any]:
    snapshot = report.get("config_snapshot", {})
    value = snapshot.get(name) if isinstance(snapshot, dict) else None
    return dict(value) if isinstance(value, dict) else {}


def _failed(samples: int, reason: str) -> dict[str, Any]:
    return {
        "passed": False,
        "event_samples": int(samples),
        "development_samples": 0,
        "holdout_samples": 0,
        "policy": None,
        "development": None,
        "holdout": None,
        "development_objective_bps": None,
        "holdout_objective_bps": None,
        "blockers": [reason],
    }


def _empty(stress_bps: float, blocker: str) -> dict[str, Any]:
    qualified = {name: [] for name in DIRECTIONS}
    qualification = {
        "passed": False,
        "qualified_horizons": [],
        "qualified_directions": qualified,
        "per_direction": {name: {} for name in DIRECTIONS},
        "blockers": [blocker],
        "economic_policy": {name: {} for name in DIRECTIONS},
        "economic_stress_cost_bps": stress_bps,
        "economic_validation_version": 1,
    }
    return {
        "schema_version": 1,
        "method": "LOCKED_CHRONOLOGICAL_DEVELOPMENT_HOLDOUT",
        "development_fraction": DEV_FRACTION,
        "economic_stress_cost_bps": stress_bps,
        "qualification": qualification,
        "aggregate_holdout_objective_bps": None,
        "passed_pairs": 0,
    }


def _promotion(
    decision: str,
    reason: str,
    candidate_score: float | None,
    incumbent_score: float | None,
) -> dict[str, Any]:
    return {
        "decision": decision,
        "reason": reason,
        "candidate_score_bps": candidate_score,
        "incumbent_score_bps": incumbent_score,
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
