from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULTS = {
    "enabled": True,
    "minimum_model_age_days": 30.0,
    "minimum_evaluation_samples": 504,
    "minimum_base_direction_accuracy": 0.505,
    "minimum_online_accuracy_gain": 0.020,
    "minimum_online_brier_gain": 0.003,
    "successful_dispatch_cooldown_hours": 168.0,
    "failed_dispatch_cooldown_hours": 6.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decide whether the heavyweight challenger workflow is needed"
    )
    parser.add_argument("--state-dir", default=".github_state")
    parser.add_argument("--adaptive-state-dir", default=".adaptive_state")
    parser.add_argument("--model-state-dir", default=".model_state")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    parser.add_argument(
        "--record-dispatch",
        choices=("SUCCESS", "FAILURE"),
        default=None,
    )
    parser.add_argument("--reason", default="UNKNOWN")
    return parser.parse_args()


def evaluate_policy(
    latest: dict[str, Any],
    price_summary: dict[str, Any],
    metadata: dict[str, Any],
    control: dict[str, Any],
    *,
    now: pd.Timestamp | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = _utc(now or pd.Timestamp.now(tz="UTC"))
    cfg = {**DEFAULTS, **(config or {})}
    market = latest.get("market_refresh")
    market = market if isinstance(market, dict) else {}
    freshness = market.get("freshness")
    freshness = freshness if isinstance(freshness, dict) else {}
    data_ready = bool(
        latest.get("run_status") == "OK"
        and freshness.get("fresh") is True
    )

    result: dict[str, Any] = {
        "checked_at": current.isoformat(),
        "required": False,
        "reason": "NOT_REQUIRED",
        "status": "IDLE",
        "data_ready": data_ready,
        "automatic": True,
    }
    if not bool(cfg["enabled"]):
        result.update(status="DISABLED", reason="AUTO_RETRAIN_DISABLED")
        return result
    if not data_ready:
        result.update(status="WAITING_FOR_FRESH_DATA", reason="DATA_NOT_FRESH")
        return result

    cooldown = _cooldown_status(control, current, cfg)
    if cooldown is not None:
        result.update(cooldown)
        return result

    metrics = price_summary.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    samples = int(_number(metrics.get("samples"), 0.0))
    base_accuracy = _optional_number(metrics.get("base_direction_accuracy"))
    online_accuracy = _optional_number(metrics.get("online_direction_accuracy"))
    base_brier = _optional_number(metrics.get("base_direction_brier"))
    online_brier = _optional_number(metrics.get("online_direction_brier"))
    minimum_samples = int(cfg["minimum_evaluation_samples"])

    training = metadata.get("training")
    training = training if isinstance(training, dict) else {}
    created = _timestamp(
        training.get("created_at")
        or metadata.get("finished_at")
        or metadata.get("created_at")
    )
    model_age_days = (
        float((current - created).total_seconds() / 86400.0)
        if created is not None
        else None
    )
    result.update(
        {
            "model_created_at": None if created is None else created.isoformat(),
            "model_age_days": model_age_days,
            "evaluation_samples": samples,
            "base_direction_accuracy": base_accuracy,
            "online_direction_accuracy": online_accuracy,
            "base_direction_brier": base_brier,
            "online_direction_brier": online_brier,
        }
    )

    health = latest.get("data_health")
    health = health if isinstance(health, dict) else {}
    if health.get("model_stale") is True:
        return _request(result, "MODEL_MARKED_STALE")

    if (
        model_age_days is not None
        and model_age_days >= float(cfg["minimum_model_age_days"])
    ):
        return _request(result, "MODEL_AGE_LIMIT")

    if (
        samples >= minimum_samples
        and base_accuracy is not None
        and base_accuracy < float(cfg["minimum_base_direction_accuracy"])
    ):
        return _request(result, "BATCH_DIRECTION_DEGRADED")

    if (
        samples >= minimum_samples
        and base_accuracy is not None
        and online_accuracy is not None
        and base_brier is not None
        and online_brier is not None
        and online_accuracy - base_accuracy
        >= float(cfg["minimum_online_accuracy_gain"])
        and base_brier - online_brier
        >= float(cfg["minimum_online_brier_gain"])
    ):
        return _request(result, "ONLINE_LEARNER_OUTPERFORMS_BATCH")

    if created is None:
        return _request(result, "MODEL_TIMESTAMP_UNAVAILABLE")
    return result


def record_dispatch(
    latest: dict[str, Any],
    control: dict[str, Any],
    status: str,
    reason: str,
    *,
    now: pd.Timestamp | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current = _utc(now or pd.Timestamp.now(tz="UTC"))
    normalized = status.upper()
    updated_control = dict(control)
    updated_control.update(
        {
            "last_attempt_at": current.isoformat(),
            "last_attempt_status": normalized,
            "last_reason": reason,
        }
    )
    if normalized == "SUCCESS":
        updated_control["last_dispatch_at"] = current.isoformat()
    updated_latest = dict(latest)
    policy = updated_latest.get("retrain_policy")
    policy = dict(policy) if isinstance(policy, dict) else {}
    policy.update(
        {
            "dispatch_status": normalized,
            "dispatch_reason": reason,
            "dispatch_recorded_at": current.isoformat(),
        }
    )
    updated_latest["retrain_policy"] = policy
    return updated_latest, updated_control


def main() -> int:
    args = parse_args()
    state_dir = Path(args.state_dir)
    adaptive_dir = Path(args.adaptive_state_dir)
    model_dir = Path(args.model_state_dir)
    latest_path = state_dir / "latest.json"
    control_path = state_dir / "retrain_control.json"
    latest = _load_dict(latest_path)
    control = _load_dict(control_path)

    if args.record_dispatch:
        latest, control = record_dispatch(
            latest,
            control,
            args.record_dispatch,
            args.reason,
        )
        _write_json(latest_path, latest)
        _write_json(control_path, control)
        _write_outputs(
            args.github_output,
            {"required": "false", "reason": args.reason},
        )
        return 0

    price_summary = _load_dict(adaptive_dir / "price_summary.json")
    metadata = _load_dict(model_dir / "model_metadata.json")
    config = _load_policy_config(Path(args.config))
    decision = evaluate_policy(
        latest,
        price_summary,
        metadata,
        control,
        config=config,
    )
    latest["retrain_policy"] = decision
    control["last_checked_at"] = decision["checked_at"]
    control["last_decision"] = decision["reason"]
    _write_json(latest_path, latest)
    _write_json(control_path, control)
    _write_outputs(
        args.github_output,
        {
            "required": "true" if decision["required"] else "false",
            "reason": str(decision["reason"]),
            "status": str(decision["status"]),
        },
    )
    print(json.dumps(decision, indent=2))
    return 0


def _request(result: dict[str, Any], reason: str) -> dict[str, Any]:
    output = dict(result)
    output.update(required=True, reason=reason, status="AUTO_REQUESTED")
    return output


def _cooldown_status(
    control: dict[str, Any],
    now: pd.Timestamp,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    success = _timestamp(control.get("last_dispatch_at"))
    if success is not None:
        elapsed = float((now - success).total_seconds() / 3600.0)
        required = float(config["successful_dispatch_cooldown_hours"])
        if elapsed < required:
            return {
                "status": "COOLDOWN",
                "reason": "SUCCESSFUL_DISPATCH_COOLDOWN",
                "cooldown_remaining_hours": max(0.0, required - elapsed),
            }
    if str(control.get("last_attempt_status") or "").upper() == "FAILURE":
        failure = _timestamp(control.get("last_attempt_at"))
        if failure is not None:
            elapsed = float((now - failure).total_seconds() / 3600.0)
            required = float(config["failed_dispatch_cooldown_hours"])
            if elapsed < required:
                return {
                    "status": "COOLDOWN",
                    "reason": "FAILED_DISPATCH_COOLDOWN",
                    "cooldown_remaining_hours": max(0.0, required - elapsed),
                }
    return None


def _load_policy_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    value = payload.get("auto_retraining")
    return value if isinstance(value, dict) else {}


def _load_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _write_outputs(path: str | None, values: dict[str, str]) -> None:
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as handle:
        for key, value in values.items():
            safe = value.replace("\n", " ").replace("\r", " ")
            handle.write(f"{key}={safe}\n")


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if pd.notna(number) else float(default)


def _optional_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _timestamp(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return _utc(value)
    except Exception:
        return None


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


if __name__ == "__main__":
    raise SystemExit(main())
