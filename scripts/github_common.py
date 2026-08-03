from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from btc_ema_trader.config import Settings, load_settings


def build_github_settings(
    project_root: Path,
    runtime_root: Path,
    *,
    model_dir: Path | None = None,
    report_dir: Path | None = None,
    adaptive_state_dir: Path | None = None,
) -> Settings:
    source = project_root / "config" / "default.yaml"
    values = yaml.safe_load(source.read_text(encoding="utf-8")) or {}

    runtime_root.mkdir(parents=True, exist_ok=True)
    model_dir = (
        model_dir or project_root / "artifacts" / "models"
    ).resolve()
    report_dir = (
        report_dir or project_root / "artifacts" / "reports"
    ).resolve()
    adaptive_state_dir = (
        adaptive_state_dir or project_root / ".adaptive_state"
    ).resolve()

    values.setdefault("paths", {})
    values["paths"].update(
        {
            "database": str(
                (runtime_root / "btc_hourly.sqlite3").resolve()
            ),
            "runtime_state": str(
                (runtime_root / "runtime_state.json").resolve()
            ),
            "adaptive_state": str(
                (adaptive_state_dir / "adaptive_state.joblib").resolve()
            ),
            "log_dir": str((runtime_root / "logs").resolve()),
            "model_dir": str(model_dir),
            "report_dir": str(report_dir),
        }
    )
    values.setdefault("model", {})["auto_retrain_days"] = 10
    values.setdefault("live", {}).update(
        {
            "auto_retrain": False,
            "collect_recent_news_each_cycle": True,
            "start_on_next_closed_candle": False,
        }
    )

    config_path = runtime_root / "github.yaml"
    config_path.write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
    )
    return load_settings(config_path)


def copy_latest_model_from_state(
    project_root: Path,
    model_state_dir: Path,
) -> bool:
    source = model_state_dir / "latest.joblib"
    if not source.exists():
        return False
    target = project_root / "artifacts" / "models" / "latest.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    for filename in (
        "latest_training_report.json",
        "latest_metrics.csv",
        "model_metadata.json",
    ):
        candidate = model_state_dir / filename
        if candidate.exists():
            destination = project_root / "artifacts" / "reports" / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
    return True


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
