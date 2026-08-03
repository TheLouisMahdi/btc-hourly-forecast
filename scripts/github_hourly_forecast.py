from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from btc_ema_trader.logging_setup import configure_logging
from btc_ema_trader.market import fetch_and_store
from btc_ema_trader.model import latest_bundle
from btc_ema_trader.runtime import RuntimeEngine
from btc_ema_trader.storage import Database

from github_common import (
    build_github_settings,
    copy_latest_model_from_state,
    json_safe,
    write_json,
)

LOGGER = logging.getLogger("github_hourly_forecast")
MAX_HISTORY = 24 * 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one adaptive hourly forecast and persist its state"
    )
    parser.add_argument("--state-dir", default=".github_state")
    parser.add_argument("--model-state-dir", default=".model_state")
    parser.add_argument("--adaptive-state-dir", default=".adaptive_state")
    parser.add_argument("--site-dir", default="site")
    parser.add_argument("--runtime-dir", default=".github_runtime/hourly")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    state_dir = (root / args.state_dir).resolve()
    model_state_dir = (root / args.model_state_dir).resolve()
    adaptive_state_dir = (root / args.adaptive_state_dir).resolve()
    site_dir = (root / args.site_dir).resolve()
    runtime_dir = (root / args.runtime_dir).resolve()

    shutil.rmtree(runtime_dir, ignore_errors=True)
    shutil.rmtree(site_dir, ignore_errors=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    adaptive_state_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)

    used_weekly_model = copy_latest_model_from_state(
        root,
        model_state_dir,
    )
    settings = build_github_settings(
        root,
        runtime_dir,
        adaptive_state_dir=adaptive_state_dir,
    )
    configure_logging(settings, verbose=True)
    database = Database(settings)
    database.initialize()

    started_at = pd.Timestamp.now(tz="UTC")
    try:
        bundle = latest_bundle(settings)
        market = fetch_and_store(
            settings,
            database,
            days=180,
            provider=bundle.provider,
        )
        result = RuntimeEngine(settings, database).run_once(force=True)
        status = (
            "OK"
            if result.get("status") != "FAIL_SAFE"
            else "FAIL_SAFE"
        )
    except Exception as exc:
        LOGGER.exception("Hourly forecast failed")
        market = None
        result = {
            "status": "FAIL_SAFE",
            "error": f"{type(exc).__name__}: {exc}",
        }
        status = "FAIL_SAFE"

    finished_at = pd.Timestamp.now(tz="UTC")
    record = json_safe(
        {
            **result,
            "run_status": status,
            "run_started_at": started_at,
            "run_finished_at": finished_at,
            "run_duration_seconds": (
                finished_at - started_at
            ).total_seconds(),
            "market_refresh": market,
            "weekly_model_loaded": used_weekly_model,
        }
    )

    history_path = state_dir / "history.json"
    history = append_unique(
        load_history(history_path),
        record,
    )[-MAX_HISTORY:]
    write_json(state_dir / "latest.json", record)
    write_json(history_path, history)
    write_json(site_dir / "latest.json", record)
    write_json(site_dir / "history.json", history)
    write_json(
        adaptive_state_dir / "summary.json",
        record.get("adaptive", {"status": "UNAVAILABLE"}),
    )
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(json.dumps(record, ensure_ascii=False, indent=2))
    if status != "OK":
        print(
            "::warning::Forecast completed in FAIL_SAFE mode; "
            "diagnostics were persisted."
        )
    return 0


def load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def append_unique(
    history: list[dict[str, Any]],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    key = record.get("candle_time") or record.get("run_finished_at")
    filtered = [
        item
        for item in history
        if (
            item.get("candle_time") or item.get("run_finished_at")
        )
        != key
    ]
    filtered.append(record)
    return sorted(
        filtered,
        key=lambda item: str(
            item.get("candle_time")
            or item.get("run_finished_at")
            or ""
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
