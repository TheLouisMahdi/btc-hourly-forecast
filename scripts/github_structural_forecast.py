from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import github_hourly_forecast

MODEL_PREFIX = "structure-breakout-hourly-"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    state_dir = root / ".github_state"
    history_path = state_dir / "history.json"
    latest_path = state_dir / "latest.json"
    history = _load_list(history_path)
    structural_history = [
        item
        for item in history
        if _is_structural_record(item)
    ]
    if structural_history != history:
        state_dir.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                structural_history,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        latest = _load_dict(latest_path)
        if latest and not _is_structural_record(latest):
            latest_path.write_text("{}\n", encoding="utf-8")
    return github_hourly_forecast.main()


def _is_structural_record(item: dict[str, Any]) -> bool:
    return str(item.get("model_id") or "").startswith(MODEL_PREFIX)


def _load_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _load_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
