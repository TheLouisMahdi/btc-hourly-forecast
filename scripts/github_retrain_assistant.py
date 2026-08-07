from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

import github_weekly_retrain as base_retrain
from btc_ema_trader.contract_training import build_segmented_feature_set
from btc_ema_trader.meta_filter import (
    build_exit_profile,
    save_precision_meta_filter,
    train_precision_meta_filter,
)
from btc_ema_trader.pattern_memory import (
    LONG,
    SHORT,
    build_static_pattern_bundle,
    save_static_pattern_bundle,
)
from btc_ema_trader.storage import Database

from github_common import build_github_settings, write_json

LOGGER = logging.getLogger("github_retrain_assistant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical retraining and attach the precision-first "
            "trade-assistant challenger artifacts"
        )
    )
    parser.add_argument("--output-dir", default="model-state-output")
    parser.add_argument("--runtime-dir", default=".github_runtime/training")
    parser.add_argument("--incumbent-dir", default="incumbent-model-state")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_exit = base_retrain.main()
    if base_exit != 0:
        return int(base_exit)

    root = Path(__file__).resolve().parents[1]
    runtime_dir = (root / args.runtime_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    report_dir = runtime_dir / "reports"
    model_dir = runtime_dir / "models"
    promotion_path = output_dir / "promotion.json"

    try:
        candidate_report = _load_json(
            output_dir / "candidate_training_report.json"
        )
        model_id = str(candidate_report.get("model_id") or "")
        provider = str(candidate_report.get("provider") or "")
        if not model_id or not provider:
            raise RuntimeError(
                "Candidate training report is missing model_id/provider"
            )

        settings = build_github_settings(
            root,
            runtime_dir,
            model_dir=model_dir,
            report_dir=report_dir,
        )
        database = Database(settings)
        database.initialize()
        symbol = str(
            settings.section("market").get("symbol", "BTCUSDT")
        )
        candles = database.load_candles(
            provider=provider,
            symbol=symbol,
        )
        news = database.load_news(
            start=candles["open_time"].min(),
            end=candles["open_time"].max() + pd.Timedelta(hours=1),
        )
        feature_set, segmentation = build_segmented_feature_set(
            candles,
            news,
            settings,
            include_labels=True,
        )
        oof_path = report_dir / f"{model_id}_oof.csv"
        if not oof_path.exists():
            raise FileNotFoundError(
                f"Candidate OOF records are unavailable: {oof_path}"
            )
        oof = pd.read_csv(oof_path)

        patterns, pattern_report = build_static_pattern_bundle(
            feature_set.frame,
            settings,
            model_id=model_id,
        )
        meta, meta_report = train_precision_meta_filter(
            feature_set.frame,
            oof,
            settings,
            model_id=model_id,
        )
        _lock_exit_profiles_before_meta_holdout(
            meta,
            meta_report,
            feature_set.frame,
            oof,
            settings,
        )
        pattern_report["market_data_segmentation"] = segmentation

        runtime_pattern = runtime_dir / "trade_assistant_patterns.joblib"
        runtime_meta = runtime_dir / "trade_assistant_meta.joblib"
        save_static_pattern_bundle(patterns, runtime_pattern)
        save_precision_meta_filter(meta, runtime_meta)
        write_json(
            report_dir / "trade_assistant_pattern_report.json",
            pattern_report,
        )
        write_json(
            report_dir / "trade_assistant_meta_report.json",
            meta_report,
        )

        shutil.copy2(
            runtime_pattern,
            output_dir / "candidate_trade_assistant_patterns.joblib",
        )
        shutil.copy2(
            runtime_meta,
            output_dir / "candidate_trade_assistant_meta.joblib",
        )
        shutil.copy2(
            report_dir / "trade_assistant_pattern_report.json",
            output_dir / "candidate_trade_assistant_pattern_report.json",
        )
        shutil.copy2(
            report_dir / "trade_assistant_meta_report.json",
            output_dir / "candidate_trade_assistant_meta_report.json",
        )

        promotion = _load_json(promotion_path)
        decision = str(
            promotion.get("decision") or "KEEP_INCUMBENT"
        ).upper()
        assistant_passed = bool(meta_report.get("passed", False))
        if decision == "PROMOTE" and not assistant_passed:
            promotion["decision"] = "KEEP_INCUMBENT"
            promotion["reason"] = (
                "Candidate model passed prior gates but no precision meta "
                "head passed the locked holdout gate"
            )
        elif decision == "PROMOTE":
            shutil.copy2(
                runtime_pattern,
                output_dir / "trade_assistant_patterns.joblib",
            )
            shutil.copy2(
                runtime_meta,
                output_dir / "trade_assistant_meta.joblib",
            )
            shutil.copy2(
                report_dir / "trade_assistant_pattern_report.json",
                output_dir / "trade_assistant_pattern_report.json",
            )
            shutil.copy2(
                report_dir / "trade_assistant_meta_report.json",
                output_dir / "trade_assistant_meta_report.json",
            )

        promotion["trade_assistant_passed"] = assistant_passed
        promotion["trade_assistant_qualified_heads"] = int(
            meta_report.get("qualified_heads", 0)
        )
        write_json(promotion_path, promotion)
        _patch_metadata(
            output_dir / "model_metadata.json",
            {
                "status": "READY",
                "meta_filter": meta_report,
                "pattern_memory": pattern_report,
            },
        )
    except Exception as exc:
        LOGGER.exception("Trade-assistant challenger training failed")
        promotion = _load_json(promotion_path)
        if str(promotion.get("decision") or "").upper() == "PROMOTE":
            promotion["decision"] = "KEEP_INCUMBENT"
            promotion["reason"] = (
                "Trade-assistant challenger training failed; incumbent "
                "champion preserved"
            )
        promotion["trade_assistant_passed"] = False
        promotion["trade_assistant_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
        write_json(promotion_path, promotion)
        _patch_metadata(
            output_dir / "model_metadata.json",
            {
                "status": "FAILED_SAFE",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
    return 0


def _lock_exit_profiles_before_meta_holdout(
    meta: Any,
    report: dict[str, Any],
    frame: pd.DataFrame,
    oof: pd.DataFrame,
    settings: Any,
) -> None:
    """Keep the final 15% event OOF interval unseen by exit-profile fitting."""
    data = frame.copy().sort_values("open_time").reset_index(drop=True)
    event_oof = oof.loc[oof["record_type"].astype(str) == "EVENT"].copy()
    event_oof["open_time"] = pd.to_datetime(event_oof["open_time"], utc=True)
    locked: dict[str, dict[int, dict[str, float]]] = {
        "LONG": {},
        "SHORT": {},
    }
    for direction, name in ((LONG, "LONG"), (SHORT, "SHORT")):
        for horizon in settings.section("model").get(
            "trade_horizons_hours", [3, 6, 12]
        ):
            horizon = int(horizon)
            rows = event_oof.loc[
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
            ].sort_values("open_time")
            if len(rows) < 100:
                profile_frame = data.iloc[0:0]
                cutoff = None
            else:
                holdout_start = min(
                    max(1, int(len(rows) * 0.85)),
                    len(rows) - 1,
                )
                cutoff = pd.Timestamp(rows.iloc[holdout_start]["open_time"])
                profile_frame = data.loc[
                    pd.to_datetime(data["open_time"], utc=True) < cutoff
                ]
            profile = build_exit_profile(
                data=profile_frame,
                direction=direction,
                horizon=horizon,
                settings=settings,
            )
            profile["fit_cutoff"] = (
                cutoff.isoformat() if cutoff is not None else None
            )
            profile["locked_holdout_fraction"] = 0.15
            locked[name][horizon] = profile
            meta.exit_profiles[name][horizon] = profile
    report["exit_profiles"] = {
        name: {str(horizon): item for horizon, item in values.items()}
        for name, values in locked.items()
    }
    report["exit_profile_contract"] = (
        "FITTED_BEFORE_FINAL_15_PERCENT_META_HOLDOUT"
    )


def _patch_metadata(path: Path, assistant: dict[str, Any]) -> None:
    payload = _load_json(path)
    payload["trade_assistant"] = assistant
    write_json(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
