from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from btc_ema_trader.config import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    root = Path(__file__).resolve().parents[1]
    values = yaml.safe_load((root / "config" / "default.yaml").read_text())
    values = deepcopy(values)
    values["paths"] = {
        "database": "data/test.sqlite3",
        "model_dir": "artifacts/models",
        "report_dir": "artifacts/reports",
        "log_dir": "logs",
        "runtime_state": "data/runtime_state.json",
    }
    values["model"].update({
        "min_train_rows": 500,
        "minimum_training_events": 20,
        "max_iter": 35,
        "walk_forward_splits": 3,
        "min_samples_leaf": 15,
    })
    values["qualification"].update({
        "minimum_auc": 0.0,
        "minimum_balanced_accuracy": 0.0,
        "minimum_event_auc": 0.0,
        "minimum_tradeability_auc": 0.0,
        "minimum_event_samples": 1,
        "maximum_calibration_error": 1.0,
        "minimum_positive_fold_fraction": 0.0,
        "minimum_qualified_horizons": 1,
        "require_positive_oof_expectancy": False,
    })
    result = Settings(tmp_path, values)
    result.ensure_runtime_dirs()
    return result


@pytest.fixture()
def candles() -> pd.DataFrame:
    n = 1800
    rng = np.random.default_rng(42)
    t = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    cycle = np.sin(np.arange(n) / 34) * 0.0019
    regime = np.where(np.sin(np.arange(n) / 150) > 0, 0.0005, -0.0004)
    impulses = np.zeros(n)
    impulses[::47] = rng.choice([-0.012, 0.012], size=len(impulses[::47]))
    ret = regime + cycle + impulses + rng.normal(0, 0.0032, n)
    close = 65_000 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0025, 0.001, n)) * close
    volume = rng.lognormal(7, 0.5, n)
    volume[::47] *= 5
    return pd.DataFrame({
        "provider": "synthetic",
        "symbol": "BTCUSDT",
        "open_time": t,
        "open": open_,
        "high": np.maximum(open_, close) + spread,
        "low": np.minimum(open_, close) - spread,
        "close": close,
        "volume": volume,
        "quote_volume": volume * close,
        "trades": rng.integers(1000, 9000, n),
        "closed": 1,
        "fetched_at": t,
    })
