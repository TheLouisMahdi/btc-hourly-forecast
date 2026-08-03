from __future__ import annotations

import json

import numpy as np
import pandas as pd

from btc_ema_trader.storage import Database


def test_log_event_serializes_pandas_and_numpy_values(settings):
    database = Database(settings)
    database.initialize()

    database.log_event(
        "INFO",
        "SERIALIZATION_TEST",
        "runtime payload",
        {
            "timestamp": pd.Timestamp("2026-08-03T14:00:00Z"),
            "duration": pd.Timedelta(hours=2),
            "integer": np.int64(7),
            "floating": np.float64(0.75),
        },
    )

    row = database.recent_events(limit=1).iloc[0]
    details = json.loads(row["details_json"])

    assert details == {
        "timestamp": "2026-08-03T14:00:00+00:00",
        "duration": "P0DT2H0M0S",
        "integer": 7,
        "floating": 0.75,
    }
