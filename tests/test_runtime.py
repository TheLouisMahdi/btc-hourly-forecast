import pandas as pd
from btc_ema_trader.runtime import next_hour_boundary


def test_next_hour_boundary():
    ts = pd.Timestamp("2026-08-02T14:10:00Z")
    assert next_hour_boundary(ts) == pd.Timestamp("2026-08-02T15:00:00Z")
