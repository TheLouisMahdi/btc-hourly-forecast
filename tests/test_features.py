import numpy as np
import pandas as pd

from btc_ema_trader.features import build_feature_set


def test_regime_events_and_binary_labels(candles, settings):
    result = build_feature_set(candles, pd.DataFrame(), settings, include_labels=True)
    frame = result.frame
    required = {"kama", "adx", "donchian_high", "event_type", "event_id", "event_direction", "is_event"}
    assert required.issubset(frame.columns)
    assert int(frame["is_event"].sum()) > 20
    event_ids = frame.loc[frame["is_event"] == 1, "event_id"].dropna()
    assert event_ids.is_unique
    values = set(frame["target_up_h1"].dropna().astype(int).unique())
    assert values.issubset({0, 1})
    assert "target_neutral" not in frame.columns


def test_labels_use_next_candle_open(candles, settings):
    frame = build_feature_set(candles, pd.DataFrame(), settings, include_labels=True).frame
    idx = 300
    expected = candles.iloc[idx + 1]["close"] / candles.iloc[idx + 1]["open"] - 1
    assert np.isclose(frame.iloc[idx]["future_return_h1"], expected)


def test_future_edit_does_not_change_old_features(candles, settings):
    first = build_feature_set(candles, pd.DataFrame(), settings, include_labels=True).frame
    changed = candles.copy()
    changed.loc[changed.index[-1], "close"] *= 1.4
    second = build_feature_set(changed, pd.DataFrame(), settings, include_labels=True).frame
    assert np.isclose(first.iloc[-20]["kama"], second.iloc[-20]["kama"])



def test_tradeability_is_event_direction_and_path_aware(candles, settings):
    frame = build_feature_set(candles, pd.DataFrame(), settings, include_labels=True).frame
    events = frame[(frame["is_event"] == 1) & frame["tradeable_h1"].notna()].copy()
    assert not events.empty
    cost = 0.0011
    buffer = 0.0008
    expected = events["event_gross_return_h1"] - cost >= buffer
    assert np.array_equal(events["tradeable_h1"].astype(bool).to_numpy(), expected.to_numpy())
    stopped = events[events["barrier_outcome_h1"] == -1]
    if not stopped.empty:
        assert (stopped["event_gross_return_h1"] < 0).all()
