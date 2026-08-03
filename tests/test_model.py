import pandas as pd

from btc_ema_trader.features import build_feature_set
from btc_ema_trader.model import latest_bundle
from btc_ema_trader.training import train_feature_set


def test_training_outputs_regime_v2_model(candles, settings):
    fs = build_feature_set(candles, pd.DataFrame(), settings, include_labels=True)
    report = train_feature_set(settings, fs, provider="synthetic", symbol="BTCUSDT")
    assert report["model_id"].startswith("regime-meta-hourly-")
    assert report["schema_version"] == 3
    bundle = latest_bundle(settings)
    usable = fs.frame.dropna(subset=["kama", "adx", "atr"]).tail(1)
    prediction = bundle.predict_frame(usable)
    assert prediction["direction"] in {"UP", "DOWN"}
    assert 0.5 <= prediction["confidence"] <= 1.0
    assert set(prediction["probabilities"]) == {1, 2, 3}
    assert set(prediction["tradeability"]) == {1, 2, 3}
    assert set(prediction["continuation"]) == {1, 2, 3}
    assert set(prediction["event_returns"]) == {1, 2, 3}
