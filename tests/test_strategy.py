from types import SimpleNamespace

import pandas as pd

from btc_ema_trader.strategy import make_decision


def test_fail_safe_wait_is_not_neutral(settings):
    bundle = SimpleNamespace(qualification={"passed": False, "qualified_horizons": []})
    row = pd.Series({
        "close": 70000,
        "atr": 500,
        "atr_pct": 0.007,
        "is_event": 1,
        "event_id": "EV1",
        "event_type": "DONCHIAN_BREAKOUT",
        "event_direction": 1,
        "regime": "TREND_UP",
        "news_shock": 0,
    })
    prediction = {
        "direction": "UP",
        "confidence": 0.72,
        "agreement": 1.0,
        "expected_return": 0.009,
        "selected_horizon": 2,
        "probabilities": {1: .7, 2: .72, 3: .65},
        "tradeability": {1: .6, 2: .72, 3: .65},
        "returns": {1: .004, 2: .009, 3: .007},
    }
    decision = make_decision(row, prediction, bundle, settings)
    assert decision.forecast_direction == "UP"
    assert decision.action == "WAIT"
    assert "MODEL_NOT_QUALIFIED" in decision.blockers


def test_no_event_blocks_trade_even_with_strong_forecast(settings):
    bundle = SimpleNamespace(qualification={"passed": True, "qualified_horizons": [2]})
    row = pd.Series({"close": 70000, "atr": 500, "atr_pct": .007, "is_event": 0, "event_direction": 0, "regime": "RANGE", "news_shock": 0})
    prediction = {"direction":"UP","confidence":.8,"agreement":1.0,"expected_return":.02,"selected_horizon":2,"probabilities":{1:.7,2:.8,3:.7},"tradeability":{1:.7,2:.8,3:.7},"returns":{1:.01,2:.02,3:.015}}
    decision = make_decision(row, prediction, bundle, settings)
    assert decision.action == "WAIT"
    assert "NO_NEW_MARKET_EVENT" in decision.blockers
