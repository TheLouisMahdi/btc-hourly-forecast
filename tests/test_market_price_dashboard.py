from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import github_market_price_dashboard as market_price


class MarketPriceDashboardTests(unittest.TestCase):
    def test_latest_chart_close_does_not_depend_on_forecast(self) -> None:
        candles = [
            {"open_time": "2026-08-11T08:00:00+00:00", "close": 118000.0},
            {"open_time": "2026-08-11T09:00:00+00:00", "close": 118250.5},
        ]

        value, timestamp = market_price._latest_market_price(candles, {})

        self.assertEqual(value, 118250.5)
        self.assertEqual(timestamp, "2026-08-11T09:00:00+00:00")

    def test_forecast_reference_is_only_last_fallback(self) -> None:
        latest = {
            "next_candle_forecast": {
                "reference_close": 117500.0,
                "source_open_time": "2026-08-11T07:00:00+00:00",
            }
        }

        value, timestamp = market_price._latest_market_price([], latest)

        self.assertEqual(value, 117500.0)
        self.assertEqual(timestamp, "2026-08-11T07:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
