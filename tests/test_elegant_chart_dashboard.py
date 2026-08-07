from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import github_chart_dashboard


class ElegantChartDashboardTests(unittest.TestCase):
    def _history(self) -> list[dict[str, object]]:
        start = pd.Timestamp("2026-08-01T00:00:00Z")
        prices = [100000, 100400, 100150, 100900, 101200, 101050, 101500, 101800]
        rows: list[dict[str, object]] = []
        for index, price in enumerate(prices):
            hour = index if index < 4 else index + 2
            result = (
                "DIRECTION_CORRECT"
                if index % 2 == 0
                else "DIRECTION_WRONG"
            )
            rows.append(
                {
                    "candle_time": (start + pd.Timedelta(hours=hour)).isoformat(),
                    "price": price,
                    "direction_result": result,
                    "next_candle_forecast": {
                        "direction": "UP" if index % 2 == 0 else "DOWN"
                    },
                }
            )
        return rows

    def test_chart_is_price_first_and_gap_aware(self) -> None:
        latest = {
            "next_candle_forecast": {
                "direction": "UP",
                "likely_close_low": 101300,
                "likely_close_high": 102100,
                "target_close_time": "2026-08-01T11:00:00Z",
            }
        }
        chart = github_chart_dashboard._chart(self._history(), latest)

        self.assertIn('data-elegant-chart="v2"', chart)
        self.assertIn('class="price-line"', chart)
        self.assertIn('class="price-area"', chart)
        self.assertIn('class="forecast-band"', chart)
        self.assertIn("Likely range", chart)
        self.assertIn("⋯ Data gap", chart)
        self.assertNotIn("gap-line", chart)
        self.assertNotIn("not published</text>", chart)
        self.assertLessEqual(chart.count('class="outcome-marker'), 6)

    def test_theme_tokens_cover_light_dark_and_control_spacing(self) -> None:
        styles = github_chart_dashboard._styles()

        self.assertIn(":root{", styles)
        self.assertIn(':root[data-theme="dark"]', styles)
        self.assertIn("--chart-price", styles)
        self.assertIn("--chart-range", styles)
        self.assertIn(".health-badge{padding:10px 12px!important}", styles)
        self.assertIn(".theme-toggle{padding:9px 13px 9px 10px!important", styles)

    def test_mobile_chart_reduces_labels_without_hiding_price_path(self) -> None:
        styles = github_chart_dashboard._styles()
        self.assertIn("@media(max-width:620px)", styles)
        self.assertIn(".chart-x-label.x-mobile-hide{display:none}", styles)
        self.assertNotIn(".price-line{display:none", styles)


if __name__ == "__main__":
    unittest.main()
