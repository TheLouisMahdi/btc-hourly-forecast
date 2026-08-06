from __future__ import annotations

import unittest

from scripts.github_resilience_dashboard import (
    _history_gap_summary,
    _time_aware_chart,
)


class ResilienceDashboardTests(unittest.TestCase):
    def test_chart_breaks_line_across_missing_workflow_hours(self) -> None:
        history = []
        for hour, price in ((0, 100.0), (1, 101.0), (4, 102.0), (5, 103.0)):
            history.append(
                {
                    "candle_time": f"2026-01-01T{hour:02d}:00:00Z",
                    "price": price,
                    "direction_result": "DIRECTION_CORRECT",
                    "next_candle_forecast": {
                        "direction": "UP",
                        "likely_close_low": 99.0,
                        "likely_close_high": 104.0,
                    },
                }
            )
        rendered = _time_aware_chart(history)
        self.assertEqual(rendered.count('<path d="'), 2)
        self.assertIn("2h not published", rendered)
        self.assertIn("Missing workflow record; not interpolated", rendered)
        self.assertNotIn('class="median"', rendered)

    def test_gap_summary_counts_only_unpublished_hours(self) -> None:
        history = [
            {"candle_time": "2026-01-01T00:00:00Z"},
            {"candle_time": "2026-01-01T01:00:00Z"},
            {"candle_time": "2026-01-01T05:00:00Z"},
        ]
        missing, largest = _history_gap_summary(history)
        self.assertEqual(missing, 3)
        self.assertEqual(largest, 3)


if __name__ == "__main__":
    unittest.main()
