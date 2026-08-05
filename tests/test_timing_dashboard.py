from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "scripts"),
)

from github_timing_dashboard import _timing_strip


class TimingDashboardTests(unittest.TestCase):
    def test_exact_timing_strip_uses_forecast_and_target_times(self) -> None:
        document = _timing_strip(
            {
                "forecast_created_at": "2026-01-01T01:04:00Z",
                "source_close_time": "2026-01-01T01:00:00Z",
                "target_open_time": "2026-01-01T01:00:00Z",
                "target_close_time": "2026-01-01T02:00:00Z",
                "forecast_horizon_seconds": 3360,
                "timing_status": "EXACT_NEXT_CLOSED_CANDLE",
            }
        )
        self.assertIn("01:04:00 UTC", document)
        self.assertIn("02:00:00 UTC", document)
        self.assertIn("56m 00s", document)
        self.assertIn("EXACT NEXT CLOSED CANDLE", document)


if __name__ == "__main__":
    unittest.main()
