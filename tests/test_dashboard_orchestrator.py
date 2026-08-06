from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import render_dashboard


class DashboardOrchestratorTests(unittest.TestCase):
    def test_components_run_once_in_locked_order(self) -> None:
        calls: list[str] = []

        with (
            patch.object(
                render_dashboard.github_pages_dashboard,
                "main",
                side_effect=lambda: calls.append("base") or 0,
            ),
            patch.object(
                render_dashboard.github_visual_dashboard,
                "main",
                side_effect=lambda: calls.append("visual") or 0,
            ),
            patch.object(
                render_dashboard.github_uncertainty_dashboard,
                "main",
                side_effect=lambda: calls.append("uncertainty") or 0,
            ),
            patch.object(
                render_dashboard.github_resilience_dashboard,
                "main",
                side_effect=lambda: calls.append("resilience") or 0,
            ),
        ):
            self.assertEqual(render_dashboard.main(), 0)

        self.assertEqual(
            calls,
            ["base", "visual", "uncertainty", "resilience"],
        )

    def test_failure_stops_later_components(self) -> None:
        calls: list[str] = []

        with (
            patch.object(
                render_dashboard.github_pages_dashboard,
                "main",
                side_effect=lambda: calls.append("base") or 0,
            ),
            patch.object(
                render_dashboard.github_visual_dashboard,
                "main",
                side_effect=lambda: calls.append("visual") or 7,
            ),
            patch.object(
                render_dashboard.github_uncertainty_dashboard,
                "main",
                side_effect=lambda: calls.append("uncertainty") or 0,
            ),
            patch.object(
                render_dashboard.github_resilience_dashboard,
                "main",
                side_effect=lambda: calls.append("resilience") or 0,
            ),
        ):
            self.assertEqual(render_dashboard.main(), 7)

        self.assertEqual(calls, ["base", "visual"])


if __name__ == "__main__":
    unittest.main()
