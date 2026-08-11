from __future__ import annotations

import sys
import tempfile
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
            patch.object(
                render_dashboard.github_assistant_dashboard,
                "main",
                side_effect=lambda: calls.append("assistant") or 0,
            ),
            patch.object(
                render_dashboard.github_chart_dashboard,
                "main",
                side_effect=lambda: calls.append("chart") or 0,
            ),
            patch.object(
                render_dashboard.github_market_price_dashboard,
                "main",
                side_effect=lambda: calls.append("market-price") or 0,
            ),
            patch.object(
                render_dashboard,
                "_ensure_resilience_panel",
                side_effect=lambda: calls.append("contract"),
            ),
        ):
            self.assertEqual(render_dashboard.main(), 0)

        self.assertEqual(
            calls,
            [
                "base",
                "visual",
                "uncertainty",
                "resilience",
                "assistant",
                "chart",
                "market-price",
                "contract",
            ],
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
            patch.object(
                render_dashboard.github_assistant_dashboard,
                "main",
                side_effect=lambda: calls.append("assistant") or 0,
            ),
            patch.object(
                render_dashboard.github_chart_dashboard,
                "main",
                side_effect=lambda: calls.append("chart") or 0,
            ),
            patch.object(
                render_dashboard.github_market_price_dashboard,
                "main",
                side_effect=lambda: calls.append("market-price") or 0,
            ),
            patch.object(
                render_dashboard,
                "_ensure_resilience_panel",
                side_effect=lambda: calls.append("contract"),
            ),
        ):
            self.assertEqual(render_dashboard.main(), 7)

        self.assertEqual(calls, ["base", "visual"])

    def test_resilience_panel_uses_position_ledger_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            site = Path(temporary) / "site"
            site.mkdir()
            index = site / "index.html"
            index.write_text(
                '<html><body><main><section class="panel ledger position-ledger">'
                "ledger"
                "</section></main></body></html>",
                encoding="utf-8",
            )
            (site / "latest.json").write_text("{}", encoding="utf-8")
            (site / "history.json").write_text("[]", encoding="utf-8")

            render_dashboard._ensure_resilience_panel(index)
            document = index.read_text(encoding="utf-8")

        self.assertIn(render_dashboard.RESILIENCE_HEADING, document)
        self.assertLess(
            document.index(render_dashboard.RESILIENCE_HEADING),
            document.index('class="panel ledger position-ledger"'),
        )


if __name__ == "__main__":
    unittest.main()
