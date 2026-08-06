from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import github_visual_dashboard


class VisualDashboardTests(unittest.TestCase):
    def test_background_contains_all_market_assets(self) -> None:
        background = github_visual_dashboard._background()
        self.assertIn('data-market-ambience="v2"', background)
        for coin in ("btc", "eth", "sol", "xrp", "bnb", "ada", "doge"):
            self.assertIn(f"coin-{coin}", background)
        self.assertIn("ambient-heart", background)

    def test_styles_are_visible_compatible_and_accessible(self) -> None:
        styles = github_visual_dashboard._styles()
        self.assertIn("@keyframes coin-drift", styles)
        self.assertIn("@media(prefers-reduced-motion:reduce)", styles)
        self.assertIn("z-index:1", styles)
        self.assertNotIn("color-mix", styles)
        self.assertNotIn("contain:strict", styles)

    def test_theme_control_persists_user_choice(self) -> None:
        script = github_visual_dashboard._script()
        self.assertIn("btc-dashboard-theme", script)
        self.assertIn("localStorage.setItem", script)
        self.assertIn('data-theme', script)


if __name__ == "__main__":
    unittest.main()
