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
    def test_background_is_subtle_and_has_no_market_clutter(self) -> None:
        background = github_visual_dashboard._background()
        self.assertIn('data-luxury-theme="v1"', background)
        self.assertIn("luxury-backdrop", background)
        self.assertNotIn("ambient-coin", background)
        self.assertNotIn("ambient-heart", background)

    def test_styles_use_formal_cream_brown_gold_palette(self) -> None:
        styles = github_visual_dashboard._styles()
        for token in ("#f2eadf", "#352820", "#b89243", "#6f4d25"):
            self.assertIn(token, styles)
        self.assertNotIn('data-theme="dark"', styles)
        self.assertNotIn("@keyframes coin-drift", styles)
        self.assertNotIn("text-shadow", styles)
        self.assertNotIn("color-mix", styles)

    def test_visual_layer_has_no_theme_toggle_or_persistent_theme_script(self) -> None:
        source = Path(github_visual_dashboard.__file__).read_text(encoding="utf-8")
        self.assertNotIn("theme-toggle", source)
        self.assertNotIn("localStorage", source)
        self.assertNotIn("btc-dashboard-theme", source)


if __name__ == "__main__":
    unittest.main()
