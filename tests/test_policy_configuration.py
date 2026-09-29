from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


class PolicyConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_canonical_aggressive_policy_configuration(self) -> None:
        config = yaml.safe_load(
            (self.root / "config" / "default.yaml").read_text(
                encoding="utf-8"
            )
        )
        strategy = config["strategy"]

        self.assertEqual(
            strategy["position_policy"],
            "AGGRESSIVE_STRUCTURAL_RISK_SCALED",
        )
        self.assertEqual(strategy["policy_version"], 2)
        self.assertEqual(
            strategy["minimum_risk_per_trade_fraction"],
            0.0075,
        )
        self.assertEqual(
            strategy["maximum_risk_per_trade_fraction"],
            0.020,
        )
        self.assertLess(
            strategy["minimum_risk_per_trade_fraction"],
            strategy["risk_per_trade_fraction"],
        )
        self.assertLess(
            strategy["risk_per_trade_fraction"],
            strategy["maximum_risk_per_trade_fraction"],
        )
        self.assertEqual(
            strategy["maximum_modeled_loss_fraction_per_trade"],
            0.020,
        )
        self.assertEqual(strategy["maximum_leverage"], 40.0)
        self.assertEqual(strategy["leverage_tiers"], [20.0, 30.0, 40.0])
        self.assertEqual(
            config["market"]["execution_quote_provider"],
            "okx_swap",
        )
        self.assertEqual(
            config["market"]["execution_quote_provider_order"],
            ["okx_swap", "coinbase_spot"],
        )
        self.assertEqual(
            strategy["execution_venue"],
            "OKX_BTC_USDT_SWAP",
        )
        self.assertEqual(strategy["margin_mode"], "isolated")
        self.assertTrue(strategy["invert_trade_direction"])
        self.assertEqual(strategy["maintenance_margin_rate"], 0.005)
        self.assertEqual(
            strategy["leverage_risk_score_thresholds"],
            [0.20, 0.50],
        )
        self.assertFalse(
            config["trade_assistant"]["require_qualified_meta_for_position"]
        )
        self.assertEqual(
            config["trade_lifecycle"]["same_bar_policy"],
            "STOP_FIRST",
        )
        self.assertEqual(config["trade_lifecycle"]["breakeven_trigger_r"], 0.75)
        self.assertEqual(config["trade_lifecycle"]["trailing_trigger_r"], 1.25)
        self.assertFalse(config["adaptive"]["enabled"])
        self.assertTrue(config["trade_lifecycle"]["enabled"])
        self.assertEqual(
            config["negative_memory"]["runtime_mode"],
            "ADAPTIVE_PENALTY_ONLY",
        )
        self.assertGreater(
            config["forecast"]["online_maximum_direction_weight"],
            0,
        )
        self.assertNotIn("aggressive_paper_mode", strategy)

    def test_github_runtime_keeps_canonical_strategy_configuration(self) -> None:
        source = (self.root / "scripts" / "github_common.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("without redefining strategy formulas", source)
        self.assertNotIn("aggressive_paper_mode", source)
        self.assertNotIn(
            'values.setdefault("adaptive", {})["enabled"] = True',
            source,
        )
        self.assertNotIn('values.setdefault("strategy"', source)
        self.assertNotIn('values.setdefault("trade_lifecycle"', source)

    def test_negative_memory_uses_its_runtime_mode_not_removed_toggle(self) -> None:
        source = (
            self.root / "src" / "btc_ema_trader" / "negative_memory.py"
        ).read_text(encoding="utf-8")
        self.assertIn('memory_cfg.get("runtime_mode", "HARD_VETO")', source)
        self.assertIn("_apply_boundary_risk_penalty", source)
        self.assertNotIn("aggressive_paper_mode", source)

    def test_release_versions_are_aligned(self) -> None:
        config = yaml.safe_load(
            (self.root / "config" / "default.yaml").read_text(
                encoding="utf-8"
            )
        )
        pyproject = (self.root / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        package = (
            self.root / "src" / "btc_ema_trader" / "__init__.py"
        ).read_text(encoding="utf-8")
        readme = (self.root / "README.md").read_text(encoding="utf-8")

        pyproject_version = re.search(
            r'^version = "([^"]+)"$',
            pyproject,
            flags=re.MULTILINE,
        )
        package_version = re.search(
            r'^__version__ = "([^"]+)"$',
            package,
            flags=re.MULTILINE,
        )

        self.assertIsNotNone(pyproject_version)
        self.assertIsNotNone(package_version)
        assert pyproject_version is not None
        assert package_version is not None
        expected = str(config["project"]["version"])
        self.assertEqual(pyproject_version.group(1), expected)
        self.assertEqual(package_version.group(1), expected)
        self.assertIn(f"version-{expected}", readme)


if __name__ == "__main__":
    unittest.main()
