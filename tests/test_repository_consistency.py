from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

import yaml

import btc_ema_trader


class RepositoryConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_all_release_versions_match(self) -> None:
        project = tomllib.loads(
            (self.root / "pyproject.toml").read_text(encoding="utf-8")
        )
        config = yaml.safe_load(
            (self.root / "config" / "default.yaml").read_text(
                encoding="utf-8"
            )
        )
        expected = str(project["project"]["version"])
        versions = {
            expected,
            str(config["project"]["version"]),
            btc_ema_trader.__version__,
        }
        self.assertEqual(versions, {expected})
        self.assertIn(
            f"version-{expected}",
            (self.root / "README.md").read_text(encoding="utf-8"),
        )

    def test_obsolete_local_ui_and_aliases_are_absent(self) -> None:
        self.assertFalse(
            (self.root / "src" / "btc_ema_trader" / "dashboard.py").exists()
        )
        self.assertFalse((self.root / "tests" / "conftest.py").exists())
        project = tomllib.loads(
            (self.root / "pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = "\n".join(project["project"]["dependencies"]).lower()
        scripts = project["project"]["scripts"]
        self.assertNotIn("gradio", dependencies)
        self.assertNotIn("plotly", dependencies)
        self.assertNotIn("pytest", dependencies)
        self.assertEqual(set(scripts), {"btc-regime"})

    def test_generated_artifacts_are_not_committed_to_main(self) -> None:
        forbidden = (
            "artifacts/models/latest.joblib",
            "artifacts/reports/latest_training_report.json",
            "artifacts/reports/latest_metrics.csv",
        )
        present = [name for name in forbidden if (self.root / name).exists()]
        self.assertEqual(present, [])
        ignore = (self.root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("artifacts/models/*", ignore)
        self.assertIn("artifacts/reports/*", ignore)

    def test_canonical_runtime_contract_modules_exist(self) -> None:
        required = (
            "src/btc_ema_trader/active_position_contract.py",
            "src/btc_ema_trader/candle_context.py",
            "src/btc_ema_trader/execution_entry.py",
            "src/btc_ema_trader/execution_path.py",
            "src/btc_ema_trader/risk_economics.py",
            "src/btc_ema_trader/strict_forecast_contract.py",
        )
        missing = [name for name in required if not (self.root / name).is_file()]
        self.assertEqual(missing, [])

    def test_workflow_automation_contract(self) -> None:
        workflow_dir = self.root / ".github" / "workflows"
        expected = {
            "quality.yml",
            "forecast.yml",
            "dashboard.yml",
            "retrain.yml",
            "manual-trigger-bridge.yml",
        }
        actual = {path.name for path in workflow_dir.glob("*.yml")}
        self.assertEqual(actual, expected)

        workflows = {
            path.name: path.read_text(encoding="utf-8")
            for path in workflow_dir.glob("*.yml")
        }
        for name, workflow in workflows.items():
            self.assertIn("workflow_dispatch:", workflow, name)

        quality = workflows["quality.yml"]
        self.assertIn("\n  push:\n", quality)
        self.assertIn("\n      - main\n", quality)
        self.assertIn("\n  schedule:\n", quality)
        self.assertIn('cron: "37 3 * * *"', quality)
        self.assertNotIn("\n  workflow_run:\n", quality)

        forecast = workflows["forecast.yml"]
        self.assertIn("\n  schedule:\n", forecast)
        self.assertIn('cron: "12 * * * *"', forecast)
        self.assertNotIn("\n  push:\n", forecast)
        self.assertNotIn("\n  workflow_run:\n", forecast)
        self.assertNotIn("gh workflow run retrain.yml", forecast)
        self.assertNotIn("inputs.allow_retrain", forecast)

        dashboard = workflows["dashboard.yml"]
        self.assertIn("\n  workflow_run:\n", dashboard)
        self.assertIn("- Hourly BTC forecast", dashboard)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", dashboard)
        self.assertNotIn("\n  schedule:\n", dashboard)
        self.assertNotIn("\n  push:\n", dashboard)

        retrain = workflows["retrain.yml"]
        self.assertNotIn("\n  schedule:\n", retrain)
        self.assertNotIn("\n  push:\n", retrain)
        self.assertNotIn("\n  workflow_run:\n", retrain)

    def test_forecast_workflow_uses_canonical_entry_point(self) -> None:
        workflow = (
            self.root / ".github" / "workflows" / "forecast.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("python scripts/github_hourly_forecast.py", workflow)
        self.assertNotIn("python scripts/github_structural_forecast.py", workflow)
        self.assertNotIn("gh workflow run retrain.yml", workflow)
        self.assertNotIn("inputs.allow_retrain", workflow)
        self.assertIn('cron: "12 * * * *"', workflow)

    def test_dashboard_workflow_has_one_render_entry_point(self) -> None:
        workflow = (
            self.root / ".github" / "workflows" / "dashboard.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "python scripts/render_dashboard.py --runtime-dir",
            workflow,
        )
        for component in (
            "github_pages_dashboard.py",
            "github_visual_dashboard.py",
            "github_uncertainty_dashboard.py",
            "github_resilience_dashboard.py",
        ):
            self.assertNotIn(f"python scripts/{component}", workflow)

    def test_github_runtime_does_not_redefine_strategy_parameters(self) -> None:
        common = (self.root / "scripts" / "github_common.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('values.setdefault("strategy"', common)
        self.assertNotIn('values.setdefault("trade_lifecycle"', common)
        self.assertNotIn('values.setdefault("negative_memory"', common)

    def test_removed_local_dashboard_settings_are_absent(self) -> None:
        config = yaml.safe_load(
            (self.root / "config" / "default.yaml").read_text(
                encoding="utf-8"
            )
        )
        live = config["live"]
        self.assertNotIn("dashboard_host", live)
        self.assertNotIn("dashboard_port", live)
        self.assertNotIn("quote_poll_seconds", live)


if __name__ == "__main__":
    unittest.main()
