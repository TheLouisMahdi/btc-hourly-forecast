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

    def test_workflow_directory_has_locked_automatic_triggers(self) -> None:
        workflow_dir = self.root / ".github" / "workflows"
        expected = {"quality.yml", "forecast.yml", "dashboard.yml", "retrain.yml"}
        actual = {path.name for path in workflow_dir.glob("*.yml")}
        self.assertEqual(actual, expected)

        for path in workflow_dir.glob("*.yml"):
            workflow = path.read_text(encoding="utf-8")
            self.assertIn("workflow_dispatch:", workflow, path.name)
            self.assertNotIn("\n  push:", workflow, path.name)
            self.assertNotIn("\n  pull_request:", workflow, path.name)

            if path.name == "forecast.yml":
                self.assertIn("\n  schedule:\n", workflow)
                self.assertIn('cron: "12 * * * *"', workflow)
                self.assertNotIn("\n  workflow_run:\n", workflow)
            elif path.name == "dashboard.yml":
                self.assertNotIn("\n  schedule:\n", workflow)
                self.assertIn("\n  workflow_run:\n", workflow)
                self.assertIn("- Hourly BTC forecast", workflow)
                self.assertIn("- completed", workflow)
                self.assertIn(
                    "github.event.workflow_run.conclusion == 'success'",
                    workflow,
                )
            else:
                self.assertNotIn("\n  schedule:\n", workflow, path.name)
                self.assertNotIn("\n  workflow_run:\n", workflow, path.name)

    def test_forecast_workflow_uses_canonical_entry_point(self) -> None:
        workflow = (
            self.root / ".github" / "workflows" / "forecast.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("python scripts/github_hourly_forecast.py", workflow)
        self.assertNotIn("python scripts/github_structural_forecast.py", workflow)
        self.assertIn("gh workflow run retrain.yml", workflow)
        self.assertIn("inputs.allow_retrain", workflow)
        self.assertIn("github.event_name == 'workflow_dispatch'", workflow)

    def test_dashboard_workflow_has_one_render_entry_point(self) -> None:
        workflow = (
            self.root / ".github" / "workflows" / "dashboard.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "python scripts/render_dashboard.py --runtime-dir",
            workflow,
        )
        self.assertIn("workflow_run:", workflow)
        self.assertIn("Hourly BTC forecast", workflow)
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
