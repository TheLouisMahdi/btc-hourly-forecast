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
        versions = {
            project["project"]["version"],
            str(config["project"]["version"]),
            btc_ema_trader.__version__,
        }
        self.assertEqual(versions, {"5.4.0"})
        self.assertIn(
            "version-5.4.0",
            (self.root / "README.md").read_text(encoding="utf-8"),
        )

    def test_obsolete_local_dashboard_and_dependencies_are_absent(self) -> None:
        self.assertFalse(
            (self.root / "src" / "btc_ema_trader" / "dashboard.py").exists()
        )
        project = tomllib.loads(
            (self.root / "pyproject.toml").read_text(encoding="utf-8")
        )
        dependencies = "\n".join(project["project"]["dependencies"]).lower()
        self.assertNotIn("gradio", dependencies)
        self.assertNotIn("plotly", dependencies)

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

    def test_pages_workflow_has_one_dashboard_entry_point(self) -> None:
        workflow = (
            self.root / ".github" / "workflows" / "pages_dashboard.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "python scripts/github_pages_dashboard.py --runtime-dir",
            workflow,
        )
        self.assertNotIn(
            "python scripts/github_boundary_dashboard.py\n",
            workflow,
        )
        self.assertNotIn(
            "python scripts/github_trade_dashboard.py\n",
            workflow,
        )
        self.assertNotIn(
            "python scripts/github_timing_dashboard.py\n",
            workflow,
        )

    def test_github_runtime_does_not_redefine_strategy_parameters(self) -> None:
        common = (self.root / "scripts" / "github_common.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('values.setdefault("strategy"', common)
        self.assertNotIn('values.setdefault("trade_lifecycle"', common)
        self.assertNotIn('values.setdefault("negative_memory"', common)


if __name__ == "__main__":
    unittest.main()
