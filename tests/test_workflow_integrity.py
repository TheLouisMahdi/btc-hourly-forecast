from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class WorkflowIntegrityTests(unittest.TestCase):
    def test_all_workflows_are_valid_structured_yaml(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow_dir = root / ".github" / "workflows"
        paths = sorted(workflow_dir.glob("*.yml")) + sorted(
            workflow_dir.glob("*.yaml")
        )
        self.assertTrue(paths, "No GitHub Actions workflows were found")
        for path in paths:
            with self.subTest(workflow=path.name):
                payload = yaml.load(
                    path.read_text(encoding="utf-8"),
                    Loader=yaml.BaseLoader,
                )
                self.assertIsInstance(payload, dict)
                self.assertIn("name", payload)
                self.assertIn("on", payload)
                self.assertIn("jobs", payload)
                self.assertIsInstance(payload["jobs"], dict)
                self.assertTrue(payload["jobs"])


if __name__ == "__main__":
    unittest.main()
