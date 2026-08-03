from __future__ import annotations

import re
import unittest
from pathlib import Path

PERSIAN_PATTERN = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
)
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".txt",
    ".sh",
    ".bat",
    ".svg",
}
TEXT_FILENAMES = {".editorconfig"}
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".github_runtime",
    ".github_state",
    ".adaptive_state",
    ".model_state",
    "site",
}
FORBIDDEN_FILES = {
    "GITHUB_FREE_DEPLOY_FA.md",
    "PATCH_INSTRUCTIONS_FA.md",
    "PATCH_V2_1_FA.md",
    "v1_to_v2_regime.patch",
    "requirements.txt",
    "setup.bat",
    "start_first_run.bat",
    "start_live.bat",
    "start_news_refresh.bat",
    "start_retrain.bat",
    "start_status.bat",
}
REQUIRED_FILES = {
    ".editorconfig",
    ".github/workflows/quality.yml",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "README.md",
    "SECURITY.md",
    "docs/assets/candlestick-loop.svg",
    "pyproject.toml",
}


class RepositoryQualityTests(unittest.TestCase):
    def test_repository_contains_no_persian_text(self) -> None:
        root = Path(__file__).resolve().parents[1]
        violations: list[str] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if (
                path.suffix.lower() not in TEXT_SUFFIXES
                and path.name not in TEXT_FILENAMES
            ):
                continue
            if any(
                part in IGNORED_DIRECTORIES for part in path.parts
            ):
                continue
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
            if PERSIAN_PATTERN.search(text):
                violations.append(str(path.relative_to(root)))
        self.assertEqual(
            violations,
            [],
            f"Persian text found in: {violations}",
        )

    def test_obsolete_artifacts_are_absent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        present = sorted(
            name
            for name in FORBIDDEN_FILES
            if (root / name).exists()
        )
        self.assertEqual(
            present,
            [],
            f"Obsolete files found: {present}",
        )

    def test_professional_repository_files_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        missing = sorted(
            name
            for name in REQUIRED_FILES
            if not (root / name).is_file()
        )
        self.assertEqual(
            missing,
            [],
            f"Required repository files are missing: {missing}",
        )

    def test_readme_uses_repository_owned_animation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "docs/assets/candlestick-loop.svg",
            readme,
        )
        self.assertNotIn(
            "capsule-render.vercel.app",
            readme,
        )


if __name__ == "__main__":
    unittest.main()
