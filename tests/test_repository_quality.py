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
}
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


class RepositoryQualityTests(unittest.TestCase):
    def test_repository_contains_no_persian_text(self) -> None:
        root = Path(__file__).resolve().parents[1]
        violations: list[str] = []
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix.lower() not in TEXT_SUFFIXES
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


if __name__ == "__main__":
    unittest.main()
