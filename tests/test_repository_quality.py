from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

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
    "scripts/github_pages_dashboard.py",
    "src/btc_ema_trader/contract_training.py",
    "src/btc_ema_trader/forecast_contract.py",
    "src/btc_ema_trader/market_structure.py",
    "src/btc_ema_trader/price_adaptive.py",
    "src/btc_ema_trader/structure_training.py",
    "tests/test_breakout_labels.py",
    "tests/test_dashboard_outcomes.py",
    "tests/test_forecast_contract.py",
    "tests/test_market_structure.py",
    "tests/test_price_adaptive.py",
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
                part in IGNORED_DIRECTORIES
                for part in path.parts
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
        readme = (root / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "docs/assets/candlestick-loop.svg",
            readme,
        )
        self.assertNotIn(
            "capsule-render.vercel.app",
            readme,
        )

    def test_readme_documents_immutable_outcomes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "NEXT_CLOSED_1H_CANDLE",
            "DIRECTION_CORRECT",
            "DIRECTION_WRONG",
            "IN_RANGE",
            "OUT_OF_RANGE",
            "immutable",
        ):
            self.assertIn(required, readme)

    def test_readme_documents_structural_breakouts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "RESISTANCE_BREAKOUT_LONG",
            "SUPPORT_BREAKDOWN_SHORT",
            "TRIANGLE_BREAKOUT_LONG",
            "false_breakout_h*",
            "structure-breakout-hourly-",
        ):
            self.assertIn(required, readme)

    def test_dashboard_contains_project_identity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dashboard = (
            root / "scripts" / "github_dashboard.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Mahdi Ghahremani", dashboard)
        self.assertIn("TheLouisMahdi", dashboard)
        self.assertIn(
            "https://github.com/TheLouisMahdi",
            dashboard,
        )

    def test_model_is_configured_for_structural_breakouts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "config" / "default.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            config["model"]["horizons_hours"],
            [1, 3, 6],
        )
        self.assertEqual(
            config["market"]["history_days"],
            365,
        )
        self.assertEqual(
            config["forecast"]["target"],
            "NEXT_CLOSED_1H_CANDLE",
        )
        self.assertIn(
            "price_adaptive_state",
            config["paths"],
        )
        structure = config["structure"]
        self.assertIn(480, structure["lookback_hours"])
        self.assertGreaterEqual(
            structure["triangle_lookback_hours"],
            120,
        )
        self.assertGreater(
            structure["breakout_crossing_tolerance_atr"],
            0.0,
        )

    def test_schema_v4_model_rejects_legacy_artifacts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        model = (
            root / "src" / "btc_ema_trader" / "model.py"
        ).read_text(encoding="utf-8")
        self.assertIn("schema_version: int = 4", model)
        self.assertIn("structure-breakout-hourly-", model)
        self.assertIn(
            "The model artifact predates structural breakout v4",
            model,
        )

    def test_contract_training_uses_structural_trainer(self) -> None:
        root = Path(__file__).resolve().parents[1]
        contract_training = (
            root
            / "src"
            / "btc_ema_trader"
            / "contract_training.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "from .structure_training import train_feature_set",
            contract_training,
        )


if __name__ == "__main__":
    unittest.main()
