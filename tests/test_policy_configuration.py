from __future__ import annotations

import re
from pathlib import Path

import yaml


def test_canonical_aggressive_policy_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    strategy = config["strategy"]

    assert strategy["position_policy"] == (
        "AGGRESSIVE_STRUCTURAL_RISK_SCALED"
    )
    assert strategy["policy_version"] == 2
    assert strategy["minimum_risk_per_trade_fraction"] == 0.005
    assert strategy["maximum_risk_per_trade_fraction"] == 0.03
    assert strategy["minimum_risk_per_trade_fraction"] < (
        strategy["risk_per_trade_fraction"]
    )
    assert strategy["risk_per_trade_fraction"] < (
        strategy["maximum_risk_per_trade_fraction"]
    )
    assert config["adaptive"]["enabled"] is False
    assert config["trade_lifecycle"]["enabled"] is True
    assert config["forecast"]["online_maximum_direction_weight"] > 0
    assert "aggressive_paper_mode" not in strategy


def test_github_runtime_keeps_canonical_strategy_configuration() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "github_common.py").read_text(
        encoding="utf-8"
    )

    assert "without redefining strategy formulas" in source
    assert "aggressive_paper_mode" not in source
    assert 'values.setdefault("adaptive", {})["enabled"] = True' not in source


def test_release_versions_are_aligned() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    package = (
        root / "src" / "btc_ema_trader" / "__init__.py"
    ).read_text(encoding="utf-8")

    pyproject_version = re.search(
        r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE
    )
    package_version = re.search(
        r'^__version__ = "([^"]+)"$', package, flags=re.MULTILINE
    )

    assert pyproject_version is not None
    assert package_version is not None
    expected = str(config["project"]["version"])
    assert pyproject_version.group(1) == expected
    assert package_version.group(1) == expected
