from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from github_common import json_safe


class GithubCommonTests(unittest.TestCase):
    def test_legacy_runtime_policy_name_is_normalized_recursively(self) -> None:
        result = json_safe(
            {
                "paper_trade_mode": "AGGRESSIVE_ADAPTIVE_5R",
                "nested": ["AGGRESSIVE_ADAPTIVE_5R"],
            }
        )
        self.assertEqual(
            result["paper_trade_mode"],
            "AGGRESSIVE_STRUCTURAL_RISK_SCALED",
        )
        self.assertEqual(
            result["nested"],
            ["AGGRESSIVE_STRUCTURAL_RISK_SCALED"],
        )


if __name__ == "__main__":
    unittest.main()
