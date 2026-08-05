from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.negative_memory import (
    BloomFilter,
    RESISTANCE,
    SUPPORT,
    _apply_boundary_risk_penalty,
    boundary_context,
    fingerprint,
)


class _Settings:
    def section(self, name: str):
        if name != "strategy":
            raise KeyError(name)
        return {
            "account_equity_usd": 1000.0,
            "risk_per_trade_fraction": 0.0125,
            "minimum_risk_per_trade_fraction": 0.005,
            "maximum_risk_per_trade_fraction": 0.03,
        }


class NegativeMemoryTests(unittest.TestCase):
    def test_bloom_filter_never_misses_inserted_keys(self) -> None:
        bloom = BloomFilter.create(250, false_positive_rate=0.005)
        keys = [f"bad-pattern-{index}" for index in range(250)]
        for key in keys:
            bloom.add(key)
        self.assertTrue(all(bloom.contains(key) for key in keys))

    def test_fingerprint_is_stable_and_side_specific(self) -> None:
        row = pd.Series(
            {
                "boundary_distance_atr": 0.24,
                "boundary_strength": 4.7,
                "boundary_age_bars": 18,
                "return_1": -0.002,
                "return_3": -0.006,
                "return_6": -0.012,
                "volume_z_24": 1.4,
                "volume_z_72": 0.8,
                "atr_percentile_168": 0.72,
                "rsi_centered": -0.35,
                "adx": 27,
                "regime_code": -1,
                "triangle_code": 0,
            }
        )
        support_a = fingerprint(row, SUPPORT, 6)
        support_b = fingerprint(row.copy(), SUPPORT, 6)
        resistance = fingerprint(row, RESISTANCE, 6)
        self.assertEqual(support_a, support_b)
        self.assertNotEqual(support_a, resistance)

    def test_boundary_context_follows_approach_direction(self) -> None:
        support_row = pd.Series(
            {
                "return_6": -0.02,
                "structure_support": 60_000,
                "distance_to_support_atr": 0.30,
                "support_strength": 5.0,
                "support_age_bars": 12,
                "structure_resistance": 62_000,
                "distance_to_resistance_atr": 2.0,
            }
        )
        resistance_row = support_row.copy()
        resistance_row["return_6"] = 0.02
        resistance_row["distance_to_support_atr"] = 2.0
        resistance_row["distance_to_resistance_atr"] = 0.25
        self.assertEqual(
            boundary_context(support_row)["boundary_side"],
            SUPPORT,
        )
        self.assertEqual(
            boundary_context(resistance_row)["boundary_side"],
            RESISTANCE,
        )

    def test_bloom_and_bad_pattern_evidence_reduce_but_do_not_zero_risk(self) -> None:
        plan = {
            "risk_fraction": 0.02,
            "risk_budget_usd": 20.0,
            "risk_assessment": {},
            "soft_risk_flags": [],
        }
        flags = [
            "KNOWN_BAD_PATTERN_FRONT_BLOOM",
            "HIGH_UNPROFITABLE_PATTERN_RISK",
        ]
        result = _apply_boundary_risk_penalty(plan, flags, _Settings())
        self.assertLess(result["risk_fraction"], 0.02)
        self.assertGreaterEqual(result["risk_fraction"], 0.005)
        self.assertEqual(
            result["risk_budget_usd"],
            1000.0 * result["risk_fraction"],
        )
        self.assertEqual(result["negative_memory_risk_multiplier"], 0.60)
        self.assertIn(
            "KNOWN_BAD_PATTERN_FRONT_BLOOM",
            result["soft_risk_flags"],
        )
        self.assertEqual(
            result["risk_assessment"]["pre_memory_risk_fraction"],
            0.02,
        )

    def test_no_memory_flags_leave_risk_unchanged(self) -> None:
        result = _apply_boundary_risk_penalty(
            {"risk_fraction": 0.0125, "risk_assessment": {}},
            [],
            _Settings(),
        )
        self.assertEqual(result["risk_fraction"], 0.0125)
        self.assertEqual(result["negative_memory_risk_multiplier"], 1.0)


if __name__ == "__main__":
    unittest.main()
