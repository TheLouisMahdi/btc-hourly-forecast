from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.negative_memory import (
    BloomFilter,
    RESISTANCE,
    SUPPORT,
    boundary_context,
    fingerprint,
)


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


if __name__ == "__main__":
    unittest.main()
