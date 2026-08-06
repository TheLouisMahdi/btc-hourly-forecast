from __future__ import annotations

import unittest

import pandas as pd

from scripts.github_retrain_policy import evaluate_policy


NOW = pd.Timestamp("2026-08-07T00:00:00Z")


def fresh_latest() -> dict:
    return {
        "run_status": "OK",
        "market_refresh": {"freshness": {"fresh": True}},
        "data_health": {"model_stale": False},
    }


def metadata(created_at: str) -> dict:
    return {"training": {"created_at": created_at}}


class RetrainPolicyTests(unittest.TestCase):
    def test_recent_healthy_model_remains_idle(self) -> None:
        decision = evaluate_policy(
            fresh_latest(),
            {
                "metrics": {
                    "samples": 504,
                    "base_direction_accuracy": 0.53,
                    "online_direction_accuracy": 0.52,
                    "base_direction_brier": 0.249,
                    "online_direction_brier": 0.251,
                }
            },
            metadata("2026-08-04T00:00:00Z"),
            {},
            now=NOW,
        )
        self.assertFalse(decision["required"])
        self.assertEqual(decision["status"], "IDLE")

    def test_old_model_requests_retraining(self) -> None:
        decision = evaluate_policy(
            fresh_latest(),
            {"metrics": {"samples": 100}},
            metadata("2026-06-01T00:00:00Z"),
            {},
            now=NOW,
        )
        self.assertTrue(decision["required"])
        self.assertEqual(decision["reason"], "MODEL_AGE_LIMIT")

    def test_online_outperformance_requests_retraining(self) -> None:
        decision = evaluate_policy(
            fresh_latest(),
            {
                "metrics": {
                    "samples": 600,
                    "base_direction_accuracy": 0.525,
                    "online_direction_accuracy": 0.550,
                    "base_direction_brier": 0.251,
                    "online_direction_brier": 0.246,
                }
            },
            metadata("2026-08-04T00:00:00Z"),
            {},
            now=NOW,
        )
        self.assertTrue(decision["required"])
        self.assertEqual(
            decision["reason"],
            "ONLINE_LEARNER_OUTPERFORMS_BATCH",
        )

    def test_stale_market_blocks_retraining_and_learning_cycle(self) -> None:
        latest = fresh_latest()
        latest["market_refresh"]["freshness"]["fresh"] = False
        decision = evaluate_policy(
            latest,
            {"metrics": {"samples": 1000}},
            metadata("2025-01-01T00:00:00Z"),
            {},
            now=NOW,
        )
        self.assertFalse(decision["required"])
        self.assertEqual(decision["status"], "WAITING_FOR_FRESH_DATA")

    def test_successful_dispatch_has_cooldown(self) -> None:
        decision = evaluate_policy(
            fresh_latest(),
            {"metrics": {"samples": 1000}},
            metadata("2025-01-01T00:00:00Z"),
            {
                "last_dispatch_at": "2026-08-06T00:00:00Z",
                "last_attempt_status": "SUCCESS",
            },
            now=NOW,
        )
        self.assertFalse(decision["required"])
        self.assertEqual(decision["status"], "COOLDOWN")


if __name__ == "__main__":
    unittest.main()
