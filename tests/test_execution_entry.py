from __future__ import annotations

import unittest

from btc_ema_trader.execution_entry import (
    BATCH_LABEL_ENTRY_CONTRACT,
    EXECUTION_ENTRY_CONTRACT,
    apply_execution_quote,
)


class ExecutionEntryTests(unittest.TestCase):
    def test_fresh_quote_rebases_trade_entry_but_not_source_close(self) -> None:
        record = {
            "price": 100.0,
            "trade_plan": {
                "entry_reference": 100.0,
                "entry_reference_kind": "CURRENT_CLOSE_PROXY",
                "label_execution_aligned": True,
            },
        }
        result = apply_execution_quote(
            record,
            provider="coinbase_spot",
            price=101.25,
            quote_time="2026-01-01T01:00:10Z",
            observed_at="2026-01-01T01:00:20Z",
            maximum_age_seconds=90,
        )
        plan = result["trade_plan"]
        self.assertEqual(result["price"], 100.0)
        self.assertEqual(plan["source_candle_close"], 100.0)
        self.assertEqual(plan["entry_reference"], 101.25)
        self.assertEqual(plan["entry_reference_kind"], EXECUTION_ENTRY_CONTRACT)
        self.assertFalse(plan["label_execution_aligned"])
        self.assertEqual(
            plan["label_entry_definition"],
            BATCH_LABEL_ENTRY_CONTRACT,
        )
        self.assertEqual(
            plan["runtime_entry_definition"],
            EXECUTION_ENTRY_CONTRACT,
        )
        self.assertEqual(
            plan["execution_alignment_status"],
            "APPROXIMATE_UNTIL_MINUTE_LEVEL_RETRAIN",
        )
        self.assertTrue(result["execution_quote"]["fresh"])
        self.assertEqual(result["execution_quote"]["age_seconds"], 10.0)

    def test_stale_quote_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "stale"):
            apply_execution_quote(
                {"price": 100.0, "trade_plan": {}},
                provider="coinbase_spot",
                price=101.0,
                quote_time="2026-01-01T01:00:00Z",
                observed_at="2026-01-01T01:02:00Z",
                maximum_age_seconds=90,
            )


if __name__ == "__main__":
    unittest.main()
