from __future__ import annotations

import unittest

from btc_ema_trader.active_position_contract import (
    ACTIVE_POSITION_CONTRACT,
    build_active_position_plan,
)


class ActivePositionContractTests(unittest.TestCase):
    def test_open_position_becomes_the_only_management_plan(self) -> None:
        trade = {
            "status": "OPEN",
            "direction": "LONG",
            "trade_id": "trade-1",
            "event_id": "event-1",
            "event_type": "RESISTANCE_BREAKOUT_LONG",
            "entry_price": 100.0,
            "target_price": 105.0,
            "initial_stop_price": 99.0,
            "current_stop_price": 100.2,
            "risk_reward": 5.0,
            "maximum_holding_hours": 72,
            "stress_execution_cost_bps": 20.0,
        }
        plan = build_active_position_plan(trade)
        self.assertEqual(plan["contract_type"], ACTIVE_POSITION_CONTRACT)
        self.assertEqual(plan["status"], "MANAGING_OPEN_TRADE")
        self.assertEqual(plan["direction"], "LONG")
        self.assertEqual(plan["entry_reference"], 100.0)
        self.assertEqual(plan["target_price"], 105.0)
        self.assertEqual(plan["stop_price"], 100.2)
        self.assertEqual(plan["event_type"], "RESISTANCE_BREAKOUT_LONG")
        self.assertNotIn("forecast_direction", plan)

    def test_closed_trade_cannot_create_active_plan(self) -> None:
        with self.assertRaisesRegex(ValueError, "OPEN"):
            build_active_position_plan(
                {
                    "status": "CLOSED",
                    "direction": "SHORT",
                    "entry_price": 100.0,
                    "target_price": 95.0,
                    "initial_stop_price": 101.0,
                    "current_stop_price": 101.0,
                }
            )


if __name__ == "__main__":
    unittest.main()
