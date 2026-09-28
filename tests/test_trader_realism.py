from __future__ import annotations

import unittest

import pandas as pd

from btc_ema_trader.costs import projected_funding_bps
from btc_ema_trader.model import build_trade_direction_context
from btc_ema_trader.risk_economics import (
    estimate_isolated_liquidation_price,
    select_leverage,
)
from btc_ema_trader.trade_lifecycle import _close_trade


class TraderRealismTests(unittest.TestCase):
    def _strategy(self) -> dict:
        return {
            "entry_order_style": "market",
            "exit_order_style": "taker",
            "maker_fee_bps": 2.0,
            "taker_fee_bps": 5.0,
            "entry_slippage_bps": 1.5,
            "exit_slippage_bps": 2.5,
            "funding_buffer_bps": 0.0,
            "funding_interval_hours": 8.0,
            "funding_rate_buffer_bps_per_interval": 1.0,
            "stress_cost_multiplier": 1.5,
            "maximum_leverage": 40.0,
            "leverage_tiers": [10.0, 20.0, 40.0],
            "leverage_risk_score_thresholds": [0.35, 0.65],
            "leverage_liquidation_safety_factor": 0.70,
            "maintenance_margin_rate": 0.004,
            "maintenance_amount_usd": 0.0,
            "margin_mode": "ISOLATED",
            "trade_direction_horizon_weights": {
                3: 1.0,
                6: 1.0,
                12: 1.0,
            },
        }

    def test_one_hour_up_can_coexist_with_multi_hour_short(self) -> None:
        context = build_trade_direction_context(
            probabilities={
                1: 0.503,
                3: 0.482,
                6: 0.482,
                12: 0.496,
            },
            general_returns={
                1: 0.0002,
                3: -0.0030,
                6: -0.0040,
                12: -0.0050,
            },
            trade_horizons=[3, 6, 12],
            strategy_cfg=self._strategy(),
        )

        self.assertEqual(context["trade_direction"], "DOWN")
        self.assertEqual(
            context["trade_direction_source"],
            "MULTI_HORIZON_GENERAL_MODEL",
        )
        self.assertGreater(context["trade_returns"][3], 0.0)
        self.assertNotIn(1, context["trade_direction_horizon_weights"])

    def test_current_example_20x_short_liquidation_is_numeric(self) -> None:
        price = estimate_isolated_liquidation_price(
            self._strategy(),
            entry=82_804.0,
            direction="SHORT",
            leverage=20.0,
        )

        self.assertIsNotNone(price)
        assert price is not None
        self.assertAlmostEqual(price, 86_597.80876494, places=6)
        self.assertAlmostEqual(
            (price - 82_804.0) / 82_804.0,
            0.0458167330677,
            places=10,
        )

    def test_40x_is_downgraded_when_stop_risk_is_too_close_to_liquidation(self) -> None:
        leverage = select_leverage(
            self._strategy(),
            risk_score=0.90,
            modeled_risk_fraction=0.015,
            entry=82_804.0,
            direction="SHORT",
        )
        self.assertEqual(leverage, 20.0)

    def test_funding_buffer_scales_with_holding_window(self) -> None:
        strategy = self._strategy()
        self.assertEqual(projected_funding_bps(strategy, 3), 1.0)
        self.assertEqual(projected_funding_bps(strategy, 8), 1.0)
        self.assertEqual(projected_funding_bps(strategy, 12), 2.0)

    def test_realized_pnl_uses_base_cost_plus_holding_funding(self) -> None:
        trade = {
            "entry_price": 100.0,
            "direction": "LONG",
            "opened_at": "2026-01-01T00:00:00Z",
            "notional_usd": 1000.0,
            "risk_budget_usd": 10.0,
            "base_execution_cost_bps": 14.0,
            "stress_execution_cost_bps": 21.0,
            "funding_interval_hours": 8.0,
            "funding_rate_buffer_bps_per_interval": 1.0,
        }
        _close_trade(
            trade,
            exit_price=101.0,
            outcome="TIME_EXIT",
            closed_at=pd.Timestamp("2026-01-01T09:00:00Z"),
        )

        self.assertEqual(trade["estimated_funding_cost_bps"], 2.0)
        self.assertEqual(trade["estimated_realized_cost_bps"], 16.0)
        self.assertAlmostEqual(trade["realized_net_pnl_usd"], 8.4, places=9)


if __name__ == "__main__":
    unittest.main()
