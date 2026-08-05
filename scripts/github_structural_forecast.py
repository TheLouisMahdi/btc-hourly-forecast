"""Backward-compatible imports for the canonical GitHub runtime.

This module no longer mutates or monkey-patches the hourly entry point. New code
must import from :mod:`btc_ema_trader.github_runtime` or execute
``scripts/github_hourly_forecast.py`` directly.
"""

from __future__ import annotations

from btc_ema_trader.github_runtime import (
    CanonicalAdaptiveTradeEngine,
    CanonicalRuntimeEngine,
    attach_optional_forecast,
    build_optional_secondary_forecast,
    open_trade_with_context,
    optional_price_prediction,
    preserve_canonical_forecast,
    retain_directional_history,
)
from btc_ema_trader.risk_economics import (
    apply_risk_scaled_economics as _apply_risk_scaled_economics,
)

__all__ = [
    "CanonicalAdaptiveTradeEngine",
    "CanonicalRuntimeEngine",
    "_apply_risk_scaled_economics",
    "attach_optional_forecast",
    "build_optional_secondary_forecast",
    "open_trade_with_context",
    "optional_price_prediction",
    "preserve_canonical_forecast",
    "retain_directional_history",
]


def main() -> int:
    """Delegate legacy command execution to the canonical hourly entry point."""
    from github_hourly_forecast import main as canonical_main

    return canonical_main()


if __name__ == "__main__":
    raise SystemExit(main())
