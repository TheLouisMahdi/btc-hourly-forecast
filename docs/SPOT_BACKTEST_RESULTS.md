# Initial Reconstruction Results: Coinbase BTC-USD Spot

**Status:** Verified as a reproducible research run on 2026-10-12. **No changes to the live paper strategy.**

[Successful GitHub Actions run](https://github.com/TheLouisMahdi/btc-hourly-forecast/actions/runs/38213050327) | [Implementation](../scripts/spot_backtest.py) | [Methodology](SPOT_BACKTEST_RECONSTRUCTION.md)

## Data lineage

- Source: Coinbase Exchange public hourly BTC-USD spot candles; January 1, 2024 to December 31, 2025 UTC.
- Expected hours: 17,544; valid returned hours: **17,539**.
- Confirmed gap: **2025-10-25 16:00 to 21:00 UTC (five missing hourly bars)**.
- Dataset canonical JSON SHA256: `5d6c3d1e51b7d710a367ea13af55c0081fee0d33f6c9715e2d2ce73b4c99b9e3`.
- Directly re-run Coinbase fetch with narrow missing-hour retries; a second independent successful Actions run returned the same input hash and primary metrics.
- 1,200 past-only candles for warmup. All replays use the unmodified `rule_based.step` rules. Entry/exit quote is proxied by the **next hour's open**.
- 3 bps fixed slippage on each side; fee scenarios are 15, 40, 60 bps **per side**. These are hypothetical fee tiers, not authenticated account statements.

## Net strategy performance

| Sample | Trades closed | Win % at 15bps | Net % at 15bps | PF at 15bps | Net % at 40bps | Net % at 60bps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 (Feb 19-Dec 31) | 67 | 29.85 | -5.581 | 0.727 | -13.254 | -18.948 |
| 2025 holdout before missing bars (Jan 1-Oct 25 15:00 UTC) | 69 | 26.09 | -11.317 | 0.438 | -20.316 | -26.863 |
| 2025 holdout after recovery and warmup (Dec 14-Dec 31) | 3 | 0 | -1.667 | 0 | -2.190 | -2.608 |
| Descriptive 2024-Feb 2025-Oct continuous interval | 136 | 27.94 | -16.267 | 0.588 | -30.877 | -40.721 |

**Do not add returns from separate segments:** each begins with a fresh $10,000 notional and the missing interval cannot be traded as though it were observed. The 2025 holdout is therefore represented by two separate evaluation segments, not one uninterrupted portfolio.

At the baseline 15-bps fee assumption, the **mean net closed-trade P&L** was **-$8.33 (2024)**, **-$16.40 (2025 before gap)** and **-$55.57 (short late-December segment)**. The corresponding buy-and-hold returns after assumed entry/exit costs were +79.618%, +17.895%, and -1.570%.

## Interpretation and next acceptance gate

- Negative net expectancy persisted in both major independent calendar samples.
- Higher cost tiers considerably worsened results. Coinbase Exchange publishes tier-specific maker/taker rates; do not assume the 15-bps low-cost scenario applies to a small account.
- **This is not a full execution-fidelity backtest:** hourly bars omit 15-minute quote checks, stop sequence, spread, limit-order book and actual order fills. Missed GitHub runs can delay paper execution.
- Before promoting a challenger, first make historical and live execution contracts comparable, measure actual spread and fees, then run pre-registered walk-forward experiments. Do not optimize and rank dozens of variants on the 2025 holdout.

No code or financial decisions are changed automatically by this report. Production stays paper-only.
