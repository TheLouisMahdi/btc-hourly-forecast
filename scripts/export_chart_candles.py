from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a compact recent closed-candle snapshot for GitHub Pages"
    )
    parser.add_argument("--runtime-dir", default=".github_runtime/hourly")
    parser.add_argument("--state-dir", default=".github_state")
    parser.add_argument("--hours", type=int, default=168)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    database = (root / args.runtime_dir / "btc_hourly.sqlite3").resolve()
    destination = (root / args.state_dir / "chart_candles.json").resolve()
    if not database.exists():
        print("::warning::Recent candle database is unavailable; preserving prior chart snapshot.")
        return 0

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT open_time, open, high, low, close
            FROM candles
            WHERE closed = 1
            ORDER BY open_time DESC
            LIMIT ?
            """,
            (max(int(args.hours), 24),),
        ).fetchall()

    payload = [dict(row) for row in reversed(rows)]
    if len(payload) < 24:
        print("::warning::Too few recent candles for a chart snapshot; preserving prior snapshot.")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Exported {len(payload)} closed candles for the dashboard chart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
