#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || scripts/setup.sh
. .venv/bin/activate
python -m btc_ema_trader bootstrap --days 180 --provider auto
python -m btc_ema_trader dashboard
