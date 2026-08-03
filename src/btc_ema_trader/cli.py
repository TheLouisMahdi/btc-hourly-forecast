from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_settings
from .contract_training import train_from_database
from .dashboard import launch_dashboard
from .logging_setup import configure_logging
from .market import fetch_and_store
from .model import latest_bundle
from .news import collect_and_store
from .runtime import RuntimeEngine
from .storage import Database

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="btc-regime",
        description=(
            "BTC structural breakout forecaster and fail-safe "
            "paper-trade decision system"
        ),
    )
    root.add_argument("--config", default=None)
    root.add_argument("--verbose", action="store_true")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("init")

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--days", type=float, default=365)
    fetch.add_argument(
        "--provider",
        choices=[
            "auto",
            "binance_futures",
            "bybit_linear",
            "okx_swap",
        ],
        default="auto",
    )

    news = sub.add_parser("news")
    news.add_argument("--historical", action="store_true")
    news.add_argument("--days", type=float, default=None)

    train = sub.add_parser("train")
    train.add_argument(
        "--provider",
        choices=[
            "binance_futures",
            "bybit_linear",
            "okx_swap",
        ],
        default=None,
    )

    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--days", type=float, default=365)
    bootstrap.add_argument(
        "--provider",
        choices=[
            "auto",
            "binance_futures",
            "bybit_linear",
            "okx_swap",
        ],
        default="auto",
    )

    cycle = sub.add_parser("cycle")
    cycle.add_argument(
        "--force",
        action="store_true",
        help="Diagnostic only: bypass the next-candle session gate",
    )

    live = sub.add_parser("live")
    live.add_argument("--once", action="store_true")

    dashboard = sub.add_parser("dashboard")
    dashboard.add_argument("--share", action="store_true")
    dashboard.add_argument("--display-only", action="store_true")

    sub.add_parser("status")
    sub.add_parser("reset-session")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
        configure_logging(settings, verbose=args.verbose)
        database = Database(settings)
        database.initialize()
        result = dispatch(args, settings, database)
        if result is not None:
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        LOGGER.exception("Command failed")
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


def dispatch(args, settings, database):
    if args.command == "init":
        return {
            "status": "initialized",
            "database": str(database.path),
        }
    if args.command == "fetch":
        provider = None if args.provider == "auto" else args.provider
        return fetch_and_store(
            settings,
            database,
            days=args.days,
            provider=provider,
        )
    if args.command == "news":
        return collect_and_store(
            settings,
            database,
            historical=args.historical,
            days=args.days,
        )
    if args.command == "train":
        return train_from_database(
            settings,
            database,
            provider=args.provider,
        )
    if args.command == "bootstrap":
        provider = None if args.provider == "auto" else args.provider
        market = fetch_and_store(
            settings,
            database,
            days=args.days,
            provider=provider,
        )
        news_backfill = _optional(
            lambda: collect_and_store(
                settings,
                database,
                historical=True,
                days=args.days,
            )
        )
        recent_news = _optional(
            lambda: collect_and_store(
                settings,
                database,
                historical=False,
            )
        )
        training = train_from_database(
            settings,
            database,
            provider=market["provider"],
        )
        runtime = RuntimeEngine(
            settings,
            database,
        ).reset_session_clock()
        return {
            "market": market,
            "news_backfill": news_backfill,
            "recent_news": recent_news,
            "training": training,
            "runtime": runtime,
        }
    if args.command == "cycle":
        return RuntimeEngine(
            settings,
            database,
        ).run_once(force=args.force)
    if args.command == "live":
        engine = RuntimeEngine(settings, database)
        if args.once:
            return engine.run_once(force=False)
        engine.reset_session_clock()
        engine.run_forever()
        return None
    if args.command == "dashboard":
        launch_dashboard(
            settings,
            database,
            share=args.share,
            start_engine=not args.display_only,
        )
        return None
    if args.command == "status":
        providers = database.providers(
            settings.section("market").get(
                "symbol",
                "BTCUSDT",
            )
        )
        try:
            bundle = latest_bundle(settings)
            model = {
                "model_id": bundle.model_id,
                "provider": bundle.provider,
                "created_at": bundle.created_at,
                "qualification": bundle.qualification,
            }
        except Exception as exc:
            model = {"error": str(exc)}
        return {
            "providers": providers,
            "model": model,
            "runtime": RuntimeEngine(
                settings,
                database,
            ).status(),
            "signals": database.recent_signals(10).to_dict(
                orient="records"
            ),
        }
    if args.command == "reset-session":
        return RuntimeEngine(
            settings,
            database,
        ).reset_session_clock()
    raise ValueError(args.command)


def _optional(callback):
    try:
        return callback()
    except Exception as exc:
        return {
            "status": "warning",
            "error": f"{type(exc).__name__}: {exc}",
        }
