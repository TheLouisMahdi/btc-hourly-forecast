from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_settings
from .dashboard import launch_dashboard
from .logging_setup import configure_logging
from .market import fetch_and_store
from .model import latest_bundle
from .news import collect_and_store
from .runtime import RuntimeEngine
from .storage import Database
from .training import train_from_database

LOGGER = logging.getLogger(__name__)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="btc-ema", description="BTC 1-hour regime-event forecaster and fail-safe paper-trade decision system")
    p.add_argument("--config", default=None)
    p.add_argument("--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--days", type=float, default=180)
    fetch.add_argument("--provider", choices=["auto","binance_futures","bybit_linear","okx_swap"], default="auto")
    news = sub.add_parser("news")
    news.add_argument("--historical", action="store_true")
    news.add_argument("--days", type=float, default=None)
    train = sub.add_parser("train")
    train.add_argument("--provider", choices=["binance_futures","bybit_linear","okx_swap"], default=None)
    bootstrap = sub.add_parser("bootstrap")
    bootstrap.add_argument("--days", type=float, default=180)
    bootstrap.add_argument("--provider", choices=["auto","binance_futures","bybit_linear","okx_swap"], default="auto")
    cycle = sub.add_parser("cycle")
    cycle.add_argument("--force", action="store_true", help="Diagnostic only: bypass next-candle session gate")
    live = sub.add_parser("live")
    live.add_argument("--once", action="store_true")
    dash = sub.add_parser("dashboard")
    dash.add_argument("--share", action="store_true")
    dash.add_argument("--display-only", action="store_true")
    sub.add_parser("status")
    sub.add_parser("reset-session")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
        configure_logging(settings, verbose=args.verbose)
        db = Database(settings)
        db.initialize()
        result = dispatch(args, settings, db)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Command failed")
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def dispatch(args, settings, db):
    if args.command == "init":
        return {"status":"initialized","database":str(db.path)}
    if args.command == "fetch":
        return fetch_and_store(settings, db, days=args.days, provider=None if args.provider=="auto" else args.provider)
    if args.command == "news":
        return collect_and_store(settings, db, historical=args.historical, days=args.days)
    if args.command == "train":
        return train_from_database(settings, db, provider=args.provider)
    if args.command == "bootstrap":
        market = fetch_and_store(settings, db, days=args.days, provider=None if args.provider=="auto" else args.provider)
        news_backfill = _optional(lambda: collect_and_store(settings, db, historical=True, days=args.days))
        recent_news = _optional(lambda: collect_and_store(settings, db, historical=False))
        training = train_from_database(settings, db, provider=market["provider"])
        runtime = RuntimeEngine(settings, db).reset_session_clock()
        return {"market":market,"news_backfill":news_backfill,"recent_news":recent_news,"training":training,"runtime":runtime}
    if args.command == "cycle":
        return RuntimeEngine(settings, db).run_once(force=args.force)
    if args.command == "live":
        engine = RuntimeEngine(settings, db)
        if args.once:
            return engine.run_once(force=False)
        engine.reset_session_clock()
        engine.run_forever()
        return None
    if args.command == "dashboard":
        launch_dashboard(settings, db, share=args.share, start_engine=not args.display_only)
        return None
    if args.command == "status":
        providers=db.providers(settings.section("market").get("symbol","BTCUSDT"))
        try:
            bundle=latest_bundle(settings)
            model={"model_id":bundle.model_id,"provider":bundle.provider,"created_at":bundle.created_at,"qualification":bundle.qualification}
        except Exception as exc:
            model={"error":str(exc)}
        return {"providers":providers,"model":model,"runtime":RuntimeEngine(settings,db).status(),"signals":db.recent_signals(10).to_dict(orient="records")}
    if args.command == "reset-session":
        return RuntimeEngine(settings,db).reset_session_clock()
    raise ValueError(args.command)


def _optional(callable_):
    try:
        return callable_()
    except Exception as exc:  # noqa: BLE001
        return {"status":"warning","error":f"{type(exc).__name__}: {exc}"}
