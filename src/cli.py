"""
Command-line interface.

    python -m src.cli run      [--mode live|record|replay] [--reset] [--emails PATH] [--out DIR]
    python -m src.cli eval     [--mode ...] [--min-intent-accuracy 0.9] [--min-order-recall 0.9] [--output FILE]
    python -m src.cli migrate  apply database migrations
    python -m src.cli seed     [--reset-stock]  load or refresh the catalog (embeds changed products)
    python -m src.cli serve    [--host 0.0.0.0] [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from src.config import LLMMode, Settings, get_settings
from src.utils.logger import configure_logging, get_logger

logger = get_logger("cli")


def _with_mode(settings: Settings, mode: Optional[str]) -> Settings:
    return settings.model_copy(update={"LLM_MODE": LLMMode(mode)}) if mode else settings


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    from src.engine import build_engine
    from src.loaders import load_emails
    from src.utils.reporter import build_report, export_report, summarize

    settings = _with_mode(settings, args.mode)
    if args.emails:
        settings = settings.model_copy(update={"INBOUND_EMAILS_CSV_PATH": args.emails})
    engine = build_engine(settings, reset=args.reset)
    try:
        outcomes = engine.process(load_emails(settings.INBOUND_EMAILS_CSV_PATH))
    finally:
        engine.close()
    export_report(build_report(outcomes), args.out or settings.EXPORT_DIR)
    summary = summarize(outcomes, settings.LLM_PRICE_INPUT_PER_1M, settings.LLM_PRICE_OUTPUT_PER_1M)
    summary.update({"llm_mode": settings.LLM_MODE.value, "model": engine.llm.model})
    print(json.dumps(summary, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    from src.evaluation import run_evaluation

    result = run_evaluation(_with_mode(settings, args.mode), args.gold)
    report = result.to_markdown()
    print(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report)
    for metric, threshold in (
        ("intent accuracy", args.min_intent_accuracy),
        ("order line recall", args.min_order_recall),
    ):
        if result.metrics[metric] < threshold:
            logger.error(
                "%s %.1f%% is below the %.1f%% threshold", metric, result.metrics[metric] * 100, threshold * 100
            )
            return 1
    return 0


def cmd_migrate(args: argparse.Namespace, settings: Settings) -> int:
    from src.storage.db import Database
    from src.storage.migrate import apply_migrations

    db = Database(settings)
    try:
        print(json.dumps({"applied": apply_migrations(db)}))
    finally:
        db.close()
    return 0


def cmd_seed(args: argparse.Namespace, settings: Settings) -> int:
    from src.catalog import sync_catalog
    from src.llm import build_llm_client
    from src.loaders import load_products
    from src.storage import open_store

    store = open_store(settings)
    try:
        sync_catalog(store, load_products(settings.CATALOG_CSV_PATH), build_llm_client(settings), args.reset_stock)
    finally:
        store.close()
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    uvicorn.run("src.api.app:create_app", factory=True, host=args.host, port=args.port, log_config=None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="commerceops", description="CommerceOps AI Engine")
    sub = parser.add_subparsers(dest="command", required=True)
    modes = [m.value for m in LLMMode]

    run = sub.add_parser("run", help="process the email queue and export the operations report")
    run.add_argument("--mode", choices=modes, help="override LLM_MODE")
    run.add_argument("--reset", action="store_true", help="clear processed emails and restore catalog stock first")
    run.add_argument("--emails", help="CSV with email_id, subject, message")
    run.add_argument("--out", help="export directory (default: EXPORT_DIR)")
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("eval", help="score the agent against evals/gold_labels.json (resets the database)")
    ev.add_argument("--mode", choices=modes)
    ev.add_argument("--gold", default="evals/gold_labels.json")
    ev.add_argument("--output", help="also write the markdown report to this file")
    ev.add_argument("--min-intent-accuracy", type=float, default=0.0)
    ev.add_argument("--min-order-recall", type=float, default=0.0)
    ev.set_defaults(func=cmd_eval)

    sub.add_parser("migrate", help="apply database migrations").set_defaults(func=cmd_migrate)

    seed = sub.add_parser("seed", help="load or refresh the catalog")
    seed.add_argument("--reset-stock", action="store_true", help="overwrite stock levels from the catalog file")
    seed.set_defaults(func=cmd_seed)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
    try:
        return int(args.func(args, settings))
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
