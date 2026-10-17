"""
Minimal forward-only migration runner.

Migrations are ``NNN_name.sql`` files applied in order within one transaction
(all or nothing), under an advisory lock so concurrent app instances can start safely.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.storage.db import Database
from src.utils.logger import get_logger

logger = get_logger("storage.migrate")

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_LOCK_ID = 7_210_431


def migration_files() -> List[Path]:
    return sorted(MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))


def apply_migrations(db: Database) -> List[str]:
    applied_now: List[str] = []
    with db.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_ID,))
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        cur.execute("SELECT version FROM schema_migrations")
        done = {row[0] for row in cur.fetchall()}
        for path in migration_files():
            version = path.stem
            if version in done:
                continue
            logger.info("Applying migration %s", version)
            cur.execute(path.read_text(encoding="utf-8"))
            cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied_now.append(version)
    if not applied_now:
        logger.info("Database schema is up to date")
    return applied_now
