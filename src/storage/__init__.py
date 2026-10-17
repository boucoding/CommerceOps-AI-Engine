"""PostgreSQL storage: connection pool, migrations and the store."""

from __future__ import annotations

from src.config import Settings
from src.storage.db import Database
from src.storage.migrate import apply_migrations
from src.storage.postgres import DuplicateEmailError, PostgresStore, UnknownProductError

__all__ = ["Database", "DuplicateEmailError", "PostgresStore", "UnknownProductError", "open_store"]


def open_store(settings: Settings, migrate: bool = True) -> PostgresStore:
    db = Database(settings)
    if migrate:
        apply_migrations(db)
    return PostgresStore(db)
