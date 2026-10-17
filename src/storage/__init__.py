"""Storage backends (in-memory and PostgreSQL + pgvector)."""

from __future__ import annotations

from src.config import Settings, StorageBackend
from src.storage.base import CatalogEntry, DuplicateEmailError, Store
from src.storage.memory import MemoryStore

__all__ = ["CatalogEntry", "DuplicateEmailError", "MemoryStore", "Store", "build_store"]


def build_store(settings: Settings, migrate: bool = True) -> Store:
    if settings.STORAGE_BACKEND == StorageBackend.MEMORY:
        return MemoryStore()

    from src.storage.db import Database
    from src.storage.migrate import apply_migrations
    from src.storage.postgres import PostgresStore

    db = Database(settings)
    if migrate:
        apply_migrations(db)
    return PostgresStore(db)
