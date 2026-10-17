"""
PostgreSQL connection pool with explicit transaction scopes.

Every unit of work goes through ``Database.cursor()``, which commits on success
and rolls back on error, so connections never return to the pool mid-transaction.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2 import pool
from psycopg2.extensions import cursor as Cursor

from src.config import Settings
from src.utils.logger import get_logger

logger = get_logger("storage.db")


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pool = self._connect()

    def _connect(self) -> pool.ThreadedConnectionPool:
        attempts = max(1, self.settings.DB_CONNECT_RETRIES)
        for attempt in range(1, attempts + 1):
            try:
                created = pool.ThreadedConnectionPool(
                    minconn=self.settings.DB_POOL_MIN,
                    maxconn=self.settings.DB_POOL_MAX,
                    dsn=self.settings.database_dsn,
                    connect_timeout=self.settings.DB_CONNECT_TIMEOUT_SECONDS,
                    application_name="commerceops",
                )
                logger.info(
                    "Connected to PostgreSQL (%s:%s/%s)",
                    self.settings.POSTGRES_HOST,
                    self.settings.POSTGRES_PORT,
                    self.settings.POSTGRES_DB,
                )
                return created
            except psycopg2.OperationalError as exc:
                if attempt == attempts:
                    raise
                delay = min(2.0 * attempt, 10.0)
                logger.warning(
                    "PostgreSQL not reachable (attempt %d/%d): %s; retrying in %.0fs",
                    attempt,
                    attempts,
                    str(exc).strip(),
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")

    @contextmanager
    def cursor(self) -> Iterator[Cursor]:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def read_only_cursor(self) -> Iterator[Cursor]:
        """A READ ONLY transaction that is always rolled back, whatever the statements did."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                yield cur
        finally:
            conn.rollback()
            self._pool.putconn(conn)

    def close(self) -> None:
        if not self._pool.closed:
            self._pool.closeall()
