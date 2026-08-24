"""
Database Connection Pooling & Health Management for CommerceOps AI Engine.
"""

import time
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional, Generator
from src.config import settings
from src.utils.logger import logger

class DatabaseManager:
    """Thread-safe connection pool manager for PostgreSQL + pgvector."""
    _instance: Optional['DatabaseManager'] = None
    _pool: Optional[pool.ThreadedConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def initialize_pool(self, max_retries: int = 5, retry_interval: float = 2.0) -> bool:
        """Initializes the threaded connection pool with retry mechanism."""
        if self._pool is not None and not self._pool.closed:
            return True

        for attempt in range(1, max_retries + 1):
            try:
                self._pool = pool.ThreadedConnectionPool(
                    minconn=settings.DB_POOL_MIN,
                    maxconn=settings.DB_POOL_MAX,
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    database=settings.POSTGRES_DB,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                    connect_timeout=3
                )
                logger.info(f"Connected to PostgreSQL database '{settings.POSTGRES_DB}' on {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
                return True
            except Exception as e:
                logger.warning(f"Database connection attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_interval)

        logger.warning("PostgreSQL connection unavailable. Operating in local memory fallback mode.")
        return False

    @contextmanager
    def get_connection(self) -> Generator[Optional[psycopg2.extensions.connection], None, None]:
        """Context manager for acquiring and releasing pool connections safely."""
        if self._pool is None or self._pool.closed:
            yield None
            return

        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        finally:
            if conn and self._pool and not self._pool.closed:
                self._pool.putconn(conn)

    def close_all(self):
        """Closes all connections in the pool."""
        if self._pool and not self._pool.closed:
            self._pool.closeall()
            logger.info("Database connection pool closed.")

db_manager = DatabaseManager()
