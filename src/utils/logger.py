"""
Structured logging for CommerceOps AI Engine.

Every record carries the email/request being processed (via a context variable)
so concurrent batch workers and API requests stay traceable. Log messages never
include email bodies or subjects: they can contain personal data.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Iterator, Optional

correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)

_ROOT_NAME = "commerceops"


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """(Re)configures the package logger. Safe to call more than once."""
    root = logging.getLogger(_ROOT_NAME)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_CorrelationFilter())
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(correlation_id)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


@contextmanager
def correlation(value: str) -> Iterator[None]:
    token = correlation_id.set(value)
    try:
        yield
    finally:
        correlation_id.reset(token)


configure_logging()
