"""
Structured Logging Utility for CommerceOps AI Engine.
Provides consistent, production-ready console and JSON log formatting.
"""

import sys
import logging
from typing import Optional

def setup_logger(name: str = "CommerceOps", level: str = "INFO") -> logging.Logger:
    """Configures and returns a thread-safe structured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()
