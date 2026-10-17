"""
Application wiring shared by the CLI, the HTTP API and the evaluation harness.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, List, Optional, Sequence

from src.agent.agent import EmailAgent
from src.agent.tools import Toolbox
from src.catalog import sync_catalog
from src.config import Settings
from src.llm import build_llm_client
from src.llm.base import LLMClient
from src.loaders import load_products
from src.models.domain import CustomerEmail, EmailOutcome, Product
from src.sandbox.sql import SQLSandbox
from src.storage import PostgresStore, open_store
from src.utils.logger import get_logger

logger = get_logger("engine")


def cached_embedder(llm: LLMClient, size: int = 2048) -> Callable[[str], List[float]]:
    """Query-embedding cache (lru_cache is thread-safe; a rare duplicate computation is harmless)."""

    @lru_cache(maxsize=size)
    def _embed(text: str) -> tuple[float, ...]:
        return tuple(llm.embed([text])[0])

    return lambda text: list(_embed(text))


@dataclass
class Engine:
    settings: Settings
    llm: LLMClient
    store: PostgresStore
    sandbox: SQLSandbox
    agent: EmailAgent

    def process(self, emails: Sequence[CustomerEmail]) -> List[EmailOutcome]:
        """Processes emails in order (or with BATCH_WORKERS threads). Email IDs must be unique."""
        ids = [e.email_id for e in emails]
        if len(set(ids)) != len(ids):
            raise ValueError("email_id values must be unique within a batch")
        logger.info("Processing %d email(s) with %d worker(s)", len(emails), self.settings.BATCH_WORKERS)
        if self.settings.BATCH_WORKERS == 1 or len(emails) <= 1:
            return [self.agent.handle(e) for e in emails]
        with ThreadPoolExecutor(max_workers=self.settings.BATCH_WORKERS, thread_name_prefix="agent") as pool:
            return list(pool.map(self.agent.handle, emails))

    def close(self) -> None:
        save = getattr(self.llm, "save", None)
        if callable(save):
            logger.info("Saved %d recorded LLM exchanges to %s", save(), self.settings.REPLAY_FIXTURES_PATH)
        self.store.close()


def build_engine(
    settings: Settings,
    reset: bool = False,
    products: Optional[List[Product]] = None,
    llm: Optional[LLMClient] = None,
) -> Engine:
    """
    Connects to PostgreSQL (applying migrations), syncs the catalog and wires the agent.
    ``reset`` clears processed emails and restores catalog stock from the source file.
    """
    llm = llm or build_llm_client(settings)
    store = open_store(settings)
    if reset:
        store.reset_operations()
    sync_catalog(
        store, products if products is not None else load_products(settings.CATALOG_CSV_PATH), llm, reset_stock=reset
    )
    sandbox = SQLSandbox(store.db, settings)
    toolbox = Toolbox(store, sandbox, cached_embedder(llm), settings)
    return Engine(settings, llm, store, sandbox, EmailAgent(llm, store, toolbox, settings))
