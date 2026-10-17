"""
Application wiring shared by the CLI, the HTTP API and the evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.catalog import sync_catalog
from src.config import LLMMode, Settings, StorageBackend
from src.llm import build_llm_client
from src.llm.base import LLMClient
from src.loaders import load_products
from src.models.domain import Product
from src.pipeline.orchestrator import CommerceOpsOrchestrator
from src.retrieval.embeddings import EmbeddingService
from src.storage import Store, build_store
from src.utils.logger import get_logger

logger = get_logger("engine")


@dataclass
class Engine:
    settings: Settings
    mode: LLMMode
    store: Store
    llm: Optional[LLMClient]
    orchestrator: CommerceOpsOrchestrator

    def close(self) -> None:
        save = getattr(self.llm, "save", None)
        if callable(save):
            count = save()
            logger.info("Saved %d recorded LLM responses to %s", count, self.settings.REPLAY_FIXTURES_PATH)
        self.store.close()


def build_engine(
    settings: Settings,
    mode: Optional[LLMMode] = None,
    products: Optional[List[Product]] = None,
    reset: bool = False,
    store: Optional[Store] = None,
    llm: Optional[LLMClient] = None,
) -> Engine:
    """
    Builds a ready-to-use engine: LLM client, store (migrated), catalog synced.

    ``reset`` restores catalog stock from the source and clears processed emails
    (PostgreSQL); the in-memory store always starts fresh.
    """
    resolved = mode or settings.resolved_llm_mode()
    if llm is None:
        llm = build_llm_client(settings, resolved)
    store = store or build_store(settings)
    if reset and settings.STORAGE_BACKEND == StorageBackend.POSTGRES:
        store.reset_operations()

    embedder = EmbeddingService(llm, settings.EMBEDDING_DIMENSIONS)
    catalog = products if products is not None else load_products(settings.CATALOG_CSV_PATH)
    sync_catalog(store, catalog, embedder, reset_stock=reset or store.backend == "memory")

    orchestrator = CommerceOpsOrchestrator(store, llm, settings, embedder)
    return Engine(settings, resolved, store, llm, orchestrator)
