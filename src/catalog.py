"""
Catalog synchronization: upserts products and (re)embeds only what changed.
"""

from __future__ import annotations

from typing import List

from src.llm.base import LLMClient
from src.models.domain import Product
from src.storage.postgres import PostgresStore, content_hash
from src.utils.logger import get_logger

logger = get_logger("catalog")


def sync_catalog(store: PostgresStore, products: List[Product], llm: LLMClient, reset_stock: bool) -> int:
    """
    Embeddings are computed only for products whose embedding text or embedding model
    changed since the last sync. Existing stock is preserved unless ``reset_stock``.
    Returns the number of products embedded.
    """
    fingerprints = store.embedding_fingerprints()
    stale = [p for p in products if fingerprints.get(p.product_id) != (llm.embedding_model, content_hash(p))]
    vectors = llm.embed([p.embedding_text() for p in stale]) if stale else []
    store.upsert_products(
        products, dict(zip((p.product_id for p in stale), vectors, strict=True)), llm.embedding_model, reset_stock
    )
    logger.info(
        "Catalog synced: %d products, %d embedded, stock %s",
        len(products),
        len(stale),
        "reset" if reset_stock else "preserved",
    )
    return len(stale)
