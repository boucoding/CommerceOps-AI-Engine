"""
Catalog synchronization: loads products into the store and (re)embeds only what changed.
"""

from __future__ import annotations

from typing import List

from src.models.domain import Product
from src.retrieval.embeddings import EmbeddingService
from src.storage.base import CatalogEntry, Store, content_hash
from src.utils.logger import get_logger

logger = get_logger("catalog")


def sync_catalog(store: Store, products: List[Product], embedder: EmbeddingService, reset_stock: bool) -> int:
    """
    Upserts ``products``. Embeddings are computed only for products whose
    embedding text or embedding model changed since the last sync. Existing stock
    levels are preserved unless ``reset_stock`` is set.

    Returns the number of products that were (re)embedded.
    """
    fingerprints = store.embedding_fingerprints()
    hashes = {p.product_id: content_hash(p) for p in products}
    stale = [p for p in products if fingerprints.get(p.product_id) != (embedder.model, hashes[p.product_id])]
    vectors = embedder.embed([p.embedding_text() for p in stale]) if stale else []
    by_id = dict(zip((p.product_id for p in stale), vectors, strict=False))

    entries = [
        CatalogEntry(
            product=p,
            embedding=by_id.get(p.product_id),
            embedding_model=embedder.model,
            embedding_hash=hashes[p.product_id],
        )
        for p in products
    ]
    store.upsert_catalog(entries, reset_stock=reset_stock)
    logger.info(
        "Catalog synced: %d products, %d embedded with %s, stock %s",
        len(products),
        len(stale),
        embedder.model,
        "reset from source" if reset_stock else "preserved",
    )
    return len(stale)
