"""
In-memory store: the default for local runs, tests and the offline demo.

Thread-safe (one lock guards stock, claims and outcomes). Retrieval primitives
mirror the PostgreSQL backend: cosine similarity for dense search and BM25 for
lexical search.
"""

from __future__ import annotations

import copy
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.models.domain import EmailOutcome, OrderLine, Product, RequestedItem
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import normalize_rows, rank_by_cosine
from src.storage.base import CatalogEntry, DuplicateEmailError, decide_reservation


class MemoryStore:
    backend = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[str, CatalogEntry] = {}
        self._claims: Dict[str, List[OrderLine]] = {}
        self._outcomes: Dict[str, EmailOutcome] = {}
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._bm25 = BM25Index([], [])

    def ping(self) -> bool:
        return True

    # ------------------------------------------------------------------ catalog
    def embedding_fingerprints(self) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        with self._lock:
            return {pid: (e.embedding_model, e.embedding_hash) for pid, e in self._entries.items()}

    def upsert_catalog(self, entries: Sequence[CatalogEntry], reset_stock: bool) -> None:
        with self._lock:
            for entry in entries:
                product = copy.copy(entry.product)
                existing = self._entries.get(product.product_id)
                if existing is not None:
                    if not reset_stock:
                        product.stock = existing.product.stock
                    if entry.embedding is None:
                        entry = CatalogEntry(
                            product, existing.embedding, existing.embedding_model, existing.embedding_hash
                        )
                self._entries[product.product_id] = CatalogEntry(
                    product, entry.embedding, entry.embedding_model, entry.embedding_hash
                )
            self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._ids = sorted(self._entries)
        vectors = [self._entries[pid].embedding for pid in self._ids]
        if self._ids and all(v is not None for v in vectors):
            self._matrix = normalize_rows(np.asarray(vectors, dtype=np.float32))
        else:
            self._matrix = None
        self._bm25 = BM25Index(self._ids, [self._entries[pid].product.embedding_text() for pid in self._ids])

    def list_products(self) -> List[Product]:
        with self._lock:
            return [copy.copy(self._entries[pid].product) for pid in self._ids]

    def get_products(self, product_ids: Sequence[str]) -> Dict[str, Product]:
        with self._lock:
            return {pid: copy.copy(self._entries[pid].product) for pid in product_ids if pid in self._entries}

    # ---------------------------------------------------------------- retrieval
    def dense_search(self, vector: Sequence[float], limit: int) -> List[str]:
        with self._lock:
            if self._matrix is None:
                return []
            return rank_by_cosine(self._ids, self._matrix, vector, limit)

    def lexical_search(self, text: str, limit: int) -> List[str]:
        with self._lock:
            return [pid for pid, _ in self._bm25.search(text, limit)]

    def sku_candidates(self, token: str, limit: int) -> List[str]:
        # The whole catalog is scanned in memory; best_sku_match() does the scoring.
        with self._lock:
            return list(self._ids)

    def name_candidates(self, text: str, limit: int) -> List[Product]:
        return self.list_products()

    # ------------------------------------------------------------ orders/audit
    def _claim(self, email_id: str) -> Optional[List[OrderLine]]:
        if email_id in self._outcomes:
            raise DuplicateEmailError(email_id)
        return self._claims.get(email_id)

    def commit_order(self, email_id: str, items: Sequence[RequestedItem]) -> List[OrderLine]:
        with self._lock:
            existing = self._claim(email_id)
            if existing is not None:
                return copy.deepcopy(existing)
            lines: List[OrderLine] = []
            for item in items:
                entry = self._entries.get(item.product_id)
                if entry is None:
                    raise KeyError(f"Unknown product {item.product_id}")
                line, new_stock = decide_reservation(item, entry.product)
                entry.product.stock = new_stock
                lines.append(line)
            self._claims[email_id] = lines
            return copy.deepcopy(lines)

    def claim_email(self, email_id: str) -> None:
        with self._lock:
            if self._claim(email_id) is None:
                self._claims[email_id] = []

    def save_outcome(self, outcome: EmailOutcome) -> None:
        with self._lock:
            self._outcomes[outcome.email_id] = copy.deepcopy(outcome)

    def get_outcome(self, email_id: str) -> Optional[EmailOutcome]:
        with self._lock:
            outcome = self._outcomes.get(email_id)
            return copy.deepcopy(outcome) if outcome else None

    def reset_operations(self) -> None:
        with self._lock:
            self._claims.clear()
            self._outcomes.clear()

    def close(self) -> None:
        pass
