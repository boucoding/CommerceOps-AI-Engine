"""
Storage contract shared by the in-memory and PostgreSQL backends.

A store owns three concerns: the product catalog (including the search
primitives used by hybrid retrieval), stock reservations, and the audit trail of
processed emails. Reservations and the email record are written atomically, which
makes processing idempotent: a re-delivered email never reserves stock twice.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from src.models.domain import STATUS_CREATED, STATUS_OUT_OF_STOCK, EmailOutcome, OrderLine, Product, RequestedItem


class DuplicateEmailError(Exception):
    """The email was already fully processed; its stored outcome is authoritative."""


# Idempotency contract for commit_order / claim_email:
# - unknown email id                   -> claim it (and reserve stock for orders)
# - claimed but no outcome saved yet   -> resume: return the lines already reserved, never reserve again
# - outcome saved                      -> raise DuplicateEmailError


@dataclass
class CatalogEntry:
    product: Product
    embedding: Optional[List[float]]
    embedding_model: str
    embedding_hash: str


def content_hash(product: Product) -> str:
    return hashlib.sha256(product.embedding_text().encode("utf-8")).hexdigest()[:32]


def decide_reservation(item: RequestedItem, product: Product) -> Tuple[OrderLine, int]:
    """
    All-or-nothing rule per line: reserve the full quantity if stock allows,
    otherwise mark the line out of stock and leave stock untouched.
    Returns the line and the new stock level.
    """
    stock = product.stock
    requested = (stock if stock > 0 else 1) if item.wants_all_available else max(1, int(item.quantity or 1))
    if stock >= requested:
        new_stock = stock - requested
        return OrderLine(product.product_id, product.name, requested, requested, STATUS_CREATED, new_stock), new_stock
    return OrderLine(product.product_id, product.name, requested, 0, STATUS_OUT_OF_STOCK, stock), stock


class Store(Protocol):
    backend: str

    def ping(self) -> bool: ...

    # Catalog
    def embedding_fingerprints(self) -> Dict[str, Tuple[Optional[str], Optional[str]]]: ...
    def upsert_catalog(self, entries: Sequence[CatalogEntry], reset_stock: bool) -> None: ...
    def list_products(self) -> List[Product]: ...
    def get_products(self, product_ids: Sequence[str]) -> Dict[str, Product]: ...

    # Retrieval primitives (ids ranked best first)
    def dense_search(self, vector: Sequence[float], limit: int) -> List[str]: ...
    def lexical_search(self, text: str, limit: int) -> List[str]: ...
    def sku_candidates(self, token: str, limit: int) -> List[str]: ...
    def name_candidates(self, text: str, limit: int) -> List[Product]: ...

    # Orders and audit trail
    def commit_order(self, email_id: str, items: Sequence[RequestedItem]) -> List[OrderLine]: ...
    def claim_email(self, email_id: str) -> None: ...
    def save_outcome(self, outcome: EmailOutcome) -> None: ...
    def get_outcome(self, email_id: str) -> Optional[EmailOutcome]: ...
    def reset_operations(self) -> None: ...
    def close(self) -> None: ...
