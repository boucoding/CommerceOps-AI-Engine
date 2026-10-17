"""
Hybrid product retrieval.

Results are split into two groups so each pipeline stage can use them precisely:

- ``mentioned``: products the customer explicitly referenced, by SKU (typo-tolerant,
  e.g. DHN0987 -> CHN0987, '[CBT 89 01]' -> CBT8901) or by product name.
- ``related``: semantically/lexically relevant products, ranked by Reciprocal Rank
  Fusion of dense vector search and lexical (BM25 / full-text) search.

Only these products, never the full catalog, are placed into prompts, so prompt
size is independent of catalog size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.config import Settings
from src.llm.base import LLMError
from src.models.domain import Product
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.fuzzy_matcher import best_sku_match, extract_sku_tokens, find_name_mentions, normalize_sku_text
from src.storage.base import Store
from src.utils.logger import get_logger

logger = get_logger("retrieval")

_SKU_CANDIDATES = 5
_NAME_CANDIDATES = 30


@dataclass
class SkuResolution:
    token: str  # as written (normalized), e.g. 'DHN0987'
    product_id: str  # catalog SKU, e.g. 'CHN0987'
    score: float


@dataclass
class Retrieval:
    mentioned: List[Product] = field(default_factory=list)
    related: List[Product] = field(default_factory=list)
    sku_resolutions: List[SkuResolution] = field(default_factory=list)
    unknown_sku_tokens: List[str] = field(default_factory=list)
    degraded_reason: Optional[str] = None

    @property
    def products(self) -> List[Product]:
        return self.mentioned + self.related

    @property
    def product_ids(self) -> List[str]:
        return [p.product_id for p in self.products]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int) -> List[str]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


class HybridRetriever:
    def __init__(self, store: Store, embedder: EmbeddingService, settings: Settings):
        self.store = store
        self.embedder = embedder
        self.settings = settings

    def find_mentions(self, text: str) -> Retrieval:
        normalized = normalize_sku_text(text)
        result = Retrieval()
        ordered_ids: List[str] = []

        for token in extract_sku_tokens(normalized):
            match = best_sku_match(
                token, self.store.sku_candidates(token, _SKU_CANDIDATES), self.settings.SKU_FUZZY_THRESHOLD
            )
            if match is None:
                result.unknown_sku_tokens.append(token)
                continue
            product_id, score = match
            result.sku_resolutions.append(SkuResolution(token, product_id, score))
            if product_id != token:
                logger.info("Resolved SKU typo %s -> %s (score %.1f)", token, product_id, score)
            if product_id not in ordered_ids:
                ordered_ids.append(product_id)

        candidates = self.store.name_candidates(normalized, _NAME_CANDIDATES)
        for mention in find_name_mentions(normalized, candidates, self.settings.NAME_MATCH_THRESHOLD):
            if mention.product.product_id not in ordered_ids:
                ordered_ids.append(mention.product.product_id)

        products = self.store.get_products(ordered_ids)
        result.mentioned = [products[pid] for pid in ordered_ids if pid in products]
        return result

    def related(self, text: str, limit: int, exclude: Sequence[str] = ()) -> List[Product]:
        """Dense + lexical fusion. Raises LLMError if the query embedding cannot be computed."""
        if limit <= 0:
            return []
        pool = limit * 4 + len(exclude)
        dense = self.store.dense_search(self.embedder.embed_one(text), pool)
        return self._fuse(dense, self.store.lexical_search(text, pool), limit, exclude)

    def _fuse(self, dense: List[str], lexical: List[str], limit: int, exclude: Sequence[str]) -> List[Product]:
        fused = [pid for pid in reciprocal_rank_fusion([dense, lexical], self.settings.RRF_K) if pid not in exclude]
        top = fused[:limit]
        products = self.store.get_products(top)
        return [products[pid] for pid in top if pid in products]

    def retrieve(self, text: str, top_k: int) -> Retrieval:
        result = self.find_mentions(text)
        query = normalize_sku_text(text)
        exclude = [p.product_id for p in result.mentioned]
        try:
            result.related = self.related(query, top_k, exclude)
        except LLMError as exc:
            # Embedding service unavailable: keep serving with lexical retrieval only.
            logger.warning("Dense retrieval unavailable, using lexical only: %s", exc)
            result.degraded_reason = f"retrieval: {exc}"
            result.related = self._fuse([], self.store.lexical_search(query, top_k * 4 + len(exclude)), top_k, exclude)
        return result

    def alternatives(self, product: Product, limit: int) -> List[Product]:
        """In-stock substitutes for ``product``, same category first."""
        try:
            candidates = self.related(product.embedding_text(), limit * 4, exclude=[product.product_id])
        except LLMError:
            candidates = self._fuse(
                [], self.store.lexical_search(product.embedding_text(), limit * 8), limit * 4, [product.product_id]
            )
        in_stock = [p for p in candidates if p.stock > 0]
        in_stock.sort(key=lambda p: p.category != product.category)  # stable: keeps relevance order
        return in_stock[:limit]
