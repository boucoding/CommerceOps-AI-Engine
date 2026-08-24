"""
Hybrid Vector Store combining PgVector HNSW Cosine Distance, Trigram Text Matching, and Score Fusion.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from src.database.connection import db_manager
from src.models.domain import Product
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.fuzzy_matcher import FuzzyProductMatcher
from src.config import settings
from src.utils.logger import logger

class HybridVectorStore:
    """
    Enterprise Hybrid Vector Store.
    Scalable to 100k+ SKUs:
    - Primary: PostgreSQL with pgvector HNSW indexing and pg_trgm fuzzy search.
    - Secondary: In-memory vector cache with RapidFuzz score fusion for local execution.
    """
    def __init__(self, products: List[Product], embedding_service: EmbeddingService):
        self.products = products
        self.embedding_service = embedding_service
        self.fuzzy_matcher = FuzzyProductMatcher(products)
        self.product_map: Dict[str, Product] = {p.product_id.upper(): p for p in products}

        # Build in-memory embeddings matrix for instant fallback
        self._ensure_product_embeddings()

    def _ensure_product_embeddings(self):
        """Generates embeddings for products that do not have them yet."""
        missing = [p for p in self.products if p.embedding is None]
        if missing:
            logger.info(f"Generating embeddings for {len(missing)} products...")
            texts = [p.to_rag_document() for p in missing]
            embeddings = self.embedding_service.get_batch_embeddings(texts)
            for p, emb in zip(missing, embeddings):
                p.embedding = emb

        # Create numpy array for local vector search
        self.doc_vectors = np.array([p.embedding for p in self.products], dtype=np.float32)

    def search(self, query: str, top_k: int = 4, alpha: float = 0.6) -> List[Product]:
        """
        Hybrid search combining Vector Cosine Similarity and Fuzzy SKU/Keyword Matching.
        alpha = weight for vector similarity (0.6 vector, 0.4 lexical/fuzzy).
        """
        normalized_query = self.fuzzy_matcher.normalize_sku_in_text(query)
        query_embedding = self.embedding_service.get_embedding(normalized_query)

        # 1. Try PostgreSQL pgvector search
        db_results = self._search_postgres(normalized_query, query_embedding, top_k)
        if db_results:
            return db_results

        # 2. In-Memory Hybrid Search Fallback
        return self._search_in_memory(normalized_query, query_embedding, top_k, alpha)

    def _search_postgres(self, query: str, query_embedding: List[float], top_k: int) -> Optional[List[Product]]:
        """Queries PostgreSQL using pgvector HNSW cosine distance and pg_trgm similarity."""
        with db_manager.get_connection() as conn:
            if conn is None:
                return None

            try:
                with conn.cursor() as cur:
                    emb_str = f"[{','.join(map(str, query_embedding))}]"
                    # Hybrid SQL combining cosine similarity and trigram text match
                    sql = """
                    SELECT product_id, name, category, description, stock, seasons, price,
                           (1 - (embedding <=> %s::vector)) AS vector_score,
                           GREATEST(similarity(product_id, %s), similarity(name, %s)) AS trgm_score
                    FROM products
                    ORDER BY (0.6 * (1 - (embedding <=> %s::vector)) + 0.4 * GREATEST(similarity(product_id, %s), similarity(name, %s))) DESC
                    LIMIT %s;
                    """
                    cur.execute(sql, (emb_str, query, query, emb_str, query, query, top_k))
                    rows = cur.fetchall()
                    if rows:
                        results = []
                        for r in rows:
                            results.append(Product(
                                product_id=r[0],
                                name=r[1],
                                category=r[2],
                                description=r[3],
                                stock=int(r[4]),
                                seasons=r[5],
                                price=float(r[6])
                            ))
                        return results
            except Exception as e:
                logger.warning(f"PostgreSQL hybrid search failed: {e}. Falling back to in-memory search.")
                return None

    def _search_in_memory(self, query: str, query_embedding: List[float], top_k: int, alpha: float) -> List[Product]:
        """In-memory cosine similarity and fuzzy score fusion."""
        # 1. Check direct or fuzzy SKU mentions in query
        sku_tokens = self.fuzzy_matcher.extract_potential_sku_tokens(query)
        priority_products: List[Product] = []

        for token in sku_tokens:
            matched = self.fuzzy_matcher.match_sku_with_typo_tolerance(token, score_threshold=settings.FUZZY_THRESHOLD)
            if matched:
                prod, score = matched
                if prod not in priority_products:
                    priority_products.append(prod)

        # Also check product names mentioned directly
        name_matches = self.fuzzy_matcher.match_product_name(query, score_threshold=80.0)
        for prod, score in name_matches:
            if prod not in priority_products:
                priority_products.append(prod)

        # 2. Vector Cosine Similarity
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        doc_norms = np.linalg.norm(self.doc_vectors, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1.0
        normalized_docs = self.doc_vectors / doc_norms

        cosine_scores = np.dot(normalized_docs, q_vec)

        # 3. Combine scores
        scored: List[Tuple[float, Product]] = []
        for i, prod in enumerate(self.products):
            v_score = float(cosine_scores[i])
            f_score = 1.0 if prod in priority_products else 0.0
            combined = (alpha * v_score) + ((1.0 - alpha) * f_score)
            # Heavy boost if directly matched SKU
            if prod in priority_products:
                combined += 10.0
            scored.append((combined, prod))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]
