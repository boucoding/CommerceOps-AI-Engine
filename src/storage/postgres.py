"""
PostgreSQL + pgvector store.

- Dense retrieval: ``ORDER BY embedding <=> query LIMIT k`` so the HNSW index is used.
- Lexical retrieval: generated ``tsvector`` columns with GIN indexes produce candidates,
  which are ranked with the shared BM25 scorer and persisted corpus statistics.
- SKU typo recovery: ``pg_trgm`` similarity on ``product_id`` (GIN trigram index).
- Reservations: the email row, stock decrements and order lines are written in one
  transaction, with the affected product rows locked via ``SELECT ... FOR UPDATE``
  (taken in primary-key order to avoid deadlocks between concurrent orders).
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from src.models.domain import EmailOutcome, OrderLine, Product, RequestedItem
from src.retrieval import bm25
from src.retrieval.dense import normalize_rows, rank_by_cosine
from src.retrieval.text import STOPWORDS, fold, stem
from src.storage.base import CatalogEntry, DuplicateEmailError, decide_reservation
from src.storage.db import Database

_PRODUCT_COLUMNS = "product_id, name, category, description, stock, seasons, price"
_WORD_RE = re.compile(r"[a-z0-9]+")
_LEXICAL_CANDIDATES = 50
_DENSE_CANDIDATES = 40
_STATS_TTL_SECONDS = 300.0


def _row_to_product(row: Sequence[Any]) -> Product:
    return Product(
        product_id=row[0],
        name=row[1],
        category=row[2],
        description=row[3],
        stock=int(row[4]),
        seasons=row[5],
        price=float(row[6]),
    )


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.7g}" for v in vector) + "]"


def _or_tsquery(text: str, max_terms: int = 64) -> Optional[str]:
    """Builds an OR tsquery from plain alphanumeric words (safe: no tsquery operators survive)."""
    words: List[str] = []
    for word in _WORD_RE.findall(fold(text)):
        if word in STOPWORDS:
            continue
        # Include our own light stem too: PostgreSQL's English stemmer maps "scarves" to "scarv", not "scarf".
        for form in (word, stem(word)):
            if form not in words:
                words.append(form)
    return " | ".join(words[:max_terms]) or None


class PostgresStore:
    backend = "postgres"

    def __init__(self, db: Database):
        self.db = db
        self._stats: Optional[bm25.CorpusStats] = None
        self._stats_loaded_at = 0.0
        self._stats_lock = threading.Lock()

    def ping(self) -> bool:
        try:
            with self.db.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone() == (1,)
        except psycopg2.Error:
            return False

    # ------------------------------------------------------------------ catalog
    def embedding_fingerprints(self) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        with self.db.cursor() as cur:
            cur.execute("SELECT product_id, embedding_model, embedding_hash FROM products")
            return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    def upsert_catalog(self, entries: Sequence[CatalogEntry], reset_stock: bool) -> None:
        if not entries:
            return
        records = [
            (
                e.product.product_id,
                e.product.name,
                e.product.category,
                e.product.description,
                e.product.stock,
                e.product.seasons,
                e.product.price,
                _vector_literal(e.embedding) if e.embedding is not None else None,
                e.embedding_model if e.embedding is not None else None,
                e.embedding_hash if e.embedding is not None else None,
            )
            for e in entries
        ]
        stock_rule = "EXCLUDED.stock" if reset_stock else "products.stock"
        sql = f"""
            INSERT INTO products (product_id, name, category, description, stock, seasons, price,
                                  embedding, embedding_model, embedding_hash)
            VALUES %s
            ON CONFLICT (product_id) DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                description = EXCLUDED.description,
                stock = {stock_rule},
                seasons = EXCLUDED.seasons,
                price = EXCLUDED.price,
                embedding = COALESCE(EXCLUDED.embedding, products.embedding),
                embedding_model = COALESCE(EXCLUDED.embedding_model, products.embedding_model),
                embedding_hash = COALESCE(EXCLUDED.embedding_hash, products.embedding_hash)
        """
        with self.db.cursor() as cur:
            execute_values(cur, sql, records, template="(%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)")
            cur.execute(f"SELECT {_PRODUCT_COLUMNS} FROM products")
            stats = bm25.CorpusStats.from_documents(_row_to_product(r).embedding_text() for r in cur.fetchall())
            cur.execute(
                "INSERT INTO lexical_stats (id, stats) VALUES (TRUE, %s::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET stats = EXCLUDED.stats, updated_at = now()",
                (json.dumps(stats.to_dict()),),
            )
        with self._stats_lock:
            self._stats, self._stats_loaded_at = stats, time.monotonic()

    def _corpus_stats(self) -> bm25.CorpusStats:
        with self._stats_lock:
            if self._stats is not None and time.monotonic() - self._stats_loaded_at < _STATS_TTL_SECONDS:
                return self._stats
        with self.db.cursor() as cur:
            cur.execute("SELECT stats FROM lexical_stats WHERE id")
            row = cur.fetchone()
        stats = bm25.CorpusStats.from_dict(row[0]) if row else bm25.CorpusStats()
        with self._stats_lock:
            self._stats, self._stats_loaded_at = stats, time.monotonic()
        return stats

    def list_products(self) -> List[Product]:
        with self.db.cursor() as cur:
            cur.execute(f"SELECT {_PRODUCT_COLUMNS} FROM products ORDER BY product_id")
            return [_row_to_product(r) for r in cur.fetchall()]

    def get_products(self, product_ids: Sequence[str]) -> Dict[str, Product]:
        if not product_ids:
            return {}
        with self.db.cursor() as cur:
            cur.execute(f"SELECT {_PRODUCT_COLUMNS} FROM products WHERE product_id = ANY(%s)", (list(product_ids),))
            return {r[0]: _row_to_product(r) for r in cur.fetchall()}

    # ---------------------------------------------------------------- retrieval
    def dense_search(self, vector: Sequence[float], limit: int) -> List[str]:
        """HNSW candidate generation, then exact shared cosine ranking of the candidates."""
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT product_id, embedding::text FROM products WHERE embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (_vector_literal(vector), max(limit * 2, _DENSE_CANDIDATES)),
            )
            rows = cur.fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        matrix = normalize_rows(np.array([json.loads(r[1]) for r in rows], dtype=np.float32))
        return rank_by_cosine(ids, matrix, vector, limit)

    def lexical_search(self, text: str, limit: int) -> List[str]:
        query = _or_tsquery(text)
        if query is None:
            return []
        with self.db.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_PRODUCT_COLUMNS} FROM products, to_tsquery('english', %s) AS q
                WHERE search_tsv @@ q
                ORDER BY ts_rank_cd(search_tsv, q) DESC, product_id
                LIMIT %s
                """,
                (query, max(limit * 5, _LEXICAL_CANDIDATES)),
            )
            docs = [(r[0], _row_to_product(r).embedding_text()) for r in cur.fetchall()]
        return [pid for pid, _ in bm25.rank(text, docs, self._corpus_stats(), limit)]

    def sku_candidates(self, token: str, limit: int) -> List[str]:
        with self.db.cursor() as cur:
            cur.execute(
                """
                SELECT product_id FROM products
                WHERE product_id = %s OR product_id %% %s
                ORDER BY similarity(product_id, %s) DESC, product_id
                LIMIT %s
                """,
                (token, token, token, limit),
            )
            return [r[0] for r in cur.fetchall()]

    def name_candidates(self, text: str, limit: int) -> List[Product]:
        query = _or_tsquery(text)
        if query is None:
            return []
        with self.db.cursor() as cur:
            cur.execute(
                f"""
                SELECT {_PRODUCT_COLUMNS} FROM products, to_tsquery('english', %s) AS q
                WHERE name_tsv @@ q
                ORDER BY ts_rank_cd(name_tsv, q) DESC, product_id
                LIMIT %s
                """,
                (query, limit),
            )
            return [_row_to_product(r) for r in cur.fetchall()]

    # ------------------------------------------------------------ orders/audit
    @staticmethod
    def _claim(cur: Any, email_id: str) -> bool:
        """Returns True if newly claimed, False if resuming; raises if already processed."""
        cur.execute(
            "INSERT INTO emails (email_id, status) VALUES (%s, 'processing') "
            "ON CONFLICT (email_id) DO NOTHING RETURNING email_id",
            (email_id,),
        )
        if cur.fetchone() is not None:
            return True
        cur.execute("SELECT outcome IS NOT NULL FROM emails WHERE email_id = %s FOR UPDATE", (email_id,))
        row = cur.fetchone()
        if row and row[0]:
            raise DuplicateEmailError(email_id)
        return False

    @staticmethod
    def _load_lines(cur: Any, email_id: str) -> List[OrderLine]:
        cur.execute(
            "SELECT product_id, product_name, requested_quantity, fulfilled_quantity, status, remaining_stock "
            "FROM order_lines WHERE email_id = %s ORDER BY line_no",
            (email_id,),
        )
        return [OrderLine(*row) for row in cur.fetchall()]

    def commit_order(self, email_id: str, items: Sequence[RequestedItem]) -> List[OrderLine]:
        with self.db.cursor() as cur:
            if not self._claim(cur, email_id):
                return self._load_lines(cur, email_id)

            product_ids = sorted({item.product_id for item in items})
            cur.execute(
                f"SELECT {_PRODUCT_COLUMNS} FROM products WHERE product_id = ANY(%s) ORDER BY product_id FOR UPDATE",
                (product_ids,),
            )
            locked = {r[0]: _row_to_product(r) for r in cur.fetchall()}
            lines: List[OrderLine] = []
            for item in items:
                product = locked.get(item.product_id)
                if product is None:
                    raise KeyError(f"Unknown product {item.product_id}")
                line, new_stock = decide_reservation(item, product)
                if new_stock != product.stock:
                    cur.execute("UPDATE products SET stock = %s WHERE product_id = %s", (new_stock, product.product_id))
                    product.stock = new_stock
                lines.append(line)
            if lines:
                execute_values(
                    cur,
                    "INSERT INTO order_lines (email_id, line_no, product_id, product_name, requested_quantity, "
                    "fulfilled_quantity, status, remaining_stock) VALUES %s",
                    [
                        (
                            email_id,
                            idx,
                            ln.product_id,
                            ln.product_name,
                            ln.requested_quantity,
                            ln.fulfilled_quantity,
                            ln.status,
                            ln.remaining_stock,
                        )
                        for idx, ln in enumerate(lines, 1)
                    ],
                )
            return lines

    def claim_email(self, email_id: str) -> None:
        with self.db.cursor() as cur:
            self._claim(cur, email_id)

    def save_outcome(self, outcome: EmailOutcome) -> None:
        payload = outcome.to_dict()
        payload.pop("lines", None)  # lines live in order_lines
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO emails (email_id, category, status, degraded, outcome, processed_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, now())
                ON CONFLICT (email_id) DO UPDATE SET
                    category = EXCLUDED.category, status = EXCLUDED.status, degraded = EXCLUDED.degraded,
                    outcome = EXCLUDED.outcome, processed_at = EXCLUDED.processed_at
                """,
                (outcome.email_id, outcome.category, outcome.status.value, outcome.degraded, json.dumps(payload)),
            )

    def get_outcome(self, email_id: str) -> Optional[EmailOutcome]:
        with self.db.cursor() as cur:
            cur.execute("SELECT outcome FROM emails WHERE email_id = %s AND outcome IS NOT NULL", (email_id,))
            row = cur.fetchone()
            if row is None:
                return None
            data = dict(row[0])
            data["lines"] = [vars(line) for line in self._load_lines(cur, email_id)]
            return EmailOutcome.from_dict(data)

    def reset_operations(self) -> None:
        with self.db.cursor() as cur:
            cur.execute("TRUNCATE order_lines, emails")

    def close(self) -> None:
        self.db.close()
