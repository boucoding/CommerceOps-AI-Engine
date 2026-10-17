"""
PostgreSQL + pgvector store: catalog, hybrid search, stock reservations and the
email audit trail.

- Hybrid search runs in one SQL statement: exact and typo-tolerant product-ID
  matches (pg_trgm), plus dense (HNSW) and full-text results fused with
  Reciprocal Rank Fusion.
- Reservations lock the affected product rows (``SELECT ... FOR UPDATE`` in
  primary-key order, so concurrent orders cannot deadlock) and are idempotent per
  (email, product): a retried email never reserves the same product twice.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from psycopg2.extras import execute_values

from src.models.domain import (
    STATUS_CREATED,
    STATUS_OUT_OF_STOCK,
    EmailOutcome,
    EmailStatus,
    OrderLine,
    Product,
    ReservationRequest,
)
from src.storage.db import Database
from src.utils.logger import get_logger

logger = get_logger("storage")

_COLUMNS = "product_id, name, category, description, stock, seasons, price"
_WORD = re.compile(r"[a-z0-9]+")
_ID_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,40}")
_RRF_K = 60
_ID_SIMILARITY = 0.4


class DuplicateEmailError(Exception):
    """The email already has a final outcome; it is returned instead of being processed again."""


class UnknownProductError(ValueError):
    pass


def _product(row: Sequence[Any]) -> Product:
    return Product(row[0], row[1], row[2], row[3], int(row[4]), row[5], float(row[6]))


def _vector(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"


def content_hash(product: Product) -> str:
    return hashlib.sha256(product.embedding_text().encode("utf-8")).hexdigest()[:32]


def id_candidates(query: str) -> List[str]:
    """Tokens that could be product IDs: contain both letters and digits ('CBT8901', 'sku-42a')."""
    seen: List[str] = []
    for token in _ID_TOKEN.findall(query):
        if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            canonical = token.upper().replace("-", "").replace("_", "")
            if canonical not in seen:
                seen.append(canonical)
    return seen


def _or_tsquery(text: str, max_terms: int = 32) -> Optional[str]:
    words: List[str] = []
    for word in _WORD.findall(text.lower()):
        if word not in words:
            words.append(word)
    return " | ".join(words[:max_terms]) or None


def decide_reservation(item: ReservationRequest, product: Product) -> Tuple[OrderLine, int]:
    """All-or-nothing per line: reserve the full quantity if stock allows, otherwise leave stock untouched."""
    stock = product.stock
    requested = (stock if stock > 0 else 1) if item.all_available else max(1, int(item.quantity or 1))
    if stock >= requested:
        return OrderLine(product.product_id, product.name, requested, requested, STATUS_CREATED, stock - requested), (
            stock - requested
        )
    return OrderLine(product.product_id, product.name, requested, 0, STATUS_OUT_OF_STOCK, stock), stock


class PostgresStore:
    def __init__(self, db: Database):
        self.db = db

    def ping(self) -> bool:
        try:
            with self.db.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone() == (1,)
        except Exception:  # noqa: BLE001 - readiness probe must never raise
            return False

    # ------------------------------------------------------------------ catalog
    def embedding_fingerprints(self) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        with self.db.cursor() as cur:
            cur.execute("SELECT product_id, embedding_model, embedding_hash FROM products")
            return {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    def upsert_products(
        self, products: Sequence[Product], embeddings: Dict[str, List[float]], model: str, reset_stock: bool
    ) -> None:
        """Inserts/updates products. Stock of existing rows is preserved unless ``reset_stock``."""
        if not products:
            return
        records = []
        for p in products:
            vector = embeddings.get(p.product_id)
            records.append(
                (
                    p.product_id,
                    p.name,
                    p.category,
                    p.description,
                    p.stock,
                    p.seasons,
                    p.price,
                    _vector(vector) if vector is not None else None,
                    model if vector is not None else None,
                    content_hash(p) if vector is not None else None,
                )
            )
        stock = "EXCLUDED.stock" if reset_stock else "products.stock"
        with self.db.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO products (product_id, name, category, description, stock, seasons, price,
                                      embedding, embedding_model, embedding_hash)
                VALUES %s
                ON CONFLICT (product_id) DO UPDATE SET
                    name = EXCLUDED.name, category = EXCLUDED.category, description = EXCLUDED.description,
                    stock = {stock}, seasons = EXCLUDED.seasons, price = EXCLUDED.price,
                    embedding = COALESCE(EXCLUDED.embedding, products.embedding),
                    embedding_model = COALESCE(EXCLUDED.embedding_model, products.embedding_model),
                    embedding_hash = COALESCE(EXCLUDED.embedding_hash, products.embedding_hash)
                """,
                records,
                template="(%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)",
            )

    def get_products(self, product_ids: Sequence[str]) -> Dict[str, Product]:
        ids = [p.strip().upper() for p in product_ids if p and p.strip()]
        if not ids:
            return {}
        with self.db.cursor() as cur:
            cur.execute(f"SELECT {_COLUMNS} FROM products WHERE product_id = ANY(%s)", (ids,))
            return {r[0]: _product(r) for r in cur.fetchall()}

    def search(self, query: str, query_vector: Sequence[float], limit: int) -> List[Tuple[Product, str]]:
        """
        Returns (product, match) pairs, best first. ``match`` is 'id' (exact product ID), 'similar_id'
        (likely typo of a product ID) or 'relevant' (hybrid dense + full-text ranking).
        """
        results: List[Tuple[Product, str]] = []
        seen: set[str] = set()
        candidates = id_candidates(query)
        with self.db.cursor() as cur:
            if candidates:
                cur.execute(f"SELECT {_COLUMNS} FROM products WHERE product_id = ANY(%s)", (candidates,))
                exact = {r[0]: _product(r) for r in cur.fetchall()}
                for token in candidates:
                    if token in exact:
                        results.append((exact[token], "id"))
                        seen.add(token)
                        continue
                    cur.execute(
                        f"""SELECT {_COLUMNS} FROM products
                            WHERE product_id %% %s AND similarity(product_id, %s) >= %s
                            ORDER BY similarity(product_id, %s) DESC, product_id LIMIT 1""",
                        (token, token, _ID_SIMILARITY, token),
                    )
                    row = cur.fetchone()
                    if row and row[0] not in seen:
                        results.append((_product(row), "similar_id"))
                        seen.add(row[0])

            pool = max(limit * 4, 20)
            tsquery = _or_tsquery(query)
            cur.execute(
                f"""
                WITH dense AS (
                    SELECT product_id, row_number() OVER (ORDER BY distance, product_id) AS rank FROM (
                        SELECT product_id, embedding <=> %(vec)s::vector AS distance FROM products
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> %(vec)s::vector LIMIT %(pool)s
                    ) d
                ),
                lexical AS (
                    SELECT product_id, row_number() OVER (ORDER BY score DESC, product_id) AS rank FROM (
                        SELECT product_id, ts_rank_cd(search_tsv, q) AS score
                        FROM products, to_tsquery('english', %(tsq)s) AS q
                        WHERE %(tsq)s <> '' AND search_tsv @@ q
                        ORDER BY score DESC LIMIT %(pool)s
                    ) l
                ),
                fused AS (
                    SELECT product_id, sum(1.0 / (%(k)s + rank)) AS score
                    FROM (SELECT * FROM dense UNION ALL SELECT * FROM lexical) u
                    GROUP BY product_id
                )
                SELECT {", ".join("p." + c.strip() for c in _COLUMNS.split(","))}
                FROM fused JOIN products p USING (product_id)
                WHERE NOT (p.product_id = ANY(%(seen)s))
                ORDER BY fused.score DESC, p.product_id
                LIMIT %(limit)s
                """,
                {
                    "vec": _vector(query_vector),
                    "pool": pool,
                    "tsq": tsquery or "",
                    "k": _RRF_K,
                    "seen": list(seen) or [""],
                    "limit": max(0, limit - len(results)),
                },
            )
            results.extend((_product(r), "relevant") for r in cur.fetchall())
        return results[: max(limit, len(seen))]

    # -------------------------------------------------------- emails & orders
    def claim_email(self, email_id: str) -> None:
        """Marks the email as processing. Raises DuplicateEmailError if it already has a final outcome."""
        with self.db.cursor() as cur:
            cur.execute(
                "INSERT INTO emails (email_id, status) VALUES (%s, 'processing') "
                "ON CONFLICT (email_id) DO NOTHING RETURNING email_id",
                (email_id,),
            )
            if cur.fetchone() is not None:
                return
            cur.execute("SELECT status FROM emails WHERE email_id = %s FOR UPDATE", (email_id,))
            row = cur.fetchone()
            if row and row[0] in {s.value for s in EmailStatus if s.is_final}:
                raise DuplicateEmailError(email_id)
            cur.execute("UPDATE emails SET status = 'processing' WHERE email_id = %s", (email_id,))

    def reserve(self, email_id: str, items: Sequence[ReservationRequest]) -> List[OrderLine]:
        """Atomically reserves stock for ``items``. Lines already reserved for this email are returned as-is."""
        ids = sorted({i.product_id for i in items})
        with self.db.cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM products WHERE product_id = ANY(%s) ORDER BY product_id FOR UPDATE", (ids,)
            )
            locked = {r[0]: _product(r) for r in cur.fetchall()}
            missing = [pid for pid in ids if pid not in locked]
            if missing:
                raise UnknownProductError(f"unknown product IDs: {missing}")
            existing = {line.product_id: line for line in self._lines(cur, email_id)}
            cur.execute("SELECT coalesce(max(line_no), 0) FROM order_lines WHERE email_id = %s", (email_id,))
            next_no = int(cur.fetchone()[0]) + 1  # type: ignore[index]

            result: List[OrderLine] = []
            for item in items:
                if item.product_id in existing:
                    result.append(existing[item.product_id])
                    continue
                product = locked[item.product_id]
                line, new_stock = decide_reservation(item, product)
                if new_stock != product.stock:
                    cur.execute("UPDATE products SET stock = %s WHERE product_id = %s", (new_stock, product.product_id))
                    product.stock = new_stock
                cur.execute(
                    "INSERT INTO order_lines (email_id, line_no, product_id, product_name, requested_quantity, "
                    "fulfilled_quantity, status, remaining_stock) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        email_id,
                        next_no,
                        line.product_id,
                        line.product_name,
                        line.requested_quantity,
                        line.fulfilled_quantity,
                        line.status,
                        line.remaining_stock,
                    ),
                )
                next_no += 1
                existing[line.product_id] = line
                result.append(line)
            return result

    @staticmethod
    def _lines(cur: Any, email_id: str) -> List[OrderLine]:
        cur.execute(
            "SELECT product_id, product_name, requested_quantity, fulfilled_quantity, status, remaining_stock "
            "FROM order_lines WHERE email_id = %s ORDER BY line_no",
            (email_id,),
        )
        return [OrderLine(*row) for row in cur.fetchall()]

    def order_lines(self, email_id: str) -> List[OrderLine]:
        with self.db.cursor() as cur:
            return self._lines(cur, email_id)

    def save_outcome(self, outcome: EmailOutcome) -> None:
        payload = outcome.to_dict()
        payload.pop("lines", None)  # stored in order_lines
        payload.pop("duplicate", None)
        with self.db.cursor() as cur:
            cur.execute(
                """
                INSERT INTO emails (email_id, category, status, outcome, processed_at)
                VALUES (%s, %s, %s, %s::jsonb, now())
                ON CONFLICT (email_id) DO UPDATE SET category = EXCLUDED.category, status = EXCLUDED.status,
                    outcome = EXCLUDED.outcome, processed_at = EXCLUDED.processed_at
                """,
                (outcome.email_id, outcome.category, outcome.status.value, json.dumps(payload, default=str)),
            )

    def get_outcome(self, email_id: str) -> Optional[EmailOutcome]:
        with self.db.cursor() as cur:
            cur.execute("SELECT outcome FROM emails WHERE email_id = %s AND outcome IS NOT NULL", (email_id,))
            row = cur.fetchone()
            if row is None:
                return None
            data = dict(row[0])
            data["lines"] = [vars(line) for line in self._lines(cur, email_id)]
            return EmailOutcome.from_dict(data)

    def reset_operations(self) -> None:
        """Clears processed emails, order lines and the SQL audit log (stock is reset by catalog sync)."""
        with self.db.cursor() as cur:
            cur.execute("TRUNCATE order_lines, emails, sql_audit")

    def close(self) -> None:
        self.db.close()
