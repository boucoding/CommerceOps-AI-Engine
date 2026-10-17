"""
PostgreSQL + pgvector integration tests. Run with:

    TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/commerceops_test pytest -m integration

The target database is wiped (all tables dropped) at the start of the session.
"""

from __future__ import annotations

import threading
from typing import Iterator, List

import pytest

from src.config import LLMMode, StorageBackend
from src.engine import Engine, build_engine
from src.models.domain import STATUS_CREATED, STATUS_OUT_OF_STOCK, EmailStatus, Product, RequestedItem
from src.storage.base import DuplicateEmailError
from src.storage.db import Database
from src.storage.migrate import apply_migrations, migration_files
from src.storage.postgres import PostgresStore
from tests.conftest import email, make_settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def pg_settings(database_url):
    settings = make_settings(
        STORAGE_BACKEND=StorageBackend.POSTGRES, DATABASE_URL=database_url, DB_CONNECT_RETRIES=1, DB_POOL_MAX=30
    )
    db = Database(settings)
    with db.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS order_lines, emails, products, lexical_stats, schema_migrations CASCADE")
    db.close()
    return settings


@pytest.fixture
def pg_engine(pg_settings, products: List[Product]) -> Iterator[Engine]:
    engine = build_engine(pg_settings, mode=LLMMode.OFFLINE, products=products, reset=True)
    yield engine
    engine.close()


def test_migrations_are_idempotent_and_recorded(pg_settings):
    db = Database(pg_settings)
    try:
        apply_migrations(db)
        assert apply_migrations(db) == []
        with db.cursor() as cur:
            cur.execute("SELECT version FROM schema_migrations ORDER BY version")
            assert [r[0] for r in cur.fetchall()] == [p.stem for p in migration_files()]
            cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'products'")
            indexes = {r[0] for r in cur.fetchall()}
        assert {"products_embedding_hnsw_idx", "products_id_trgm_idx", "products_search_tsv_idx"} <= indexes
    finally:
        db.close()


def test_catalog_has_embeddings_and_preserves_stock_on_resync(pg_engine, pg_settings, products):
    store = pg_engine.store
    assert all(model == "local:hash-v1" and digest for model, digest in store.embedding_fingerprints().values())
    store.commit_order("S1", [RequestedItem("SWL2345", 2)])
    resynced = build_engine(pg_settings, mode=LLMMode.OFFLINE, products=products)  # no reset
    try:
        assert resynced.store.get_products(["SWL2345"])["SWL2345"].stock == 3
    finally:
        resynced.close()


def test_hnsw_query_plan_uses_vector_index(pg_engine):
    store = pg_engine.store
    assert isinstance(store, PostgresStore)
    vector = "[" + ",".join(["0.01"] * 1536) + "]"
    with store.db.cursor() as cur:
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute("EXPLAIN SELECT product_id FROM products ORDER BY embedding <=> %s::vector LIMIT 5", (vector,))
        plan = "\n".join(r[0] for r in cur.fetchall())
    assert "products_embedding_hnsw_idx" in plan


def test_retrieval_matches_the_in_memory_backend(pg_engine, build):
    memory = build()
    for query in (
        "Do you have info on DHN0987?",
        "I need a bag for my laptop and documents for work",
        "I'd like to order three to four Versatile Scarves",
        "retro sun glasses (RSG 89 01)",
    ):
        pg = pg_engine.orchestrator.retriever.retrieve(query, 4)
        mem = memory.orchestrator.retriever.retrieve(query, 4)
        assert pg.product_ids == mem.product_ids, query


def test_reservation_rules_and_resume(pg_engine):
    store = pg_engine.store
    lines = store.commit_order("R1", [RequestedItem("RSG8901", 1), RequestedItem("CLF2109", 5)])
    assert [(ln.status, ln.remaining_stock) for ln in lines] == [(STATUS_CREATED, 0), (STATUS_OUT_OF_STOCK, 2)]
    assert store.commit_order("R1", [RequestedItem("RSG8901", 1)]) == lines  # resume: nothing reserved again
    assert store.get_products(["RSG8901"])["RSG8901"].stock == 0


def test_concurrent_orders_never_oversell(pg_engine):
    store = pg_engine.store
    statuses: List[str] = []
    lock = threading.Lock()

    def order(i: int) -> None:
        line = store.commit_order(f"C{i}", [RequestedItem("LTH1098", 1), RequestedItem("SWL2345", 1)])
        with lock:
            statuses.extend(ln.status for ln in line if ln.product_id == "LTH1098")

    threads = [threading.Thread(target=order, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert statuses.count(STATUS_CREATED) == 7 and statuses.count(STATUS_OUT_OF_STOCK) == 13
    stock = store.get_products(["LTH1098", "SWL2345"])
    assert stock["LTH1098"].stock == 0 and stock["SWL2345"].stock == 0


def test_end_to_end_outcome_is_persisted_and_idempotent(pg_engine):
    orchestrator = pg_engine.orchestrator
    first = orchestrator.process_email(email("P1", "Hi, this is Liz. Please send me 2 SWL2345."))
    assert first.status == EmailStatus.PROCESSED and first.lines[0].status == STATUS_CREATED

    stored = pg_engine.store.get_outcome("P1")
    assert stored is not None and stored.response == first.response and stored.lines == first.lines
    again = orchestrator.process_email(email("P1", "Hi, this is Liz. Please send me 2 SWL2345."))
    assert again.duplicate and pg_engine.store.get_products(["SWL2345"])["SWL2345"].stock == 3
    with pytest.raises(DuplicateEmailError):
        pg_engine.store.claim_email("P1")
