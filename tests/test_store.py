"""
PostgreSQL store: migrations, hybrid search, reservations and the audit trail.
"""

import threading

from src.models.domain import STATUS_CREATED, STATUS_OUT_OF_STOCK, EmailOutcome, EmailStatus, ReservationRequest
from src.storage.migrate import apply_migrations, migration_files
from src.storage.postgres import DuplicateEmailError, id_candidates
from tests.conftest import hash_embedding


def test_id_candidates_are_generic():
    assert id_candidates("order CBT8901, sku-42a and the 1950s look") == ["CBT8901", "SKU42A", "1950S"]
    assert id_candidates("a nice warm hat") == []


def test_migrations_are_idempotent_and_create_indexes(make_engine):
    store = make_engine().store
    assert apply_migrations(store.db) == []
    with store.db.cursor() as cur:
        cur.execute("SELECT version FROM schema_migrations ORDER BY version")
        assert [r[0] for r in cur.fetchall()] == [p.stem for p in migration_files()]
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute(
            "EXPLAIN SELECT product_id FROM products ORDER BY embedding <=> %s::vector LIMIT 5",
            ("[" + ",".join(["0.01"] * 1536) + "]",),
        )
        assert "products_embedding_hnsw_idx" in "\n".join(r[0] for r in cur.fetchall())


def _search(store, query, limit=4):
    return [(p.product_id, match) for p, match in store.search(query, hash_embedding(query), limit)]


def test_search_exact_typo_and_semantic(make_engine):
    store = make_engine().store
    assert _search(store, "CBT8901")[0] == ("CBT8901", "id")
    assert _search(store, "do you have DHN0987?")[0] == ("CHN0987", "similar_id")
    hits = _search(store, "leather bag with a laptop sleeve")
    assert hits[0][0] == "LTH1098" and all(m == "relevant" for _, m in hits)
    assert len(hits) == 4 and len({pid for pid, _ in hits}) == 4


def test_reservations_are_atomic_all_or_nothing_and_idempotent(make_engine):
    store = make_engine().store
    store.claim_email("R1")
    lines = store.reserve("R1", [ReservationRequest("RSG8901", 1), ReservationRequest("CLF2109", 5)])
    assert [(ln.status, ln.remaining_stock) for ln in lines] == [(STATUS_CREATED, 0), (STATUS_OUT_OF_STOCK, 2)]
    assert store.reserve("R1", [ReservationRequest("RSG8901", 1)]) == lines[:1]  # retry: no second reservation
    assert store.get_products(["RSG8901"])["RSG8901"].stock == 0


def test_concurrent_orders_never_oversell(make_engine):
    store = make_engine().store
    statuses = []
    lock = threading.Lock()

    def order(i):
        store.claim_email(f"C{i}")
        lines = store.reserve(f"C{i}", [ReservationRequest("LTH1098", 1), ReservationRequest("SWL2345", 1)])
        with lock:
            statuses.append(lines[0].status)

    threads = [threading.Thread(target=order, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert statuses.count(STATUS_CREATED) == 7 and statuses.count(STATUS_OUT_OF_STOCK) == 13
    stock = store.get_products(["LTH1098", "SWL2345"])
    assert stock["LTH1098"].stock == 0 and stock["SWL2345"].stock == 0


def test_claim_allows_retry_after_failure_but_not_after_final_outcome(make_engine):
    store = make_engine().store
    store.claim_email("F1")
    store.save_outcome(EmailOutcome("F1", EmailStatus.FAILED, error="model down"))
    store.claim_email("F1")  # failed is retryable
    store.save_outcome(EmailOutcome("F1", EmailStatus.PROCESSED, response="ok"))
    try:
        store.claim_email("F1")
        raise AssertionError("expected DuplicateEmailError")
    except DuplicateEmailError:
        pass
    assert store.get_outcome("F1").response == "ok"
