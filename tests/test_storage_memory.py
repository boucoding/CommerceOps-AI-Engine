import threading

import pytest

from src.catalog import sync_catalog
from src.models.domain import STATUS_CREATED, STATUS_OUT_OF_STOCK, EmailOutcome, EmailStatus, RequestedItem
from src.retrieval.embeddings import EmbeddingService
from src.storage.base import DuplicateEmailError
from src.storage.memory import MemoryStore


@pytest.fixture
def store(products):
    s = MemoryStore()
    sync_catalog(s, products, EmbeddingService(None, 64), reset_stock=True)
    return s


def _outcome(email_id):
    return EmailOutcome(email_id, "order request", EmailStatus.PROCESSED, "reply")


def test_reservation_decrements_stock(store):
    (line,) = store.commit_order("E1", [RequestedItem("SWL2345", 2)])
    assert (line.status, line.fulfilled_quantity, line.remaining_stock) == (STATUS_CREATED, 2, 3)
    assert store.get_products(["SWL2345"])["SWL2345"].stock == 3


def test_insufficient_stock_is_all_or_nothing(store):
    (line,) = store.commit_order("E1", [RequestedItem("RSG8901", 3)])
    assert (line.status, line.fulfilled_quantity, line.remaining_stock) == (STATUS_OUT_OF_STOCK, 0, 1)
    assert store.get_products(["RSG8901"])["RSG8901"].stock == 1


def test_all_available_takes_remaining_stock(store):
    (line,) = store.commit_order("E1", [RequestedItem("SWL2345", None, wants_all_available=True)])
    assert (line.requested_quantity, line.status, line.remaining_stock) == (5, STATUS_CREATED, 0)
    (line,) = store.commit_order("E2", [RequestedItem("SWL2345", None, wants_all_available=True)])
    assert (line.requested_quantity, line.status) == (1, STATUS_OUT_OF_STOCK)


def test_sequential_orders_deplete_stock(store):
    assert store.commit_order("E1", [RequestedItem("RSG8901", 1)])[0].status == STATUS_CREATED
    assert store.commit_order("E2", [RequestedItem("RSG8901", 1)])[0].status == STATUS_OUT_OF_STOCK


def test_resume_does_not_reserve_twice_and_saved_outcome_is_final(store):
    first = store.commit_order("E1", [RequestedItem("SWL2345", 2)])
    resumed = store.commit_order("E1", [RequestedItem("SWL2345", 2)])
    assert resumed == first
    assert store.get_products(["SWL2345"])["SWL2345"].stock == 3
    store.save_outcome(_outcome("E1"))
    with pytest.raises(DuplicateEmailError):
        store.commit_order("E1", [RequestedItem("SWL2345", 2)])
    with pytest.raises(DuplicateEmailError):
        store.claim_email("E1")


def test_concurrent_reservations_never_oversell(store):
    results = []
    lock = threading.Lock()

    def order(i):
        lines = store.commit_order(f"E{i}", [RequestedItem("LTH1098", 1)])
        with lock:
            results.append(lines[0].status)

    threads = [threading.Thread(target=order, args=(i,)) for i in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(STATUS_CREATED) == 7
    assert results.count(STATUS_OUT_OF_STOCK) == 18
    assert store.get_products(["LTH1098"])["LTH1098"].stock == 0


def test_resync_preserves_stock_unless_reset(store, products):
    store.commit_order("E1", [RequestedItem("SWL2345", 2)])
    sync_catalog(store, products, EmbeddingService(None, 64), reset_stock=False)
    assert store.get_products(["SWL2345"])["SWL2345"].stock == 3
    sync_catalog(store, products, EmbeddingService(None, 64), reset_stock=True)
    assert store.get_products(["SWL2345"])["SWL2345"].stock == 5


def test_sync_only_reembeds_changed_products(products):
    store = MemoryStore()
    embedder = EmbeddingService(None, 64)
    assert sync_catalog(store, products, embedder, reset_stock=True) == len(products)
    assert sync_catalog(store, products, embedder, reset_stock=False) == 0
    products[0].description += " Now polarized."
    assert sync_catalog(store, products, embedder, reset_stock=False) == 1
