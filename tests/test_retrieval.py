import pytest

from src.retrieval.bm25 import BM25Index
from src.retrieval.fuzzy_matcher import (
    best_sku_match,
    extract_sku_tokens,
    find_name_mentions,
    normalize_sku_text,
)
from src.retrieval.retriever import reciprocal_rank_fusion
from src.retrieval.text import normalize_phrase


def test_normalize_spaced_and_bracketed_skus():
    assert normalize_sku_text("buy [CBT 89 01] and rsg-8901") == "buy CBT8901 and RSG8901"


def test_extract_sku_tokens_dedupes_in_order():
    assert extract_sku_tokens("CBT8901, DHN0987 and CBT 89 01 again") == ["CBT8901", "DHN0987"]


def test_sku_typo_recovery_and_threshold():
    ids = ["CHN0987", "CLF2109", "RSG8901"]
    match = best_sku_match("DHN0987", ids, 80)
    assert match is not None and match[0] == "CHN0987" and match[1] == pytest.approx(85.714, abs=0.01)
    assert best_sku_match("RSG8901", ids, 80) == ("RSG8901", 100.0)
    assert best_sku_match("XYZ1111", ids, 80) is None


def test_name_mentions_handle_plurals_and_spacing(products):
    found = {m.product.product_id for m in find_name_mentions("2 pairs of the retro sun glasses please", products, 88)}
    assert found == {"RSG8901"}
    found = {m.product.product_id for m in find_name_mentions("one of your Versatile Scarves", products, 88)}
    assert found == {"VSC6789"}


def test_unrelated_text_does_not_match_names(products):
    assert find_name_mentions("I love my straw tote from the beach", products, 88) == []


def test_normalize_phrase_stems_plurals():
    assert normalize_phrase("Cable Knit Beanies") == normalize_phrase("cable knit beanie")
    assert normalize_phrase("scarves") == "scarf"


def test_bm25_ranks_relevant_documents_first():
    index = BM25Index(["a", "b", "c"], ["leather laptop backpack", "silk scarf", "leather wallet"])
    assert [doc for doc, _ in index.search("laptop bag leather", 3)] == ["a", "c"]


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert fused == ["b", "c", "a", "d"]


def test_retriever_splits_mentions_from_related(build):
    engine = build()
    result = engine.orchestrator.retriever.retrieve("Do you have info on DHN0987? I need a laptop bag too.", 3)
    assert [p.product_id for p in result.mentioned] == ["CHN0987"]
    assert result.sku_resolutions[0].token == "DHN0987"
    assert "CHN0987" not in [p.product_id for p in result.related]
    assert {"LTH1098"} <= {p.product_id for p in result.related}
    assert len(result.related) == 3


def test_alternatives_are_in_stock_and_exclude_the_product(build):
    engine = build()
    beanie = engine.store.get_products(["CLF2109"])["CLF2109"]
    alts = engine.orchestrator.retriever.alternatives(beanie, 2)
    assert alts and all(p.stock > 0 and p.product_id != "CLF2109" for p in alts)
    assert alts[0].category == "Accessories"
