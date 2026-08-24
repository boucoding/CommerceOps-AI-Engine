"""
Unit Tests for Hybrid Vector Store (PgVector / Dense Cosine & Fuzzy Scoring).
"""

from src.retrieval.hybrid_vector_store import HybridVectorStore

def test_semantic_retrieval_laptop_bag(hybrid_vector_store):
    """Tests semantic search for laptop work bag."""
    query = "I need a bag for work that has a padded laptop sleeve and pockets."
    results = hybrid_vector_store.search(query, top_k=2)
    
    assert len(results) > 0
    top_ids = [p.product_id for p in results]
    # LTH1098 Leather Backpack or LTH5432 Leather Tote should be in top results
    assert "LTH1098" in top_ids or "LTH5432" in top_ids

def test_fuzzy_sku_retrieval_priority(hybrid_vector_store):
    """Tests that a query with a typo SKU prioritizes that corrected product."""
    query = "Do you have any info on DHN0987?"
    results = hybrid_vector_store.search(query, top_k=2)
    
    assert len(results) > 0
    assert results[0].product_id == "CHN0987"
    assert results[0].name == "Chunky Knit Beanie"

def test_retrieval_limit_top_k(hybrid_vector_store):
    """Tests that top_k is strictly respected."""
    results = hybrid_vector_store.search("leather accessories", top_k=3)
    assert len(results) <= 3
