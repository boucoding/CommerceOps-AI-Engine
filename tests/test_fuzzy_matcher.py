"""
Unit Tests for Fuzzy Product Matching, Typo Recovery, and SKU Normalization.
"""

from src.retrieval.fuzzy_matcher import FuzzyProductMatcher
from src.models.domain import Product

def test_bracketed_sku_normalization(sample_products):
    matcher = FuzzyProductMatcher(sample_products)
    
    # Test normalization of spaced/bracketed SKU codes
    raw_text = "I would like to order [CBT 89 01] and RSG 89 01 please."
    normalized = matcher.normalize_sku_in_text(raw_text)
    
    assert "CBT8901" in normalized
    assert "RSG8901" in normalized

def test_sku_token_extraction(sample_products):
    matcher = FuzzyProductMatcher(sample_products)
    text = "Please check availability for CBT8901 and DHN0987."
    tokens = matcher.extract_potential_sku_tokens(text)
    
    assert "CBT8901" in tokens
    assert "DHN0987" in tokens

def test_typo_sku_recovery(sample_products):
    """Verifies that 1-character typos (e.g. DHN0987 -> CHN0987) resolve correctly."""
    matcher = FuzzyProductMatcher(sample_products)
    
    matched = matcher.match_sku_with_typo_tolerance("DHN0987", score_threshold=70.0)
    assert matched is not None
    product, score = matched
    assert product.product_id == "CHN0987"
    assert product.name == "Chunky Knit Beanie"
    assert score > 80.0

def test_exact_sku_matching(sample_products):
    matcher = FuzzyProductMatcher(sample_products)
    matched = matcher.match_sku_with_typo_tolerance("RSG8901")
    
    assert matched is not None
    product, score = matched
    assert product.product_id == "RSG8901"
    assert score == 100.0

def test_product_name_matching(sample_products):
    matcher = FuzzyProductMatcher(sample_products)
    text = "I would like to buy a Sleek Wallet today."
    matches = matcher.match_product_name(text)
    
    assert len(matches) > 0
    top_prod, score = matches[0]
    assert top_prod.product_id == "SWL2345"
    assert top_prod.name == "Sleek Wallet"
    assert score == 100.0
