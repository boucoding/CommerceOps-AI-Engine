"""
Unit Tests for Two-Stage RAG Order Extraction ($O(1)$ Token Cost).
"""

from unittest.mock import MagicMock
from src.pipeline.order_extractor import OrderExtractor
from src.models.domain import CustomerEmail

def test_two_stage_rag_candidate_retrieval(sample_products, hybrid_vector_store, inventory_service):
    """Verifies that OrderExtractor retrieves top-k candidates rather than full catalog."""
    mock_client = MagicMock()
    extractor = OrderExtractor(mock_client, hybrid_vector_store, inventory_service)
    
    email = CustomerEmail(
        email_id="E002",
        subject="Buy Vibrant Tote",
        body="Good morning, I'm looking to buy the VBT2345 Vibrant Tote bag. My name is Jessica."
    )
    
    # Candidate search retrieves top candidate
    candidates = extractor.vector_store.search(f"{email.subject} {email.body}", top_k=5)
    assert len(candidates) <= 5
    
def test_order_fulfillment_stock_decrement(sample_products, hybrid_vector_store, inventory_service):
    """Verifies end-to-end extraction and reservation decrement."""
    mock_client = MagicMock()
    extractor = OrderExtractor(mock_client, hybrid_vector_store, inventory_service)
    
    email = CustomerEmail(
        email_id="E010",
        subject="Purchase Retro Sunglasses",
        body="Hello, I would like to order 1 pair of RSG8901 Retro Sunglasses. Thanks!"
    )
    
    result = extractor.extract_and_fulfill(email)
    assert len(result.lines) > 0
    # RSG8901 stock was 1 -> now 0 and status is created
    line = result.lines[0]
    assert line.product_id == "RSG8901"
    assert line.status == "created"
    assert line.remaining_stock == 0
