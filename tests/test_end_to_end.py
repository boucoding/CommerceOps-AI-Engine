"""
End-to-End Integration and Batch Pipeline Tests for CommerceOps AI Engine.
"""

from unittest.mock import MagicMock
from src.pipeline.orchestrator import CommerceOpsOrchestrator
from src.models.domain import CustomerEmail

def test_full_pipeline_batch_execution(sample_products):
    """Executes orchestrator across sample customer emails and validates output DataFrame schemas."""
    mock_client = MagicMock()
    orchestrator = CommerceOpsOrchestrator(sample_products, mock_client)
    
    test_emails = [
        CustomerEmail(
            email_id="E001",
            subject="Order Request",
            body="Hi, I want to order 1 Sleek Wallet (SWL2345)."
        ),
        CustomerEmail(
            email_id="E002",
            subject="Product Inquiry",
            body="Hello, what material is the Chunky Knit Beanie (CHN0987) made of?"
        ),
        CustomerEmail(
            email_id="E003",
            subject="Out of Stock Order",
            body="I would like to order 5 Cargo Pants (CGN2345)."
        )
    ]
    
    df_class, df_status, df_order_resp, df_inquiry_resp = orchestrator.process_batch(test_emails)
    
    # 1. Validate email-classification schema & content
    assert list(df_class.columns) == ["email ID", "category"]
    assert len(df_class) == 3
    
    # 2. Validate order-status schema & content
    assert list(df_status.columns) == ["email ID", "product ID", "quantity", "status"]
    assert len(df_status) >= 2  # E001 and E003 are order requests
    
    # 3. Validate order-response schema
    assert list(df_order_resp.columns) == ["email ID", "response"]
    assert len(df_order_resp) == 2
    
    # 4. Validate inquiry-response schema
    assert list(df_inquiry_resp.columns) == ["email ID", "response"]
    assert len(df_inquiry_resp) == 1
    
    # Check that responses have zero leaked placeholder brackets
    for resp in df_order_resp["response"]:
        assert "[" not in resp and "]" not in resp
    for resp in df_inquiry_resp["response"]:
        assert "[" not in resp and "]" not in resp
