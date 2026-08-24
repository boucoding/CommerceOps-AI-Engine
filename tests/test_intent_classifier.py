"""
Unit Tests for Intent Classifier handling operational edge cases.
"""

from unittest.mock import MagicMock
from src.pipeline.intent_classifier import IntentClassifier
from src.models.domain import CustomerEmail
from src.models.llm_schemas import IntentClassification

def test_postponed_purchase_classification():
    """Verifies that postponed purchases (e.g., E006: 'wait until Fall') are product inquiry."""
    mock_client = MagicMock()
    classifier = IntentClassifier(mock_client)
    
    email = CustomerEmail(
        email_id="E006",
        subject="Boots question",
        body="Hey there, I was thinking of ordering a pair of CBT8901 Chelsea Boots, but I'll wait until Fall to actually place the order. My name is Sam."
    )
    
    result = classifier.classify(email)
    assert result.category == "product inquiry"

def test_immediate_order_classification():
    """Verifies that immediate purchase commitment is categorized as order request."""
    mock_client = MagicMock()
    classifier = IntentClassifier(mock_client)
    
    email = CustomerEmail(
        email_id="E001",
        subject="Leather Wallets",
        body="Hi there, I want to order all the remaining LTH0976 Leather Bifold Wallets you have in stock."
    )
    
    result = classifier.classify(email)
    assert result.category == "order request"

def test_rambling_noise_classification():
    """Verifies that rambling emails (lawnmower, dinner) are classified correctly."""
    mock_client = MagicMock()
    classifier = IntentClassifier(mock_client)
    
    email = CustomerEmail(
        email_id="E012",
        subject="Rambling About a New Work Bag",
        body="Speaking of broken things, I also need to get my lawnmower fixed before spring. What were some of the other messenger bag options you have?"
    )
    
    result = classifier.classify(email)
    assert result.category == "product inquiry"
