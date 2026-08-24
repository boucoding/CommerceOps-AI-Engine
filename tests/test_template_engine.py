"""
Unit Tests for Deterministic Template Engine & Brand Signature Post-Processing.
"""

from src.pipeline.template_engine import TemplatePostProcessor
from src.models.domain import CustomerEmail

def test_placeholder_scrubbing():
    """Asserts that all raw bracket tags ([Your Name], [Brand Name], etc.) are eliminated."""
    raw_text = """
Dear [Customer's Name],

Thank you for choosing [Brand Name]! Your order has been processed.

Best regards,
[Your Full Name]
[Your Title]
[Brand Name]
[Contact Information]
"""
    email = CustomerEmail(
        email_id="E002",
        subject="Order Confirmation",
        body="My name is Jessica and I want to order."
    )
    
    cleaned = TemplatePostProcessor.process_response(raw_text, email, customer_name="Jessica")
    
    # Assert zero square brackets remaining in production email text
    assert "[" not in cleaned
    assert "]" not in cleaned
    assert "Jessica" in cleaned
    assert "Aura Fashion Atelier" in cleaned
    assert "Aura Concierge" in cleaned

def test_name_extraction_from_conversational_text():
    body = "Hi, this is Liz. Please send me 5 beanies."
    name = TemplatePostProcessor._extract_name_from_text(body)
    assert name == "Liz"
    
    body2 = "Can't wait to get my hands on them. Thanks, Monica"
    name2 = TemplatePostProcessor._extract_name_from_text(body2)
    assert name2 == "Monica"
