"""
Pydantic Schemas for Strict Structured Outputs via OpenAI API.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    """Structured response for email intent categorization."""
    category: Literal["product inquiry", "order request"] = Field(
        description="Must be strictly 'product inquiry' or 'order request'."
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    customer_name: Optional[str] = Field(
        default=None,
        description="Customer's first name if mentioned in the email."
    )
    language: str = Field(
        default="en",
        description="Detected language code (e.g. 'en', 'es', 'fr')."
    )
    reasoning: str = Field(
        description="Concise rationale for classification, noting edge cases (e.g. postponed purchase, noise)."
    )

class ExtractedOrderItem(BaseModel):
    """Represents a single product line extracted from an order email."""
    product_id: Optional[str] = Field(
        default=None,
        description="Canonical uppercase alphanumeric product ID matched from candidate list (e.g. 'SWL2345'). Null if unknown."
    )
    product_name: Optional[str] = Field(
        default=None,
        description="Product name as mentioned in the email."
    )
    quantity: int = Field(
        default=1,
        description="Number of units requested. If 'all remaining in stock', set to -1. If vague range like '3 to 4', pick upper bound (4)."
    )
    is_active_order: bool = Field(
        default=True,
        description="True if the customer is ordering this item NOW. False if mentioning a past purchase or future intent."
    )

class ExtractedOrder(BaseModel):
    """Structured extraction of all items from an order email."""
    items: List[ExtractedOrderItem] = Field(
        default_factory=list,
        description="List of product items actively being ordered in this email."
    )
