"""
Pydantic schemas for OpenAI Structured Outputs.

Every field is required (no defaults, no numeric constraints) so the schemas are
valid under strict JSON-schema mode; values are range-checked in code instead.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class IntentClassification(BaseModel):
    category: Literal["product inquiry", "order request"] = Field(
        description="'order request' only when the customer is placing an order now; otherwise 'product inquiry'."
    )
    confidence: float = Field(description="Confidence between 0.0 and 1.0.")
    customer_name: Optional[str] = Field(
        description="The sender's own first name if they state it (greeting or signature). "
        "Never a name of someone else they mention. Null if not stated."
    )
    language: str = Field(description="ISO 639-1 code of the email's language, e.g. 'en', 'es'.")
    reasoning: str = Field(description="One sentence explaining the decision.")


class ExtractedOrderItem(BaseModel):
    product_id: Optional[str] = Field(
        description="Product ID from the candidate list that matches the item. Null if no candidate matches."
    )
    product_name: Optional[str] = Field(description="The item as the customer wrote it.")
    quantity: Optional[int] = Field(
        description="Units requested. Upper bound for ranges ('three to four' -> 4). 1 when unspecified. "
        "Null only when the customer asks for all available stock."
    )
    wants_all_available: bool = Field(description="True when the customer asks for all remaining stock.")
    is_active_order: bool = Field(
        description="True only if the customer is ordering this item now (not a past purchase or a future plan)."
    )


class ExtractedOrder(BaseModel):
    items: List[ExtractedOrderItem]
