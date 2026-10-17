"""
HTTP request/response models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.domain import EmailOutcome, Product


class EmailIn(BaseModel):
    email_id: Optional[str] = Field(
        default=None,
        pattern=r"^[A-Za-z0-9._:-]{1,64}$",
        description="Idempotency key. Re-sending the same id returns the stored outcome. Generated if omitted.",
    )
    subject: str = Field(default="", max_length=1_000)
    body: str = Field(min_length=1, max_length=20_000)


class BatchIn(BaseModel):
    emails: List[EmailIn] = Field(min_length=1)


class OrderLineOut(BaseModel):
    product_id: str
    product_name: str
    requested_quantity: int
    fulfilled_quantity: int
    status: str
    remaining_stock: int


class UsageOut(BaseModel):
    llm_calls: int
    replayed_calls: int
    prompt_tokens: int
    completion_tokens: int


class OutcomeOut(BaseModel):
    email_id: str
    status: str
    category: Optional[str]
    response: Optional[str]
    language: Optional[str]
    customer_name: Optional[str]
    lines: List[OrderLineOut]
    plan: Optional[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    context_product_ids: List[str]
    guardrail_flags: List[str]
    error: Optional[str]
    usage: UsageOut
    latency_ms: int
    duplicate: bool

    @classmethod
    def from_outcome(cls, outcome: EmailOutcome) -> OutcomeOut:
        return cls.model_validate(outcome.to_dict())


class BatchOut(BaseModel):
    outcomes: List[OutcomeOut]
    summary: Dict[str, Any]


class ProductOut(BaseModel):
    product_id: str
    name: str
    category: str
    description: str
    stock: int
    seasons: str
    price: float

    @classmethod
    def from_product(cls, product: Product) -> ProductOut:
        return cls.model_validate(product.to_dict())


class SearchHit(ProductOut):
    match: str


class SearchOut(BaseModel):
    results: List[SearchHit]
