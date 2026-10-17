"""
Core domain models for CommerceOps AI Engine.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional

ORDER_REQUEST = "order request"
PRODUCT_INQUIRY = "product inquiry"

STATUS_CREATED = "created"
STATUS_OUT_OF_STOCK = "out of stock"


class EmailStatus(StrEnum):
    PROCESSED = "processed"  # reply sent; any order lines reserved
    NEEDS_CLARIFICATION = "needs_clarification"  # order request where no product could be identified; reply asks
    ESCALATED = "escalated"  # blocked by the input guardrail; never shown to the model
    NEEDS_REVIEW = "needs_review"  # the agent gave up or ran out of budget; a person must finish it
    FAILED = "failed"  # infrastructure error (e.g. model unavailable); safe to retry

    @property
    def is_final(self) -> bool:
        return self is not EmailStatus.FAILED


@dataclass
class Product:
    product_id: str
    name: str
    category: str
    description: str
    stock: int
    seasons: str
    price: float

    def embedding_text(self) -> str:
        """Text used for semantic embeddings. Excludes stock and price so restocks don't force re-embedding."""
        return f"{self.name}. Category: {self.category}. Seasons: {self.seasons}. {self.description}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CustomerEmail:
    email_id: str
    subject: str
    body: str


@dataclass
class ReservationRequest:
    product_id: str
    quantity: Optional[int]  # None when the customer asked for all available stock
    all_available: bool = False


@dataclass
class OrderLine:
    product_id: str
    product_name: str
    requested_quantity: int
    fulfilled_quantity: int
    status: str  # STATUS_CREATED | STATUS_OUT_OF_STOCK
    remaining_stock: int


@dataclass
class UsageStats:
    llm_calls: int = 0
    replayed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record_call(self, prompt_tokens: int, completion_tokens: int, replayed: bool) -> None:
        self.llm_calls += 1
        self.replayed_calls += int(replayed)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens


@dataclass
class EmailOutcome:
    """Complete, auditable result of processing one customer email."""

    email_id: str
    status: EmailStatus
    category: Optional[str] = None
    response: Optional[str] = None
    language: Optional[str] = None
    customer_name: Optional[str] = None
    lines: List[OrderLine] = field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    trace: List[Dict[str, Any]] = field(default_factory=list)
    context_product_ids: List[str] = field(default_factory=list)
    guardrail_flags: List[str] = field(default_factory=list)
    error: Optional[str] = None
    usage: UsageStats = field(default_factory=UsageStats)
    latency_ms: int = 0
    duplicate: bool = False

    @property
    def tool_calls(self) -> List[str]:
        return [e["tool"] for e in self.trace if e.get("type") == "tool"]

    @property
    def replans(self) -> int:
        return sum(1 for e in self.trace if e.get("type") == "replan")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmailOutcome:
        payload = dict(data)
        payload["status"] = EmailStatus(payload["status"])
        payload["lines"] = [OrderLine(**line) for line in payload.get("lines", [])]
        payload["usage"] = UsageStats(**payload.get("usage", {}))
        return cls(**payload)
