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
    PROCESSED = "processed"  # handled end to end
    NEEDS_CLARIFICATION = "needs_clarification"  # order request with no identifiable product
    ESCALATED = "escalated"  # flagged by the security guardrail; no LLM call, no stock reserved


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
class RequestedItem:
    """A catalog product the customer is ordering now, resolved against the catalog."""

    product_id: str
    quantity: Optional[int]  # None when the customer asked for all available stock
    wants_all_available: bool = False


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

    def add(self, other: UsageStats) -> None:
        self.llm_calls += other.llm_calls
        self.replayed_calls += other.replayed_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


@dataclass
class EmailOutcome:
    """Complete, auditable result of processing one customer email."""

    email_id: str
    category: str
    status: EmailStatus
    response: str
    confidence: Optional[float] = None
    language: str = "en"
    customer_name: Optional[str] = None
    lines: List[OrderLine] = field(default_factory=list)
    unresolved_mentions: List[str] = field(default_factory=list)
    context_product_ids: List[str] = field(default_factory=list)
    guardrail_flags: List[str] = field(default_factory=list)
    degraded_reasons: List[str] = field(default_factory=list)
    usage: UsageStats = field(default_factory=UsageStats)
    latency_ms: int = 0
    duplicate: bool = False

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_reasons)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["degraded"] = self.degraded
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmailOutcome:
        payload = {k: v for k, v in data.items() if k != "degraded"}
        payload["status"] = EmailStatus(payload["status"])
        payload["lines"] = [OrderLine(**line) for line in payload.get("lines", [])]
        payload["usage"] = UsageStats(**payload.get("usage", {}))
        return cls(**payload)
