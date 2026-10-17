"""
Structured-output schemas for planning and strict function-calling schemas for tools.

All fields are required and unconstrained in the JSON schema (OpenAI strict mode);
ranges and business rules are enforced in code when a tool runs.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

ToolName = Literal["search_catalog", "get_products", "query_database", "reserve_stock", "submit_reply"]


class PlanStep(BaseModel):
    tool: ToolName = Field(description="The tool this step will use.")
    purpose: str = Field(description="What this step must establish or do, in one sentence.")


class Plan(BaseModel):
    intent: Literal["order request", "product inquiry"] = Field(
        description="'order request' only if the customer is ordering now; questions, comparisons and "
        "postponed purchases are 'product inquiry'."
    )
    customer_name: Optional[str] = Field(
        description="The sender's own first name if they state it; never someone else they mention. Null if absent."
    )
    language: str = Field(description="ISO 639-1 code of the email's language, e.g. 'en' or 'es'.")
    summary: str = Field(description="One sentence: what the customer wants.")
    steps: List[PlanStep] = Field(description="Ordered steps; the last one must be submit_reply.")
    success_criteria: str = Field(description="How we know the email has been fully handled.")


class ReplanDecision(BaseModel):
    action: Literal["continue", "revise", "escalate"] = Field(
        description="continue: the failure is recoverable with the current plan; revise: replace the remaining "
        "steps; escalate: a person must handle this email."
    )
    reason: str
    steps: List[PlanStep] = Field(description="The new remaining steps when action is 'revise', otherwise empty.")


# ---------------------------------------------------------------- tool arguments


class SearchCatalogArgs(BaseModel):
    """Hybrid catalog search (product IDs incl. typos, names, and meaning)."""

    query: str = Field(description="What to look for: product IDs, product names or a description of needs.")


class GetProductsArgs(BaseModel):
    """Current details and live stock for specific products."""

    product_ids: List[str] = Field(description="Exact product IDs as returned by search_catalog.")


class QueryDatabaseArgs(BaseModel):
    """Read-only SQL over the analytics views (sandboxed)."""

    sql: str = Field(description="A single PostgreSQL SELECT statement over the documented views.")
    purpose: str = Field(description="Why this query is needed, in one sentence (recorded in the audit log).")


class ReservationItem(BaseModel):
    product_id: str = Field(description="Exact product ID previously returned by a tool.")
    quantity: Optional[int] = Field(description="Units ordered; null only when all_available is true.")
    all_available: bool = Field(description="True if the customer asked for all remaining stock.")


class ReserveStockArgs(BaseModel):
    """Reserve stock for the items the customer is ordering now."""

    items: List[ReservationItem]


class SubmitReplyArgs(BaseModel):
    """Final step: the reply email to send."""

    reply: str = Field(
        description="The complete reply including greeting and closing line, without signature (it is added "
        "automatically)."
    )
