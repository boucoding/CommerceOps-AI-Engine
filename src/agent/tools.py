"""
Agent tools: strict JSON schemas for the model, validated arguments, permission
gates and bounded outputs.

Every tool returns a JSON-serializable dict. Failures are returned to the model as
``{"error": ...}`` results (``ok=False``) so it can correct itself; the executor
uses them as the trigger for replanning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import openai
from pydantic import BaseModel, ValidationError

from src.agent.schemas import (
    GetProductsArgs,
    QueryDatabaseArgs,
    ReserveStockArgs,
    SearchCatalogArgs,
    SubmitReplyArgs,
)
from src.config import Settings
from src.models.domain import ORDER_REQUEST, OrderLine, Product, ReservationRequest
from src.safety.reply_validation import ReplyContext, ReplyValidationError, finalize_reply
from src.sandbox.sql import SQLRejected, SQLSandbox
from src.storage.postgres import PostgresStore, UnknownProductError

_DESCRIPTION_CHARS = 300


class ToolError(Exception):
    """A recoverable problem reported back to the model."""


@dataclass
class RunContext:
    """Per-email state shared by the tools during one agent run."""

    email_id: str
    intent: str
    token_map: Dict[str, str]
    observed_products: Dict[str, Product] = field(default_factory=dict)
    lines: Dict[str, OrderLine] = field(default_factory=dict)
    reply: Optional[str] = None

    def observe(self, products: List[Product]) -> None:
        for p in products:
            self.observed_products[p.product_id] = p


@dataclass
class ToolResult:
    ok: bool
    content: Dict[str, Any]

    def to_message(self, call_id: str) -> Dict[str, Any]:
        return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(self.content, ensure_ascii=False)}


@dataclass
class Tool:
    name: str
    description: str
    args_model: Type[BaseModel]
    handler: Callable[[Any, RunContext], Dict[str, Any]]

    def schema(self) -> Dict[str, Any]:
        return dict(openai.pydantic_function_tool(self.args_model, name=self.name, description=self.description))


def _product_view(p: Product, match: Optional[str] = None) -> Dict[str, Any]:
    view: Dict[str, Any] = {
        "product_id": p.product_id,
        "name": p.name,
        "category": p.category,
        "price": round(p.price, 2),
        "seasons": p.seasons,
        "stock": p.stock,
        "description": p.description[:_DESCRIPTION_CHARS],
    }
    if match:
        view["match"] = match
    return view


class Toolbox:
    def __init__(
        self, store: PostgresStore, sandbox: SQLSandbox, embed: Callable[[str], List[float]], settings: Settings
    ):
        self.store = store
        self.sandbox = sandbox
        self.embed = embed
        self.settings = settings
        self.tools: Dict[str, Tool] = {
            t.name: t
            for t in (
                Tool(
                    "search_catalog",
                    "Search the product catalog by product ID (typos tolerated), name or need.",
                    SearchCatalogArgs,
                    self._search_catalog,
                ),
                Tool(
                    "get_products",
                    "Get current details and live stock for specific product IDs.",
                    GetProductsArgs,
                    self._get_products,
                ),
                Tool(
                    "query_database",
                    "Run one read-only SELECT over the product_catalog and order_history views.",
                    QueryDatabaseArgs,
                    self._query_database,
                ),
                Tool(
                    "reserve_stock",
                    "Reserve stock for the items the customer is ordering now (all-or-nothing "
                    "per line). Only for order requests.",
                    ReserveStockArgs,
                    self._reserve_stock,
                ),
                Tool(
                    "submit_reply",
                    "Submit the final reply email. Ends the task if the reply passes validation.",
                    SubmitReplyArgs,
                    self._submit_reply,
                ),
            )
        }

    def schemas(self, intent: str) -> List[Dict[str, Any]]:
        """Tools offered to the model. reserve_stock is not even visible for inquiries (least privilege)."""
        return [t.schema() for t in self.tools.values() if t.name != "reserve_stock" or intent == ORDER_REQUEST]

    def run(self, name: str, raw_arguments: str, ctx: RunContext) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None or (name == "reserve_stock" and ctx.intent != ORDER_REQUEST):
            return ToolResult(False, {"error": f"tool '{name}' is not available for this email"})
        try:
            args = tool.args_model.model_validate_json(raw_arguments or "{}")
        except ValidationError as exc:
            return ToolResult(False, {"error": f"invalid arguments: {exc.errors(include_url=False)}"})
        try:
            return ToolResult(True, tool.handler(args, ctx))
        except ToolError as exc:
            return ToolResult(False, {"error": str(exc)})

    # ---------------------------------------------------------------- handlers
    def _search_catalog(self, args: SearchCatalogArgs, ctx: RunContext) -> Dict[str, Any]:
        query = args.query.strip()
        if not query:
            raise ToolError("query must not be empty")
        hits = self.store.search(query, self.embed(query[:2000]), self.settings.SEARCH_RESULT_LIMIT)
        ctx.observe([p for p, _ in hits])
        return {
            "results": [_product_view(p, match) for p, match in hits],
            "note": "match: id = exact product ID, similar_id = probable typo, relevant = related by meaning",
        }

    def _get_products(self, args: GetProductsArgs, ctx: RunContext) -> Dict[str, Any]:
        if not args.product_ids or len(args.product_ids) > 20:
            raise ToolError("pass between 1 and 20 product IDs")
        found = self.store.get_products(args.product_ids)
        ctx.observe(list(found.values()))
        missing = sorted({p.strip().upper() for p in args.product_ids} - set(found))
        return {"products": [_product_view(p) for p in found.values()], "not_found": missing}

    def _query_database(self, args: QueryDatabaseArgs, ctx: RunContext) -> Dict[str, Any]:
        try:
            result = self.sandbox.run(args.sql, email_id=ctx.email_id, purpose=args.purpose)
        except SQLRejected as exc:
            raise ToolError(f"query rejected: {exc}") from exc
        payload = result.to_tool_payload()
        if "product_id" in result.columns:  # products surfaced by SQL may be cited in the reply
            idx = result.columns.index("product_id")
            ids = [str(row[idx]) for row in result.rows if row[idx] is not None]
            ctx.observe(list(self.store.get_products(ids).values()))
        return payload

    def _reserve_stock(self, args: ReserveStockArgs, ctx: RunContext) -> Dict[str, Any]:
        if not args.items:
            raise ToolError("items must not be empty")
        requests: List[ReservationRequest] = []
        for item in args.items:
            pid = item.product_id.strip().upper()
            if pid not in ctx.observed_products:
                raise ToolError(
                    f"{pid} has not been looked up in this conversation; find it with search_catalog "
                    "or get_products first"
                )
            if not item.all_available:
                if item.quantity is None or item.quantity < 1:
                    raise ToolError(f"{pid}: quantity must be a positive integer unless all_available is true")
                if item.quantity > self.settings.AGENT_MAX_ORDER_QUANTITY:
                    raise ToolError(
                        f"{pid}: quantities above {self.settings.AGENT_MAX_ORDER_QUANTITY} need human "
                        "review; do not reserve, tell the customer a specialist will follow up"
                    )
            requests.append(ReservationRequest(pid, None if item.all_available else item.quantity, item.all_available))
        try:
            lines = self.store.reserve(ctx.email_id, requests)
        except UnknownProductError as exc:
            raise ToolError(str(exc)) from exc
        for line in lines:
            ctx.lines[line.product_id] = line
        return {"lines": [vars(line) for line in lines]}

    def _submit_reply(self, args: SubmitReplyArgs, ctx: RunContext) -> Dict[str, Any]:
        reply_ctx = ReplyContext(
            allowed_product_ids=set(ctx.observed_products),
            required_product_ids=set(ctx.lines),
            token_map=ctx.token_map,
        )
        try:
            ctx.reply = finalize_reply(args.reply, reply_ctx, self.settings)
        except ReplyValidationError as exc:
            raise ToolError(f"reply rejected: {exc}") from exc
        return {"status": "accepted"}
