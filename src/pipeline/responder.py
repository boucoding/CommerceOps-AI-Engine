"""
Customer reply generation for orders and inquiries.

The model writes only the reply body from structured, verified facts; framing and
validation happen in ``template_engine``. Any model failure or validation failure
falls back to a deterministic template that states the same facts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from src.config import Settings
from src.llm.base import LLMClient, LLMError
from src.models.domain import STATUS_CREATED, OrderLine, Product, UsageStats
from src.pipeline import template_engine as tpl
from src.pipeline.guardrails import ScreenedEmail
from src.pipeline.prompts import inquiry_reply_prompt, order_reply_prompt
from src.utils.logger import get_logger

logger = get_logger("pipeline.responder")

_LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
}


@dataclass
class Reply:
    text: str
    degraded_reason: Optional[str] = None


def _product_context(p: Product) -> str:
    availability = "in stock" if p.stock > 0 else "out of stock"
    return (
        f"- {p.product_id} | {p.name} | {p.category} | seasons: {p.seasons} | ${p.price:.2f} | {availability}\n"
        f"  {p.description}"
    )


class ReplyWriter:
    def __init__(self, llm: Optional[LLMClient], settings: Settings):
        self.llm = llm
        self.settings = settings

    def _generate(
        self, task: str, system: str, user: str, ctx: tpl.ReplyContext, fallback_body: str, usage: UsageStats
    ) -> Reply:
        reason: Optional[str] = None
        if self.llm is not None:
            try:
                result = self.llm.complete(task=task, system=system, user=user)
                usage.record_call(result.usage.prompt_tokens, result.usage.completion_tokens, result.replayed)
                return Reply(tpl.finalize_reply(result.value, ctx, self.settings))
            except (LLMError, tpl.ReplyValidationError) as exc:
                reason = f"{task}: {exc}"
                logger.warning("Reply fell back to template: %s", reason)
        return Reply(tpl.finalize_reply(fallback_body, ctx, self.settings), reason)

    def order_reply(
        self,
        email: ScreenedEmail,
        language: str,
        customer_name: Optional[str],
        lines: Sequence[OrderLine],
        alternatives: Dict[str, List[Product]],
        unresolved: Sequence[str],
        usage: UsageStats,
    ) -> Reply:
        needs_clarification = not lines
        facts = {
            "language": _LANGUAGE_NAMES.get(language, language),
            "needs_clarification": needs_clarification,
            "lines": [
                {
                    "product_id": ln.product_id,
                    "product_name": ln.product_name,
                    "requested_quantity": ln.requested_quantity,
                    "status": ln.status,
                    **({"available_quantity": ln.remaining_stock} if ln.status != STATUS_CREATED else {}),
                }
                for ln in lines
            ],
            "alternatives": {
                pid: [{"product_id": p.product_id, "name": p.name, "price": f"${p.price:.2f}"} for p in alts]
                for pid, alts in alternatives.items()
            },
            "unidentified_items": list(unresolved),
        }
        allowed = {ln.product_id for ln in lines} | {p.product_id for alts in alternatives.values() for p in alts}
        ctx = tpl.ReplyContext(language, customer_name, allowed, {ln.product_id for ln in lines}, email.token_map)
        fallback = (
            tpl.clarification_template(self.settings)
            if needs_clarification
            else tpl.order_template(lines, alternatives, self.settings)
        )
        user = (
            f"Email ID: {email.email_id}\n{email.prompt_block()}\n\n"
            f"Order facts (authoritative):\n{json.dumps(facts, indent=2, ensure_ascii=False)}"
        )
        return self._generate("order_reply", order_reply_prompt(self.settings), user, ctx, fallback, usage)

    def inquiry_reply(
        self,
        email: ScreenedEmail,
        language: str,
        customer_name: Optional[str],
        mentioned: Sequence[Product],
        related: Sequence[Product],
        usage: UsageStats,
    ) -> Reply:
        allowed = {p.product_id for p in [*mentioned, *related]}
        ctx = tpl.ReplyContext(language, customer_name, allowed, (), email.token_map)
        context = (
            "Products the customer mentioned:\n"
            + ("\n".join(_product_context(p) for p in mentioned) or "- none")
            + "\n\nOther relevant products:\n"
            + ("\n".join(_product_context(p) for p in related) or "- none")
        )
        user = (
            f"Email ID: {email.email_id}\nReply language: {_LANGUAGE_NAMES.get(language, language)}\n"
            f"{email.prompt_block()}\n\nCatalog context:\n{context}"
        )
        fallback = tpl.inquiry_template(mentioned, related, self.settings)
        return self._generate("inquiry_reply", inquiry_reply_prompt(self.settings), user, ctx, fallback, usage)

    def escalation_reply(self, email: ScreenedEmail, language: str, customer_name: Optional[str]) -> Reply:
        ctx = tpl.ReplyContext(language, customer_name, (), (), email.token_map)
        return Reply(tpl.finalize_reply(tpl.escalation_template(self.settings), ctx, self.settings))
