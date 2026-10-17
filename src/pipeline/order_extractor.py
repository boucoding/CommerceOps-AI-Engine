"""
Two-stage order extraction.

Stage 1 retrieves a small candidate set (explicit SKU/name mentions plus hybrid
search results). Stage 2 asks the model to map the email onto those candidates
with Structured Outputs. Every product ID the model returns is then re-validated
against the catalog: exact candidate IDs are accepted, near-miss IDs are corrected
with the same typo-tolerant matcher, and anything else is reported as unresolved
rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import Settings
from src.llm.base import LLMClient, LLMError
from src.models.domain import Product, RequestedItem, UsageStats
from src.models.llm_schemas import ExtractedOrder
from src.pipeline import rules
from src.pipeline.guardrails import ScreenedEmail
from src.pipeline.prompts import extraction_prompt
from src.retrieval.fuzzy_matcher import best_sku_match, canonical_sku, find_name_mentions
from src.retrieval.retriever import Retrieval
from src.utils.logger import get_logger

logger = get_logger("pipeline.extractor")


@dataclass
class ExtractionResult:
    items: List[RequestedItem] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    degraded_reason: Optional[str] = None


def _candidate_lines(products: List[Product]) -> str:
    return "\n".join(f"- {p.product_id} | {p.name} | {p.category}" for p in products) or "- (no candidates found)"


def _merge(items: List[RequestedItem]) -> List[RequestedItem]:
    """Combines duplicate lines for the same product (quantities add up; 'all available' wins)."""
    merged: Dict[str, RequestedItem] = {}
    for item in items:
        current = merged.get(item.product_id)
        if current is None:
            merged[item.product_id] = RequestedItem(item.product_id, item.quantity, item.wants_all_available)
        elif current.wants_all_available or item.wants_all_available:
            current.wants_all_available, current.quantity = True, None
        else:
            current.quantity = (current.quantity or 1) + (item.quantity or 1)
    return list(merged.values())


class OrderExtractor:
    def __init__(self, llm: Optional[LLMClient], settings: Settings):
        self.llm = llm
        self.settings = settings

    def rules_extract(self, email: ScreenedEmail, retrieval: Retrieval, reason: Optional[str]) -> ExtractionResult:
        confident = self._confident_mentions(email, retrieval)
        rule_items = rules.extract_items(
            email.masked_body,
            [p for p in retrieval.mentioned if p.product_id in confident],
            [r for r in retrieval.sku_resolutions if r.product_id in confident],
            self.settings.NAME_MATCH_THRESHOLD,
        )
        items = [RequestedItem(i.product_id, i.quantity, i.wants_all_available) for i in rule_items]
        return ExtractionResult(items=_merge(items), degraded_reason=reason)

    def _confident_mentions(self, email: ScreenedEmail, retrieval: Retrieval) -> set[str]:
        """
        Mentioned products we are willing to order without model confirmation: exact SKU matches and name
        matches. A typo-corrected SKU is accepted only if the product's name also appears in the email.
        """
        named = {
            m.product.product_id
            for m in find_name_mentions(email.masked_text, retrieval.mentioned, self.settings.NAME_MATCH_THRESHOLD)
        }
        confident = set(named)
        for res in retrieval.sku_resolutions:
            if res.score >= 100.0 or res.product_id in named:
                confident.add(res.product_id)
        return confident

    def extract(self, email: ScreenedEmail, retrieval: Retrieval, usage: UsageStats) -> ExtractionResult:
        if self.llm is None:
            return self.rules_extract(email, retrieval, None)

        candidates = retrieval.products
        try:
            result = self.llm.parse(
                task="extract_order",
                system=extraction_prompt(self.settings, _candidate_lines(candidates)),
                user=f"Email ID: {email.email_id}\n{email.prompt_block()}",
                schema=ExtractedOrder,
            )
        except LLMError as exc:
            logger.warning("Order extraction fell back to rules: %s", exc)
            return self.rules_extract(email, retrieval, f"extract_order: {exc}")

        usage.record_call(result.usage.prompt_tokens, result.usage.completion_tokens, result.replayed)
        candidate_ids = [p.product_id for p in candidates]
        items: List[RequestedItem] = []
        unresolved: List[str] = []
        for extracted in result.value.items:
            if not extracted.is_active_order:
                continue
            product_id = None
            if extracted.product_id:
                match = best_sku_match(
                    canonical_sku(extracted.product_id), candidate_ids, self.settings.SKU_FUZZY_THRESHOLD
                )
                product_id = match[0] if match else None
            if product_id is None:
                label = extracted.product_name or extracted.product_id or "unspecified item"
                logger.info("Unresolved order item: %r", label)
                unresolved.append(label)
                continue
            quantity = None if extracted.wants_all_available else max(1, int(extracted.quantity or 1))
            items.append(RequestedItem(product_id, quantity, extracted.wants_all_available))
        return ExtractionResult(items=_merge(items), unresolved=unresolved)
