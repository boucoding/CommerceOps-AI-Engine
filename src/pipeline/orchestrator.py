"""
End-to-end orchestration.

Each email goes through three phases:

1. analyze  (parallel)   guardrail screening, PII masking, intent classification,
                         retrieval and order extraction
2. commit   (sequential) idempotent claim of the email id and atomic stock
                         reservation, in arrival order so stock is allocated
                         first-come-first-served and results are reproducible
3. respond  (parallel)   grounded reply generation, validation and audit record

Batches run phases 1 and 3 on a thread pool; a single email runs all three inline.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, TypeVar

from src.config import Settings
from src.llm.base import LLMClient
from src.models.domain import (
    ORDER_REQUEST,
    STATUS_CREATED,
    CustomerEmail,
    EmailOutcome,
    EmailStatus,
    OrderLine,
    Product,
    UsageStats,
)
from src.pipeline.guardrails import ScreenedEmail, SecurityGuardrail
from src.pipeline.intent_classifier import IntentClassifier, IntentResult
from src.pipeline.order_extractor import ExtractionResult, OrderExtractor
from src.pipeline.responder import Reply, ReplyWriter
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.retriever import HybridRetriever, Retrieval
from src.storage.base import DuplicateEmailError, Store
from src.utils.logger import correlation, get_logger

logger = get_logger("pipeline.orchestrator")

T = TypeVar("T")
R = TypeVar("R")

_ALTERNATIVES_PER_LINE = 2


@dataclass
class _Work:
    email: CustomerEmail
    usage: UsageStats = field(default_factory=UsageStats)
    degraded: List[str] = field(default_factory=list)
    elapsed: float = 0.0
    screened: Optional[ScreenedEmail] = None
    intent: Optional[IntentResult] = None
    retrieval: Optional[Retrieval] = None
    extraction: Optional[ExtractionResult] = None
    lines: List[OrderLine] = field(default_factory=list)
    outcome: Optional[EmailOutcome] = None  # set early for duplicates and failures

    @property
    def escalated(self) -> bool:
        return self.screened is not None and not self.screened.is_safe


class CommerceOpsOrchestrator:
    def __init__(
        self, store: Store, llm: Optional[LLMClient], settings: Settings, embedder: Optional[EmbeddingService] = None
    ):
        self.store = store
        self.llm = llm
        self.settings = settings
        self.embedder = embedder or EmbeddingService(llm, settings.EMBEDDING_DIMENSIONS)
        self.retriever = HybridRetriever(store, self.embedder, settings)
        support_domain = settings.SUPPORT_EMAIL.rsplit("@", 1)[-1]
        self.guardrail = SecurityGuardrail(allowed_email_domains=[support_domain])
        self.classifier = IntentClassifier(llm, settings)
        self.extractor = OrderExtractor(llm, settings)
        self.writer = ReplyWriter(llm, settings)

    # ------------------------------------------------------------------ public
    def process_email(self, email: CustomerEmail) -> EmailOutcome:
        return self.process_batch([email])[0]

    def process_batch(self, emails: Sequence[CustomerEmail]) -> List[EmailOutcome]:
        ids = [e.email_id for e in emails]
        if len(set(ids)) != len(ids):
            raise ValueError("email_id values must be unique within a batch")
        logger.info("Processing %d email(s)", len(emails))
        works = [_Work(email) for email in emails]
        self._parallel(self._analyze, works)
        for work in works:
            self._guarded(work, "commit", self._commit)
        self._parallel(self._respond, works)
        outcomes = [w.outcome for w in works]
        assert all(o is not None for o in outcomes)
        return outcomes  # type: ignore[return-value]

    # --------------------------------------------------------------- execution
    def _parallel(self, fn: Callable[[_Work], None], works: List[_Work]) -> None:
        name = fn.__name__.lstrip("_")
        if len(works) <= 1 or self.settings.BATCH_WORKERS <= 1:
            for work in works:
                self._guarded(work, name, fn)
            return
        with ThreadPoolExecutor(max_workers=self.settings.BATCH_WORKERS, thread_name_prefix="email") as pool:
            list(pool.map(lambda w: self._guarded(w, name, fn), works))

    def _guarded(self, work: _Work, stage: str, fn: Callable[[_Work], None]) -> None:
        """Runs one stage for one email; an unexpected error escalates that email instead of failing the batch."""
        if work.outcome is not None:  # duplicate or already failed
            return
        started = time.perf_counter()
        with correlation(work.email.email_id):
            try:
                fn(work)
            except Exception as exc:  # noqa: BLE001 - isolate failures per email
                logger.exception("Unexpected error in %s stage", stage)
                work.degraded.append(f"{stage}: internal error: {type(exc).__name__}")
                self._fail(work)
            finally:
                work.elapsed += time.perf_counter() - started

    # ------------------------------------------------------------------ stages
    def _analyze(self, work: _Work) -> None:
        email = work.email
        existing = self.store.get_outcome(email.email_id)
        if existing is not None:
            logger.info("Already processed; returning stored outcome")
            existing.duplicate = True
            work.outcome = existing
            return

        work.screened = screened = self.guardrail.screen(email.email_id, email.subject, email.body)
        if not screened.is_safe:
            # Never send flagged content to the model; classify with rules for routing/reporting only.
            work.intent = self.classifier.rules_result(screened, None)
            return

        work.intent = intent = self.classifier.classify(screened, work.usage)
        if intent.degraded_reason:
            work.degraded.append(intent.degraded_reason)

        top_k = self.settings.ORDER_CANDIDATE_TOP_K if intent.category == ORDER_REQUEST else self.settings.INQUIRY_TOP_K
        work.retrieval = self.retriever.retrieve(screened.masked_text, top_k)
        if work.retrieval.degraded_reason:
            work.degraded.append(work.retrieval.degraded_reason)

        if intent.category == ORDER_REQUEST:
            work.extraction = self.extractor.extract(screened, work.retrieval, work.usage)
            if work.extraction.degraded_reason:
                work.degraded.append(work.extraction.degraded_reason)

    def _commit(self, work: _Work) -> None:
        email_id = work.email.email_id
        try:
            if work.extraction is not None and not work.escalated:
                work.lines = self.store.commit_order(email_id, work.extraction.items)
                for line in work.lines:
                    logger.info(
                        "Order line %s x%d -> %s (stock now %d)",
                        line.product_id,
                        line.requested_quantity,
                        line.status,
                        line.remaining_stock,
                    )
            else:
                self.store.claim_email(email_id)
        except DuplicateEmailError:
            stored = self.store.get_outcome(email_id)
            if stored is None:
                raise
            stored.duplicate = True
            work.outcome = stored

    def _respond(self, work: _Work) -> None:
        screened, intent = work.screened, work.intent
        assert screened is not None and intent is not None
        status = EmailStatus.PROCESSED
        context_ids: List[str] = []
        unresolved: List[str] = []

        if work.escalated:
            status = EmailStatus.ESCALATED
            reply = self.writer.escalation_reply(screened, intent.language, intent.customer_name)
        elif intent.category == ORDER_REQUEST:
            assert work.extraction is not None
            unresolved = work.extraction.unresolved
            if not work.lines:
                status = EmailStatus.NEEDS_CLARIFICATION
            alternatives = self._alternatives(work.lines)
            context_ids = [ln.product_id for ln in work.lines] + [
                p.product_id for alts in alternatives.values() for p in alts
            ]
            reply = self.writer.order_reply(
                screened, intent.language, intent.customer_name, work.lines, alternatives, unresolved, work.usage
            )
        else:
            assert work.retrieval is not None
            fresh = self.store.get_products(work.retrieval.product_ids)  # current availability
            mentioned = [fresh.get(p.product_id, p) for p in work.retrieval.mentioned]
            related = [fresh.get(p.product_id, p) for p in work.retrieval.related]
            context_ids = [p.product_id for p in mentioned + related]
            reply = self.writer.inquiry_reply(
                screened, intent.language, intent.customer_name, mentioned, related, work.usage
            )

        if reply.degraded_reason:
            work.degraded.append(reply.degraded_reason)
        work.outcome = self._outcome(work, status, reply, context_ids, unresolved)
        self.store.save_outcome(work.outcome)
        logger.info("Done: %s, %s%s", intent.category, status.value, " (degraded)" if work.degraded else "")

    # ----------------------------------------------------------------- helpers
    def _alternatives(self, lines: Sequence[OrderLine]) -> Dict[str, List[Product]]:
        unavailable = [ln.product_id for ln in lines if ln.status != STATUS_CREATED]
        if not unavailable:
            return {}
        products = self.store.get_products(unavailable)
        ordered = {ln.product_id for ln in lines}
        result: Dict[str, List[Product]] = {}
        for pid in unavailable:
            if pid in products:
                alts = self.retriever.alternatives(products[pid], _ALTERNATIVES_PER_LINE + len(ordered))
                result[pid] = [p for p in alts if p.product_id not in ordered][:_ALTERNATIVES_PER_LINE]
        return result

    def _outcome(
        self, work: _Work, status: EmailStatus, reply: Reply, context_ids: List[str], unresolved: List[str]
    ) -> EmailOutcome:
        intent = work.intent
        assert intent is not None and work.screened is not None
        return EmailOutcome(
            email_id=work.email.email_id,
            category=intent.category,
            status=status,
            response=reply.text,
            confidence=intent.confidence,
            language=intent.language,
            customer_name=intent.customer_name,
            lines=list(work.lines),
            unresolved_mentions=list(unresolved),
            context_product_ids=context_ids,
            guardrail_flags=list(work.screened.flags),
            degraded_reasons=list(work.degraded),
            usage=work.usage,
            latency_ms=int(work.elapsed * 1000),
        )

    def _fail(self, work: _Work) -> None:
        """Last-resort outcome: route to a human with a holding reply. Stock already reserved is kept and reported."""
        screened = work.screened or self.guardrail.screen(work.email.email_id, work.email.subject, work.email.body)
        intent = work.intent or self.classifier.rules_result(screened, None)
        work.screened, work.intent = screened, intent
        reply = self.writer.escalation_reply(screened, intent.language, intent.customer_name)
        work.outcome = self._outcome(work, EmailStatus.ESCALATED, reply, [], [])
        try:
            self.store.save_outcome(work.outcome)
        except Exception:  # noqa: BLE001
            logger.exception("Could not persist failure outcome")
