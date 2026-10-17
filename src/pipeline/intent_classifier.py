"""
Intent classification with Structured Outputs and a deterministic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import Settings
from src.llm.base import LLMClient, LLMError
from src.models.domain import UsageStats
from src.models.llm_schemas import IntentClassification
from src.pipeline import rules
from src.pipeline.guardrails import ScreenedEmail
from src.pipeline.prompts import classification_prompt
from src.utils.logger import get_logger

logger = get_logger("pipeline.intent")


@dataclass
class IntentResult:
    category: str
    confidence: Optional[float]
    customer_name: Optional[str]
    language: str
    degraded_reason: Optional[str] = None


class IntentClassifier:
    def __init__(self, llm: Optional[LLMClient], settings: Settings):
        self.llm = llm
        self.settings = settings

    def rules_result(self, email: ScreenedEmail, reason: Optional[str]) -> IntentResult:
        return IntentResult(
            category=rules.classify_intent(email.masked_subject, email.masked_body),
            confidence=None,
            customer_name=rules.extract_customer_name(email.masked_body),
            language=rules.detect_language(email.masked_text),
            degraded_reason=reason,
        )

    def classify(self, email: ScreenedEmail, usage: UsageStats) -> IntentResult:
        if self.llm is None:
            return self.rules_result(email, None)
        try:
            result = self.llm.parse(
                task="classify",
                system=classification_prompt(self.settings),
                user=f"Email ID: {email.email_id}\n{email.prompt_block()}",
                schema=IntentClassification,
            )
        except LLMError as exc:
            logger.warning("Classification fell back to rules: %s", exc)
            return self.rules_result(email, f"classify: {exc}")

        usage.record_call(result.usage.prompt_tokens, result.usage.completion_tokens, result.replayed)
        parsed = result.value
        # Only accept a name that literally appears in the email: guards against hallucinated names.
        name = parsed.customer_name.strip() if parsed.customer_name else None
        if not rules.is_plausible_name(name, email.masked_body):
            name = rules.extract_customer_name(email.masked_body)
        language = (parsed.language or "en").strip().lower()[:2] or "en"
        return IntentResult(
            category=parsed.category,
            confidence=min(1.0, max(0.0, float(parsed.confidence))),
            customer_name=name,
            language=language,
        )
