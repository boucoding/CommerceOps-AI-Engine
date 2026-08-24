"""
Security Guardrails & PII Compliance Engine for CommerceOps AI.
Provides GDPR/CCPA-compliant reversible PII tokenization and sub-millisecond prompt injection defense.
"""

import re
import html
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from src.utils.logger import logger

@dataclass
class GuardrailResult:
    """Result from security and compliance inspection."""
    is_safe: bool
    sanitized_text: str
    token_map: Dict[str, str] = field(default_factory=dict)
    flagged_patterns: List[str] = field(default_factory=list)
    risk_level: str = "low"  # 'low', 'medium', 'critical'

class PIIScrubber:
    """
    High-speed, deterministic PII Masker & Tokenizer.
    Reversibly masks sensitive customer data before LLM transmission to ensure GDPR/CCPA compliance.
    """
    # Regex patterns for sensitive PII entities
    PATTERNS = {
        "EMAIL": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "PHONE": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "CREDIT_CARD": r'\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{15,16}\b',
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "STREET_ADDRESS": r'\b\d+\s+[A-Za-z0-9\.\,\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way)\b'
    }

    @classmethod
    def mask_pii(cls, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replaces PII entities with unique indexed tokens.
        Returns:
            Tuple of (masked_text, token_to_original_map)
        """
        token_map: Dict[str, str] = {}
        masked_text = text

        for pii_type, pattern in cls.PATTERNS.items():
            matches = list(re.finditer(pattern, masked_text, flags=re.IGNORECASE))
            # Reverse order to avoid index shifts during replacement
            for idx, match in enumerate(matches, 1):
                original_value = match.group(0)
                # Avoid masking standard company support addresses
                if "@aurafashion.com" in original_value.lower():
                    continue

                token = f"{{{{{pii_type}_{idx}}}}}"
                token_map[token] = original_value
                masked_text = masked_text.replace(original_value, token)

        return masked_text, token_map

    @classmethod
    def unmask_pii(cls, text: str, token_map: Dict[str, str]) -> str:
        """Restores masked tokens back to their original values in outbound responses."""
        unmasked = text
        for token, original in token_map.items():
            unmasked = unmasked.replace(token, original)
        return unmasked

class PromptInjectionGuard:
    """
    Lightweight, sub-millisecond Prompt Injection & Jailbreak Defense.
    Detects system prompt override attempts, instruction leaks, and adversarial business tampering.
    """
    # Known adversarial jailbreak signatures
    INJECTION_SIGNATURES = [
        r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)",
        r"system\s+(override|prompt|reset)",
        r"you\s+are\s+now\s+(in\s+)?(developer\s+mode|dan|unrestricted)",
        r"(reveal|print|show|output)\s+(your\s+)?(system\s+prompt|instructions)",
        r"disregard\s+(stock|catalog|inventory|prices)",
        r"set\s+(all\s+)?prices?\s+to\s+\$?(0|free)",
        r"grant\s+100%\s+discount",
        r"confirm\s+(order|purchase)\s+for\s+free",
        r"format\s+as\s+admin\s+token",
        r"<\s*script\s*>",
        r"javascript\s*:"
    ]

    @classmethod
    def inspect(cls, text: str) -> Tuple[bool, List[str], str]:
        """
        Inspects text for prompt injection patterns.
        Returns:
            Tuple of (is_safe: bool, flagged_patterns: List[str], risk_level: str)
        """
        flagged = []
        normalized = text.lower()

        for pattern in cls.INJECTION_SIGNATURES:
            if re.search(pattern, normalized, re.IGNORECASE):
                flagged.append(pattern)

        if flagged:
            logger.warning(f"[Security Guardrail Alert] Prompt injection pattern detected: {flagged}")
            return False, flagged, "critical"

        return True, [], "low"

    @classmethod
    def wrap_in_quarantine_tags(cls, text: str) -> str:
        """
        Quarantines customer input inside strict XML boundaries to prevent execution context escapes.
        """
        escaped = html.escape(text.strip())
        return f"<customer_untrusted_input>\n{escaped}\n</customer_untrusted_input>"

class SecurityGuardrail:
    """
    Unified In-Process Security & Compliance Guardrail.
    Executes PII scrubbing, injection screening, and input boundary quarantining.
    """
    @classmethod
    def process_inbound(cls, text: str) -> GuardrailResult:
        """Runs end-to-end inbound sanitization."""
        # 1. Prompt Injection Detection
        is_safe, flagged, risk = PromptInjectionGuard.inspect(text)
        if not is_safe:
            return GuardrailResult(
                is_safe=False,
                sanitized_text=text,
                flagged_patterns=flagged,
                risk_level=risk
            )

        # 2. Reversible PII Masking (GDPR/CCPA)
        masked_text, token_map = PIIScrubber.mask_pii(text)

        # 3. XML Boundary Quarantining
        quarantined_text = PromptInjectionGuard.wrap_in_quarantine_tags(masked_text)

        return GuardrailResult(
            is_safe=True,
            sanitized_text=quarantined_text,
            token_map=token_map,
            risk_level="low"
        )
