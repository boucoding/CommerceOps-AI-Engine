"""
Inbound security guardrails: prompt-injection screening, reversible PII masking
and quarantining of untrusted customer text.

Design notes
- Masking runs in a single pass over one combined pattern, so a value is never
  re-matched by a later pattern, and patterns are tightly bounded so ordinary
  prose ("5 of them, exactly as we discussed...") is never swallowed.
- Long digit runs are always masked; only Luhn-valid ones are labelled CREDIT_CARD.
- Flagged emails are never sent to the model (see ``EmailAgent``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from src.utils.logger import get_logger

logger = get_logger("guardrails")

_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct|Place|Pl|Terrace|Parkway|Pkwy)"
)

# Order matters: earlier alternatives win when spans overlap.
_PII_PATTERNS: List[Tuple[str, str]] = [
    ("EMAIL", r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
    ("CREDIT_CARD", r"(?<!\w)(?:\d[ -]?){12,18}\d(?!\w)"),
    ("SSN", r"(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)"),
    ("PHONE", r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"),
    ("STREET_ADDRESS", rf"\b\d{{1,6}}\s+(?:[A-Z][a-zA-Z]+\.?\s+){{1,4}}{_STREET_SUFFIX}\b\.?"),
]
_COMBINED = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _PII_PATTERNS))
_TOKEN_RE = re.compile(r"\{\{\s*([A-Z_]+_\d+)\s*\}\}")


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for idx, digit in enumerate(reversed(digits)):
        if idx % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


class PIIScrubber:
    """Reversibly replaces personal data with stable tokens such as ``{{EMAIL_1}}``."""

    def __init__(self, allowed_email_domains: Iterable[str] = ()):
        self.allowed_domains = {d.lower() for d in allowed_email_domains}

    def mask(self, text: str) -> Tuple[str, Dict[str, str]]:
        token_map: Dict[str, str] = {}
        by_value: Dict[Tuple[str, str], str] = {}
        counters: Dict[str, int] = {}

        def replace(match: re.Match[str]) -> str:
            kind = match.lastgroup or "PII"
            value = match.group(0)
            if kind == "EMAIL" and value.rsplit("@", 1)[-1].lower() in self.allowed_domains:
                return value
            if kind == "CREDIT_CARD" and not _luhn_valid(value):
                kind = "ACCOUNT_NUMBER"  # long digit runs are still masked, just labelled differently
            key = (kind, value)
            if key not in by_value:
                counters[kind] = counters.get(kind, 0) + 1
                token = f"{{{{{kind}_{counters[kind]}}}}}"
                by_value[key] = token
                token_map[token] = value
            return by_value[key]

        return _COMBINED.sub(replace, text), token_map

    @staticmethod
    def unmask(text: str, token_map: Dict[str, str]) -> str:
        """Restores tokens, tolerating whitespace the model may add inside the braces."""

        def restore(match: re.Match[str]) -> str:
            return token_map.get(f"{{{{{match.group(1)}}}}}", match.group(0))

        return _TOKEN_RE.sub(restore, text)

    @staticmethod
    def leftover_tokens(text: str) -> List[str]:
        return _TOKEN_RE.findall(text)


class PromptInjectionGuard:
    """Signature-based screen for instruction-override and business-tampering attempts."""

    SIGNATURES: List[Tuple[str, str]] = [
        (
            "instruction_override",
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|prompts|rules|messages)",
        ),
        ("system_override", r"\bsystem\s*(?:override|prompt|reset|message)\b"),
        ("role_hijack", r"\byou\s+are\s+now\s+(?:in\s+)?(?:developer\s+mode|dan|unrestricted|an?\s+admin)"),
        (
            "prompt_exfiltration",
            r"\b(?:reveal|print|show|output|repeat)\s+(?:me\s+)?(?:your|the)\s+"
            r"(?:system\s+prompt|hidden\s+prompt|(?:system|initial|original|hidden|secret)\s+instructions)",
        ),
        ("new_instructions", r"\bnew\s+instructions\s*:"),
        ("inventory_tampering", r"\bdisregard\s+(?:the\s+)?(?:stock|catalog|inventory|prices?)\b"),
        ("price_tampering", r"\bset\s+(?:all\s+)?(?:the\s+)?prices?\s+to\s+(?:\$?0\b|free\b)"),
        ("discount_tampering", r"\b(?:grant|apply|give)\s+(?:me\s+)?(?:a\s+)?100\s*%\s+discount"),
        ("free_order", r"\bconfirm\s+(?:the\s+|my\s+)?(?:order|purchase)\s+for\s+free\b"),
        ("boundary_escape", r"</?\s*customer_untrusted_input\s*>"),
        ("markup_injection", r"<\s*script\b|javascript\s*:"),
    ]
    _COMPILED = [(name, re.compile(pattern, re.IGNORECASE)) for name, pattern in SIGNATURES]

    @classmethod
    def inspect(cls, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text))
        return [name for name, pattern in cls._COMPILED if pattern.search(normalized)]

    @staticmethod
    def quarantine(text: str) -> str:
        """Wraps customer text in boundary tags. Only angle brackets are neutralized so the text stays readable."""
        escaped = text.strip().replace("<", "‹").replace(">", "›")
        return f"<customer_untrusted_input>\n{escaped}\n</customer_untrusted_input>"


@dataclass
class ScreenedEmail:
    """Customer email after the inbound guardrail."""

    email_id: str
    masked_subject: str
    masked_body: str
    token_map: Dict[str, str] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return not self.flags

    @property
    def masked_text(self) -> str:
        return f"{self.masked_subject}\n{self.masked_body}".strip()

    def prompt_block(self) -> str:
        """The only representation of customer content ever placed in a prompt."""
        return PromptInjectionGuard.quarantine(f"Subject: {self.masked_subject}\n\n{self.masked_body}")


class SecurityGuardrail:
    def __init__(self, allowed_email_domains: Iterable[str] = ()):
        self.scrubber = PIIScrubber(allowed_email_domains)

    def screen(self, email_id: str, subject: str, body: str) -> ScreenedEmail:
        flags = PromptInjectionGuard.inspect(f"{subject}\n{body}")
        if flags:
            logger.warning("Prompt-injection signatures detected: %s", flags)
        combined, token_map = self.scrubber.mask(f"{subject}\x00{body}")
        masked_subject, masked_body = combined.split("\x00", 1)
        return ScreenedEmail(email_id, masked_subject, masked_body, token_map, flags)
