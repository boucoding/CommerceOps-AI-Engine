"""
Outbound reply validation.

A reply is only accepted if it is grounded: every product-ID-like token it cites
was returned by a tool during this run, every reserved product is mentioned, it
contains no template placeholders and no unknown redaction tokens. On success
the masked PII is restored and the brand signature is appended. Violations are
returned to the agent as a tool error so it can correct its reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Collection, Dict, List

from src.config import Settings
from src.safety.guardrails import PIIScrubber

_PLACEHOLDER = re.compile(r"\[(?:[A-Z][A-Za-z'’]*)(?:[ /'’-][A-Za-z'’]+){0,5}\]")
# Upper-case tokens with at least two letters and two digits look like product IDs (e.g. CBT8901, SKU-1042A).
_PRODUCT_ID_LIKE = re.compile(
    r"\b(?=[A-Z0-9-]*[A-Z][A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d[A-Z0-9-]*\d)[A-Z0-9][A-Z0-9-]{3,31}\b"
)
MAX_REPLY_CHARS = 4000


class ReplyValidationError(ValueError):
    pass


@dataclass
class ReplyContext:
    allowed_product_ids: Collection[str]
    required_product_ids: Collection[str] = ()
    token_map: Dict[str, str] = field(default_factory=dict)


def cited_product_ids(text: str) -> List[str]:
    return sorted({m.group(0).replace("-", "") for m in _PRODUCT_ID_LIKE.finditer(text)})


def finalize_reply(body: str, ctx: ReplyContext, settings: Settings) -> str:
    """Validates the model's reply and returns the text to send. Raises ReplyValidationError."""
    body = body.strip()
    if not body:
        raise ReplyValidationError("the reply is empty")
    if len(body) > MAX_REPLY_CHARS:
        raise ReplyValidationError(f"the reply is longer than {MAX_REPLY_CHARS} characters")
    placeholder = _PLACEHOLDER.search(body)
    if placeholder:
        raise ReplyValidationError(f"the reply contains a template placeholder {placeholder.group(0)}")

    allowed = {p.replace("-", "") for p in ctx.allowed_product_ids}
    cited = set(cited_product_ids(body))
    ungrounded = sorted(cited - allowed)
    if ungrounded:
        raise ReplyValidationError(
            f"the reply cites product IDs that no tool returned in this conversation: {ungrounded}"
        )
    missing = sorted(set(ctx.required_product_ids) - cited)
    if missing:
        raise ReplyValidationError(f"the reply must state the outcome for every reserved product ID: {missing}")

    body = PIIScrubber.unmask(body, ctx.token_map)
    leftover = PIIScrubber.leftover_tokens(body)
    if leftover:
        raise ReplyValidationError(f"the reply contains unknown redaction tokens: {leftover}")
    if settings.SUPPORT_EMAIL not in body:
        body = f"{body}\n\n{settings.signature_block}"
    return body
