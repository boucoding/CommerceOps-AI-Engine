"""
Deterministic reply framing, fallback templates and outbound validation.

Every outbound reply is assembled as: localized greeting + body + localized
sign-off + brand signature from settings. The body comes from the LLM when
available and passes validation; otherwise from the templates below. Validation
rejects placeholder leaks, unknown redaction tokens and product IDs that are not
part of the grounded context, so a reply never cites a product we did not verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Sequence

from src.config import Settings
from src.models.domain import STATUS_CREATED, OrderLine, Product
from src.pipeline.guardrails import PIIScrubber
from src.retrieval.fuzzy_matcher import extract_sku_tokens

_GREETINGS = {"en": ("Dear {name},", "Hello,"), "es": ("Hola {name},", "Hola,"), "fr": ("Bonjour {name},", "Bonjour,")}
_CLOSINGS = {"en": "Warm regards,", "es": "Saludos cordiales,", "fr": "Bien cordialement,"}

_PLACEHOLDER = re.compile(r"\[(?:[A-Z][A-Za-z'’]*)(?:[ /'’-][A-Za-z'’]+){0,5}\]")
_LEADING_GREETING = re.compile(r"^\s*(?:dear|hello|hi|hey|hola|bonjour|estimad[oa])\b[^\n]{0,60},\s*\n+", re.IGNORECASE)
# A short closing line ("Best regards,", "Saludos cordiales,", "Thanks,") plus up to five short signature lines.
_TRAILING_SIGNOFF = re.compile(
    r"\n+[ \t]*(?:(?:warm|warmest|kind|best|with)\s+)?(?:regards|wishes|sincerely|cheers|best|thanks|thank you|"
    r"saludos(?: cordiales)?|atentamente|cordialement|bien cordialement)[ \t]*[,!.]?[ \t]*(?:\n[^\n]{0,60}){0,5}\s*$",
    re.IGNORECASE,
)


class ReplyValidationError(ValueError):
    pass


@dataclass
class ReplyContext:
    language: str
    customer_name: Optional[str]
    allowed_product_ids: Collection[str]
    required_product_ids: Collection[str] = ()
    token_map: Optional[Dict[str, str]] = None


def _lang(language: str) -> str:
    return language if language in _GREETINGS else "en"


def greeting(language: str, name: Optional[str]) -> str:
    named, anonymous = _GREETINGS[_lang(language)]
    return named.format(name=name) if name else anonymous


def strip_framing(body: str) -> str:
    """Removes a greeting or sign-off the model added despite instructions."""
    body = _LEADING_GREETING.sub("", body.strip(), count=1)
    return _TRAILING_SIGNOFF.sub("", body).strip()


def finalize_reply(body: str, ctx: ReplyContext, settings: Settings) -> str:
    """Frames, unmasks and validates a reply body. Raises ReplyValidationError on any violation."""
    body = strip_framing(body)
    if not body:
        raise ReplyValidationError("empty reply body")
    if len(body) > 4000:
        raise ReplyValidationError("reply body too long")
    if _PLACEHOLDER.search(body):
        raise ReplyValidationError(f"placeholder leak: {_PLACEHOLDER.search(body).group(0)}")  # type: ignore[union-attr]

    mentioned = set(extract_sku_tokens(body))
    unknown = sorted(mentioned - set(ctx.allowed_product_ids))
    if unknown:
        raise ReplyValidationError(f"ungrounded product IDs: {unknown}")
    missing = sorted(set(ctx.required_product_ids) - mentioned)
    if missing:
        raise ReplyValidationError(f"reply omits ordered product IDs: {missing}")

    if ctx.token_map:
        body = PIIScrubber.unmask(body, ctx.token_map)
    leftover = PIIScrubber.leftover_tokens(body)
    if leftover:
        raise ReplyValidationError(f"unknown redaction tokens: {leftover}")

    lang = _lang(ctx.language)
    text = f"{greeting(lang, ctx.customer_name)}\n\n{body}\n\n{_CLOSINGS[lang]}\n{settings.signature_block}"
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --------------------------------------------------------------------------- templates


def _money(value: float) -> str:
    return f"${value:,.2f}"


def order_template(lines: Sequence[OrderLine], alternatives: Dict[str, List[Product]], settings: Settings) -> str:
    created = [ln for ln in lines if ln.status == STATUS_CREATED]
    unavailable = [ln for ln in lines if ln.status != STATUS_CREATED]
    parts: List[str] = [f"Thank you for your order with {settings.BRAND_NAME}."]
    if created:
        items = "\n".join(f"- {ln.product_name} ({ln.product_id}) x {ln.requested_quantity}" for ln in created)
        parts.append(
            f"We're pleased to confirm the following item(s), which are now being prepared for shipment:\n{items}"
        )
    for ln in unavailable:
        text = (
            f"Unfortunately we can't fulfil {ln.requested_quantity} x {ln.product_name} ({ln.product_id}) "
            f"right now: only {ln.remaining_stock} in stock. Nothing has been reserved or charged for this item."
        )
        alts = alternatives.get(ln.product_id) or []
        if alts:
            options = "; ".join(f"{p.name} ({p.product_id}, {_money(p.price)})" for p in alts)
            text += f" You may like these available alternatives: {options}."
        text += " Reply to this email and we'll let you know as soon as it's back in stock."
        parts.append(text)
    return "\n\n".join(parts)


def clarification_template(settings: Settings) -> str:
    return (
        f"Thank you for wanting to order from {settings.BRAND_NAME}! We couldn't identify which product you'd like. "
        "Could you reply with the product name or its product ID (the code of three letters and four digits "
        "shown on the product page) and the quantity? We'll confirm availability right away."
    )


def inquiry_template(mentioned: Sequence[Product], related: Sequence[Product], settings: Settings) -> str:
    parts = [f"Thank you for your interest in {settings.BRAND_NAME}."]
    if mentioned:
        parts.append("Here are the details for the product(s) you asked about:")
        parts.extend(
            f"- {p.name} ({p.product_id}), {_money(p.price)}, {p.seasons}, "
            f"{'in stock' if p.stock > 0 else 'currently out of stock'}: {p.description}"
            for p in mentioned
        )
    suggestions = list(related[:3])
    if suggestions:
        parts.append("You might also like:" if mentioned else "Based on your message, these pieces may interest you:")
        parts.extend(f"- {p.name} ({p.product_id}), {_money(p.price)}, {p.seasons}" for p in suggestions)
    parts.append("If you have any other questions, just reply to this email and a member of our team will help.")
    return "\n".join(parts[:1]) + "\n\n" + "\n".join(parts[1:])


def escalation_template(settings: Settings) -> str:
    return (
        f"Thank you for contacting {settings.BRAND_NAME}. Your message has been passed to a member of our "
        "customer care team, who will review it personally and get back to you shortly."
    )
