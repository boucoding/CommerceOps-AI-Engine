"""
Deterministic fallbacks used in offline mode and whenever the LLM is unavailable
(timeouts, outages, replay misses, invalid output).

They are deliberately conservative: sentence-level intent cues, explicit SKU or
product-name mentions only, and quantities parsed from the words around a mention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from src.models.domain import ORDER_REQUEST, PRODUCT_INQUIRY, Product
from src.retrieval.fuzzy_matcher import find_name_mentions, normalize_sku_text
from src.retrieval.retriever import SkuResolution
from src.retrieval.text import fold

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

_ORDER_CUES = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:want|wanna|would like|'d like|'d want|wish|need|ready|looking|going)\s+to\s+"
        r"(?:order|buy|purchase|place\s+(?:an|my|the)\s+order)\b",
        r"\bplease\s+(?:send|ship)\s+me\b",
        r"\b(?:commit|committing)\s+to\b",
        r"\bi(?:'ll| will)\s+(?:take|buy|order)\b",
        r"\bplace\s+(?:an|my)\s+order\b",
    )
]
_POSTPONE_CUES = re.compile(
    r"\b(?:wait\s+until|next\s+time|later\s+on|maybe\s+later|next\s+(?:week|month|season|year)|not\s+(?:yet|now)|"
    r"thinking\s+(?:of|about)\s+(?:ordering|buying)|considering)\b",
    re.IGNORECASE,
)
_PAST_CUES = re.compile(
    r"\b(?:bought|purchased|got\s+me|ordered\s+before|last\s+(?:year|summer|winter|month)|previously|years\s+ago)\b",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "single": 1,
    "two": 2,
    "couple": 2,
    "pair": 1,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "dozen": 12,
}
_NUM = r"(?:\d{1,3}|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
_RANGE = re.compile(rf"\b({_NUM})\s*(?:to|-|or)\s*({_NUM})\b", re.IGNORECASE)
_ALL_N = re.compile(rf"\ball\s+({_NUM})\b", re.IGNORECASE)
_ALL_AVAILABLE = re.compile(
    r"\ball\s+(?:of\s+)?(?:the\s+)?(?:remaining|you\s+have|that\s+(?:are\s+)?left|available|in\s+stock|your\s+stock)\b",
    re.IGNORECASE,
)
_QUANTITY = re.compile(rf"(?<![$\w])({_NUM})(?!\s*(?:%|s\b|\d))\b", re.IGNORECASE)

_SPANISH = re.compile(r"[¿¡ñ]|\b(?:hola|gracias|pregunta|quiero|está|qué|cómo|tengo|por favor)\b", re.IGNORECASE)
_FRENCH = re.compile(r"\b(?:bonjour|merci|je voudrais|s'il vous plaît|commande)\b", re.IGNORECASE)

_NAME_STOPWORDS = {
    "you",
    "your",
    "for",
    "in",
    "the",
    "a",
    "an",
    "and",
    "me",
    "my",
    "here",
    "there",
    "again",
    "so",
    "all",
    "advance",
    "everyone",
    "team",
    "sir",
    "madam",
    "customer",
    "really",
    "just",
    "very",
    "much",
    "looking",
}
_INTRO_NAME = re.compile(r"\b(?:[Mm]y name is|[Tt]his is|[Ii] am|[Ii]'m)\s+([A-Z][a-z]{1,20})\b")
_SIGNOFF_NAME = re.compile(
    r"(?:[Tt]hanks|[Tt]hank you|[Rr]egards|[Cc]heers|[Bb]est|[Ss]incerely|[Gg]racias|[Ss]aludos)[,!.]?\s+"
    r"(?:so much\s+|again\s+)?[,-]?\s*([A-Z][a-z]{1,20})\s*[.!]?\s*$"
)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def detect_language(text: str) -> str:
    if _SPANISH.search(text):
        return "es"
    if _FRENCH.search(text):
        return "fr"
    return "en"


def _is_active_order_sentence(sentence: str) -> bool:
    return any(c.search(sentence) for c in _ORDER_CUES) and not _POSTPONE_CUES.search(sentence)


def classify_intent(subject: str, body: str) -> str:
    """'order request' if any sentence commits to buying now; postponed or hypothetical purchases are inquiries."""
    for sentence in split_sentences(f"{subject}.\n{body}" if subject else body):
        if _is_active_order_sentence(sentence) and not _PAST_CUES.search(sentence):
            return ORDER_REQUEST
    return PRODUCT_INQUIRY


def is_plausible_name(name: Optional[str], text: str) -> bool:
    """A name is accepted only if it is a capitalized word that literally appears in the email."""
    if not name:
        return False
    name = name.strip()
    if not re.fullmatch(r"[A-ZÀ-Ý][a-zà-ÿ'-]{1,30}", name) or name.lower() in _NAME_STOPWORDS:
        return False
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def extract_customer_name(body: str) -> Optional[str]:
    """The sender's own name from 'my name is X', 'this is X' or a sign-off like 'Thanks, X'."""
    for pattern in (_INTRO_NAME, _SIGNOFF_NAME):
        match = pattern.search(body.strip())
        if match and is_plausible_name(match.group(1), body):
            return match.group(1)
    return None


def _word_value(token: str) -> int:
    token = token.lower()
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


@dataclass
class ParsedQuantity:
    quantity: Optional[int]
    wants_all_available: bool = False


def parse_quantity(segment: str) -> Optional[ParsedQuantity]:
    """Quantity expressed in ``segment``: 'all 5' -> 5, 'all remaining' -> all, 'three to four' -> 4, '2' -> 2."""
    if m := _ALL_N.search(segment):
        return ParsedQuantity(_word_value(m.group(1)))
    if _ALL_AVAILABLE.search(segment):
        return ParsedQuantity(None, wants_all_available=True)
    if m := _RANGE.search(segment):
        return ParsedQuantity(max(_word_value(m.group(1)), _word_value(m.group(2))))
    values = [_word_value(m.group(1)) for m in _QUANTITY.finditer(segment)]
    values = [v for v in values if 0 < v < 1000]
    if values:
        return ParsedQuantity(values[-1])  # the number closest to the product mention
    return None


@dataclass
class RuleItem:
    product_id: str
    quantity: Optional[int]
    wants_all_available: bool


def extract_items(
    body: str, mentioned: Sequence[Product], resolutions: Sequence[SkuResolution], name_threshold: float
) -> List[RuleItem]:
    """
    Items ordered now: products mentioned (by SKU or name) in sentences that carry
    an active order cue and no past/postponement cue. The quantity comes from the
    text between the previous mention and this one, else the text that follows it.
    """
    text = normalize_sku_text(body)
    token_to_id = {r.token: r.product_id for r in resolutions}
    items: List[RuleItem] = []
    seen: set[str] = set()
    sentences = split_sentences(text)
    has_order_sentence = any(_is_active_order_sentence(s) for s in sentences)

    for sentence in sentences:
        if _PAST_CUES.search(sentence) or _POSTPONE_CUES.search(sentence):
            continue
        if has_order_sentence and not _is_active_order_sentence(sentence) and not _QUANTITY.search(sentence):
            continue
        positions = []
        for token, pid in token_to_id.items():
            for m in re.finditer(rf"\b{token}\b", sentence):
                positions.append((m.start(), m.end(), pid))
        for mention in find_name_mentions(sentence, mentioned, name_threshold):
            pid = mention.product.product_id
            if all(p[2] != pid for p in positions):
                idx = fold(sentence).find(fold(mention.product.name.split()[0]))
                start = max(idx, 0)
                positions.append((start, start + len(mention.product.name), pid))
        positions.sort()

        cursor = 0
        for i, (start, end, pid) in enumerate(positions):
            if pid in seen:
                cursor = end
                continue
            before = sentence[cursor:start]
            nxt = positions[i + 1][0] if i + 1 < len(positions) else len(sentence)
            after = sentence[end:nxt]
            parsed = parse_quantity(before) or parse_quantity(after) or ParsedQuantity(1)
            items.append(RuleItem(pid, parsed.quantity, parsed.wants_all_available))
            seen.add(pid)
            cursor = end
    return items
