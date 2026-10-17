"""
SKU normalization, typo-tolerant SKU resolution and product-name mention detection.

These functions are pure: both storage backends fetch candidates their own way
(Python scan vs. pg_trgm index) and then apply exactly the same scoring here, so
in-memory and PostgreSQL retrieval agree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

from src.models.domain import Product
from src.retrieval.text import normalize_phrase

# Three letters followed by four digits, optionally bracketed or split by spaces/dashes: "[CBT 89 01]", "CBT-8901".
_SKU_LIKE = re.compile(r"\[?\b([A-Za-z]{3})[\s\-_]*(\d{2})[\s\-_]*(\d{2})\b\]?")
_SKU_TOKEN = re.compile(r"\b[A-Z]{3}\d{4}\b")


def normalize_sku_text(text: str) -> str:
    """Rewrites SKU-like spans to canonical form: '[CBT 89 01]' -> 'CBT8901'."""
    return _SKU_LIKE.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}".upper(), text)


def extract_sku_tokens(text: str) -> List[str]:
    """SKU-shaped tokens in order of first appearance, de-duplicated."""
    seen: List[str] = []
    for token in _SKU_TOKEN.findall(normalize_sku_text(text)):
        if token not in seen:
            seen.append(token)
    return seen


def canonical_sku(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()


def best_sku_match(token: str, candidate_ids: Iterable[str], threshold: float) -> Optional[Tuple[str, float]]:
    """Exact match wins; otherwise the closest catalog SKU by edit-distance ratio above ``threshold``."""
    token = canonical_sku(token)
    best: Optional[Tuple[str, float]] = None
    for candidate in candidate_ids:
        score = 100.0 if candidate == token else float(fuzz.ratio(token, candidate))
        if best is None or score > best[1]:
            best = (candidate, score)
    if best is not None and best[1] >= threshold:
        return best
    return None


@dataclass
class NameMention:
    product: Product
    score: float
    position: int


def find_name_mentions(text: str, products: Sequence[Product], threshold: float) -> List[NameMention]:
    """
    Products whose name appears in ``text``. Multi-word names may match fuzzily
    ('retro sun glasses' ~ 'Retro Sunglasses'); single-word names must match exactly.
    """
    haystack = f" {normalize_phrase(text)} "
    mentions: List[NameMention] = []
    for product in products:
        needle = normalize_phrase(product.name)
        if not needle:
            continue
        position = haystack.find(f" {needle} ")
        if position >= 0:
            mentions.append(NameMention(product, 100.0, position))
            continue
        if " " not in needle:
            continue
        alignment = fuzz.partial_ratio_alignment(needle, haystack)
        if alignment is not None and alignment.score >= threshold:
            mentions.append(NameMention(product, float(alignment.score), alignment.dest_start))
    mentions.sort(key=lambda m: (-m.score, m.position))
    return mentions
