"""
Shared text normalization used by lexical search, name matching and local embeddings.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

_STOPWORDS_TEXT = """
a about above after again all also am an and any are as at be been before being below between both
but by can could did do does doing down during each few for from further had has have having he her
here hers him his how i i'd i'll i'm i've if in into is it its itself just let me more most my no
nor not now of off on once only or other our ours out over own please same she should so some such
than thank thanks that the their them then there these they this those through to too under until up
us very was we were what when where which while who whom why will with would you your yours hi hello
hey dear regards
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())  # noqa: SIM905 - readable word list

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def fold(text: str) -> str:
    """Lowercases and strips accents (so 'Está' and 'esta' compare equal)."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def stem(token: str) -> str:
    """Very light plural stemming, applied identically to queries and catalog text."""
    if len(token) <= 3 or token.isdigit():
        return token
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("ves"):
        return token[:-3] + "f"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    tokens = _TOKEN_RE.findall(fold(text))
    return [stem(t) for t in tokens if not (drop_stopwords and t in STOPWORDS)]


def normalize_phrase(text: str) -> str:
    """Normalized form used for product-name matching: folded, punctuation-free, stemmed words."""
    return " ".join(tokenize(text, drop_stopwords=False))
