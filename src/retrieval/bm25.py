"""
Okapi BM25 lexical scoring shared by both storage backends.

The in-memory backend scores the whole catalog. The PostgreSQL backend uses its
GIN full-text index only to generate candidates and then applies exactly this
scoring with corpus statistics persisted at catalog-sync time, so both backends
rank lexical matches identically.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.retrieval.text import tokenize

K1 = 1.5
B = 0.75


@dataclass
class CorpusStats:
    documents: int = 0
    avg_length: float = 0.0
    document_frequency: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_documents(cls, documents: Iterable[str]) -> CorpusStats:
        df: Counter[str] = Counter()
        lengths: List[int] = []
        for doc in documents:
            terms = tokenize(doc)
            lengths.append(len(terms))
            df.update(set(terms))
        n = len(lengths)
        return cls(n, (sum(lengths) / n) if n else 0.0, dict(df))

    def idf(self, term: str) -> float:
        freq = self.document_frequency.get(term, 0)
        return math.log(1 + (self.documents - freq + 0.5) / (freq + 0.5))

    def to_dict(self) -> Dict[str, object]:
        return {"documents": self.documents, "avg_length": self.avg_length, "df": self.document_frequency}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CorpusStats:
        return cls(int(data["documents"]), float(data["avg_length"]), dict(data["df"]))


def rank(query: str, docs: Sequence[Tuple[str, str]], stats: CorpusStats, limit: int) -> List[Tuple[str, float]]:
    """Scores (doc_id, text) pairs for ``query``; returns the best ``limit`` with a positive score."""
    terms = set(tokenize(query))
    if not terms:
        return []
    scored: List[Tuple[str, float]] = []
    for doc_id, text in docs:
        tokens = tokenize(text)
        tf = Counter(tokens)
        norm_len = K1 * (1 - B + B * len(tokens) / (stats.avg_length or 1.0))
        score = sum(stats.idf(t) * tf[t] * (K1 + 1) / (tf[t] + norm_len) for t in terms if t in tf)
        if score > 0:
            scored.append((doc_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]


class BM25Index:
    """Whole-corpus BM25 used by the in-memory store."""

    def __init__(self, doc_ids: Sequence[str], documents: Sequence[str]):
        self._docs = list(zip(doc_ids, documents, strict=False))
        self.stats = CorpusStats.from_documents(documents)

    def search(self, query: str, limit: int) -> List[Tuple[str, float]]:
        return rank(query, self._docs, self.stats, limit)
