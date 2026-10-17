"""
Embedding generation with a content-addressed cache.

Live mode uses the configured OpenAI embedding model. Offline mode (and the
bundled hand-authored replay fixtures) use ``LocalHashEmbedder``: a deterministic
feature-hashing vectorizer. It is lexical rather than semantic, and is labelled
as such everywhere it is used.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Dict, List, Optional

import numpy as np

from src.llm.base import LLMClient, text_key
from src.retrieval.text import tokenize

LOCAL_EMBEDDING_MODEL = "local:hash-v1"


class LocalHashEmbedder:
    embedding_model = LOCAL_EMBEDDING_MODEL

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for feature in features:
            digest = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
            sign = 1.0 if digest & 1 else -1.0
            weight = 0.5 if " " in feature else 1.0
            vec[(digest >> 1) % self.dimensions] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec.tolist()
        return (vec / norm).tolist()


class EmbeddingService:
    """Embeds text through the active LLM client (or locally) and caches results by content."""

    def __init__(self, llm: Optional[LLMClient], dimensions: int):
        self._local = LocalHashEmbedder(dimensions)
        self._llm = llm
        self.model = llm.embedding_model if llm is not None else LOCAL_EMBEDDING_MODEL
        self._cache: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def embed(self, texts: List[str]) -> List[List[float]]:
        keys = [text_key(self.model, t) for t in texts]
        with self._lock:
            missing = [(k, t) for k, t in zip(keys, texts, strict=False) if k not in self._cache]
        if missing:
            unique = dict(missing)
            batch_texts = list(unique.values())
            if self._llm is None or self.model == LOCAL_EMBEDDING_MODEL:
                vectors = self._local.embed(batch_texts)
            else:
                vectors = self._llm.embed(batch_texts)
            with self._lock:
                self._cache.update(zip(unique.keys(), vectors, strict=False))
        with self._lock:
            return [self._cache[k] for k in keys]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
