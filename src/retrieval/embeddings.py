"""
Embeddings Generation Client for Vector Operations.
Supports OpenAI text-embedding-3-small with local deterministic fallback.
"""

import math
import hashlib
from typing import List, Optional
import numpy as np
from openai import OpenAI
from src.config import settings
from src.utils.logger import logger

class EmbeddingService:
    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client
        self.dimension = settings.EMBEDDING_DIMENSIONS

    def get_embedding(self, text: str) -> List[float]:
        """Generates 1536-dimensional embedding vector for input text."""
        cleaned_text = text.replace("\n", " ").strip()
        if not cleaned_text:
            return [0.0] * self.dimension

        if self.client and settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("<"):
            try:
                response = self.client.embeddings.create(
                    input=[cleaned_text],
                    model=settings.EMBEDDING_MODEL
                )
                return response.data[0].embedding
            except Exception as e:
                logger.warning(f"OpenAI embedding API failed: {e}. Using deterministic local embedding fallback.")

        return self._generate_deterministic_embedding(cleaned_text)

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates batch embeddings."""
        if not texts:
            return []

        if self.client and settings.OPENAI_API_KEY and not settings.OPENAI_API_KEY.startswith("<"):
            try:
                response = self.client.embeddings.create(
                    input=[t.replace("\n", " ") for t in texts],
                    model=settings.EMBEDDING_MODEL
                )
                return [d.embedding for d in response.data]
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}. Using local fallback.")

        return [self._generate_deterministic_embedding(t) for t in texts]

    def _generate_deterministic_embedding(self, text: str) -> List[float]:
        """
        Fast deterministic hashing-based unit vector for local testing and offline fallback.
        Preserves consistent dimensionality (1536) and cosine properties.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words):
            # Compute hash bucket
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            val = ((h >> 16) % 1000) / 500.0 - 1.0
            vec[idx] += val * (1.0 / (math.sqrt(i + 1)))

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()
