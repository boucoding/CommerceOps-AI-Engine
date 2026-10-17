"""
Exact cosine ranking shared by both storage backends.

The in-memory backend ranks the whole catalog; the PostgreSQL backend ranks the
candidates returned by its HNSW index. Scores are rounded before sorting and ties
are broken by product ID, so both backends produce the same order.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def rank_by_cosine(ids: Sequence[str], normalized: np.ndarray, query: Sequence[float], limit: int) -> List[str]:
    if not len(ids):
        return []
    q = np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return []
    scores = np.round((normalized @ (q / norm)).astype(np.float64), 6)
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    return [ids[i] for i in order[:limit]]
