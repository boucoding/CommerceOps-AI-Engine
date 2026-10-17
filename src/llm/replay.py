"""
Deterministic replay and recording of LLM traffic.

A fixture file maps the SHA-256 fingerprint of each request (model + task +
system prompt + user prompt + schema) to the answer. Replays are strict: any
change to a prompt, the catalog context or the input email produces a different
fingerprint and therefore a ``ReplayMissError``. The pipeline never fabricates
an answer for a miss; it degrades to deterministic rules/templates and records why.

Fixture ``source`` is either ``recorded`` (captured from the real model with
``make record``) or ``hand-authored`` (reference answers written by a person and
bound to the exact prompts by ``scripts/build_reference_fixtures.py``).
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from src.llm.base import LLMClient, LLMResult, LLMUsage, ReplayMissError, SchemaT, request_key, text_key
from src.retrieval.embeddings import LOCAL_EMBEDDING_MODEL, LocalHashEmbedder

FIXTURE_VERSION = 1
_EMAIL_ID_RE = re.compile(r"^Email ID: (\S+)", re.MULTILINE)


def _email_id(user: str) -> Optional[str]:
    match = _EMAIL_ID_RE.search(user)
    return match.group(1) if match else None


def _encode_vector(vector: List[float]) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def _decode_vector(data: str) -> List[float]:
    return np.frombuffer(base64.b64decode(data), dtype=np.float32).astype(float).tolist()


class ReplayLLMClient:
    def __init__(self, path: str, dimensions: int):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != FIXTURE_VERSION:
            raise ValueError(f"Unsupported replay fixture version in {path}")
        self.path = path
        self.source: str = data.get("source", "unknown")
        self.model: str = data["model"]
        self.embedding_model: str = data["embedding_model"]
        self._responses: Dict[str, Dict[str, Any]] = data.get("responses", {})
        self._embeddings: Dict[str, str] = data.get("embeddings", {})
        self._local = LocalHashEmbedder(dimensions)

    def __len__(self) -> int:
        return len(self._responses)

    def _lookup(self, task: str, system: str, user: str, schema: Optional[str]) -> Dict[str, Any]:
        key = request_key(self.model, task, system, user, schema)
        entry = self._responses.get(key)
        if entry is None:
            raise ReplayMissError(f"{task}: no replay recorded for email {_email_id(user) or '?'}")
        return entry

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]:
        entry = self._lookup(task, system, user, schema.__name__)
        return LLMResult(
            value=schema.model_validate(entry["output"]), usage=LLMUsage(**entry.get("usage", {})), replayed=True
        )

    def complete(self, *, task: str, system: str, user: str) -> LLMResult[str]:
        entry = self._lookup(task, system, user, None)
        return LLMResult(value=str(entry["output"]), usage=LLMUsage(**entry.get("usage", {})), replayed=True)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.embedding_model == LOCAL_EMBEDDING_MODEL:
            return self._local.embed(texts)
        vectors = []
        for text in texts:
            encoded = self._embeddings.get(text_key(self.embedding_model, text))
            if encoded is None:
                raise ReplayMissError("embeddings: text was not embedded during recording")
            vectors.append(_decode_vector(encoded))
        return vectors


class RecordingLLMClient:
    """Delegates to a live client and captures every answer for later replay."""

    def __init__(self, inner: LLMClient, path: str, source: str = "recorded"):
        self.inner = inner
        self.path = path
        self.source = source
        self.model = inner.model
        self.embedding_model = inner.embedding_model
        self._responses: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _store(self, key: str, task: str, user: str, output: Any, usage: LLMUsage) -> None:
        with self._lock:
            self._responses[key] = {
                "task": task,
                "email_id": _email_id(user),
                "output": output,
                "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
            }

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]:
        result = self.inner.parse(task=task, system=system, user=user, schema=schema)
        key = request_key(self.model, task, system, user, schema.__name__)
        self._store(key, task, user, result.value.model_dump(mode="json"), result.usage)
        return result

    def complete(self, *, task: str, system: str, user: str) -> LLMResult[str]:
        result = self.inner.complete(task=task, system=system, user=user)
        self._store(request_key(self.model, task, system, user, None), task, user, result.value, result.usage)
        return result

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.inner.embed(texts)
        if self.embedding_model != LOCAL_EMBEDDING_MODEL:
            with self._lock:
                for text, vector in zip(texts, vectors, strict=False):
                    self._embeddings[text_key(self.embedding_model, text)] = _encode_vector(vector)
        return vectors

    def save(self) -> int:
        with self._lock:
            payload = {
                "version": FIXTURE_VERSION,
                "source": self.source,
                "model": self.model,
                "embedding_model": self.embedding_model,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "responses": dict(
                    sorted(self._responses.items(), key=lambda kv: (kv[1]["email_id"] or "", kv[1]["task"]))
                ),
                "embeddings": dict(sorted(self._embeddings.items())),
            }
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; fixtures must be readable by other users (e.g. containers)
        os.replace(tmp, target)
        return len(self._responses)
