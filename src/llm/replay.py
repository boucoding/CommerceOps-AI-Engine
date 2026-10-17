"""
Record and replay of real LLM traffic.

``make record`` runs the live model and saves every exchange keyed by a SHA-256
fingerprint of the full request (model, task, messages, tools/schema). Replay is
strict: if a prompt, tool definition, catalog result or email changes, the
fingerprint changes and the call fails with ``ReplayMissError``. A replay never
returns an answer that was given to a different question.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Type

import numpy as np

from src.llm.base import (
    AssistantTurn,
    LLMClient,
    LLMResult,
    LLMUsage,
    ReplayMissError,
    SchemaT,
    request_key,
    text_key,
)

FIXTURE_VERSION = 2


def _encode(vector: List[float]) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def _decode(data: str) -> List[float]:
    return np.frombuffer(base64.b64decode(data), dtype=np.float32).astype(float).tolist()


def _parse_payload(system: str, user: str, schema: str) -> Dict[str, Any]:
    return {"system": system, "user": user, "schema": schema}


def _chat_payload(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"messages": messages, "tools": tools}


class ReplayLLMClient:
    def __init__(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"No recordings at {path}. Run `make record` with an OPENAI_API_KEY first.")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") != FIXTURE_VERSION:
            raise ValueError(f"{path} was recorded with an incompatible version; re-run `make record`.")
        self.model: str = data["model"]
        self.embedding_model: str = data["embedding_model"]
        self.recorded_at: str = data.get("created_at", "")
        self._responses: Dict[str, Dict[str, Any]] = data["responses"]
        self._embeddings: Dict[str, str] = data["embeddings"]

    def __len__(self) -> int:
        return len(self._responses)

    def _lookup(self, task: str, payload: Any) -> Dict[str, Any]:
        entry = self._responses.get(request_key(self.model, task, payload))
        if entry is None:
            raise ReplayMissError(f"{task}: no recording for this exact request")
        return entry

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]:
        entry = self._lookup(task, _parse_payload(system, user, schema.__name__))
        return LLMResult(schema.model_validate(entry["output"]), LLMUsage(**entry["usage"]), replayed=True)

    def chat(
        self, *, task: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> LLMResult[AssistantTurn]:
        entry = self._lookup(task, _chat_payload(messages, tools))
        return LLMResult(AssistantTurn.from_dict(entry["output"]), LLMUsage(**entry["usage"]), replayed=True)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            encoded = self._embeddings.get(text_key(self.embedding_model, text))
            if encoded is None:
                raise ReplayMissError("embeddings: text was not embedded during recording")
            vectors.append(_decode(encoded))
        return vectors


class RecordingLLMClient:
    """Delegates to a live client and captures every exchange."""

    def __init__(self, inner: LLMClient, path: str):
        self.inner = inner
        self.path = path
        self.model = inner.model
        self.embedding_model = inner.embedding_model
        self._responses: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _store(self, task: str, payload: Any, output: Any, usage: LLMUsage) -> None:
        with self._lock:
            self._responses[request_key(self.model, task, payload)] = {
                "task": task,
                "output": output,
                "usage": {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens},
            }

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]:
        result = self.inner.parse(task=task, system=system, user=user, schema=schema)
        self._store(
            task, _parse_payload(system, user, schema.__name__), result.value.model_dump(mode="json"), result.usage
        )
        return result

    def chat(
        self, *, task: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> LLMResult[AssistantTurn]:
        result = self.inner.chat(task=task, messages=messages, tools=tools)
        self._store(task, _chat_payload(messages, tools), result.value.to_dict(), result.usage)
        return result

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.inner.embed(texts)
        with self._lock:
            for text, vector in zip(texts, vectors, strict=True):
                self._embeddings[text_key(self.embedding_model, text)] = _encode(vector)
        return vectors

    def save(self) -> int:
        with self._lock:
            payload = {
                "version": FIXTURE_VERSION,
                "model": self.model,
                "embedding_model": self.embedding_model,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "responses": dict(sorted(self._responses.items())),
                "embeddings": dict(sorted(self._embeddings.items())),
            }
            count = len(self._responses)
        target = Path(self.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; recordings must be readable by other users (e.g. containers)
        os.replace(tmp, target)
        return count
