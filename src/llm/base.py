"""
Provider-neutral LLM interface used by every pipeline stage.

Implementations: ``OpenAILLMClient`` (live), ``ReplayLLMClient`` (deterministic
fixtures) and ``RecordingLLMClient`` (live + capture). Pipeline code depends
only on this protocol, which keeps it testable without network access.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Generic, List, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T")
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure to obtain a usable answer from the model."""


class ReplayMissError(LLMError):
    """The replay fixture file has no answer recorded for this exact request."""


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMResult(Generic[T]):
    value: T
    usage: LLMUsage = field(default_factory=LLMUsage)
    replayed: bool = False


class LLMClient(Protocol):
    model: str
    embedding_model: str

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]: ...

    def complete(self, *, task: str, system: str, user: str) -> LLMResult[str]: ...

    def embed(self, texts: List[str]) -> List[List[float]]: ...


def request_key(model: str, task: str, system: str, user: str, schema: Optional[str]) -> str:
    """Stable fingerprint of a request. Any change to model, prompt or input changes the key."""
    payload = json.dumps([model, task, system, user, schema], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()
