"""
Provider-neutral LLM interface.

Two interaction styles are used by the agent:
- ``parse``: one structured-output call validated against a Pydantic schema (planning).
- ``chat``: one tool-calling turn (execution). The model must answer with a tool call.

Implementations: ``OpenAILLMClient`` (live), ``RecordingLLMClient`` (live + capture)
and ``ReplayLLMClient`` (deterministic playback of captured traffic).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMError(RuntimeError):
    """Any failure to obtain a usable answer from the model (network, refusal, invalid output, replay miss)."""


class ReplayMissError(LLMError):
    """The recordings file has no answer for this exact request."""


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMResult[T]:
    value: T
    usage: LLMUsage = field(default_factory=LLMUsage)
    replayed: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text exactly as produced by the model


@dataclass
class AssistantTurn:
    content: Optional[str]
    tool_calls: List[ToolCall]

    def to_message(self) -> Dict[str, Any]:
        """The assistant message to append to the conversation (OpenAI chat format)."""
        message: Dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return message

    def to_dict(self) -> Dict[str, Any]:
        return {"content": self.content, "tool_calls": [vars(c) for c in self.tool_calls]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AssistantTurn:
        return cls(data.get("content"), [ToolCall(**c) for c in data.get("tool_calls", [])])


class LLMClient(Protocol):
    model: str
    embedding_model: str

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]: ...

    def chat(
        self, *, task: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> LLMResult[AssistantTurn]: ...

    def embed(self, texts: List[str]) -> List[List[float]]: ...


def request_key(model: str, task: str, payload: Any) -> str:
    """Stable fingerprint of a request: any change to the model, prompt, tools or input changes it."""
    blob = json.dumps([model, task, payload], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def text_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()
