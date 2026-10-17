"""
OpenAI implementation: Structured Outputs for planning, strict function calling for
execution, and embeddings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import openai
from openai import OpenAI

from src.config import Settings
from src.llm.base import AssistantTurn, LLMError, LLMResult, LLMUsage, SchemaT, ToolCall

_EMBED_BATCH = 256


def _usage(response: Any) -> LLMUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMUsage()
    return LLMUsage(int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0))


class OpenAILLMClient:
    def __init__(self, settings: Settings, client: Optional[OpenAI] = None):
        self.settings = settings
        self.model = settings.LLM_MODEL
        self.embedding_model = f"openai:{settings.EMBEDDING_MODEL}"
        self._client = client or OpenAI(
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,  # SDK retries 429/5xx/timeouts with exponential backoff
        )

    def _sampling(self) -> Dict[str, Any]:
        return {} if self.settings.LLM_TEMPERATURE is None else {"temperature": self.settings.LLM_TEMPERATURE}

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]:
        try:
            response = self._client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format=schema,
                **self._sampling(),
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"{task}: {type(exc).__name__}: {exc}") from exc
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMError(f"{task}: model refused: {message.refusal}")
        if message.parsed is None:
            raise LLMError(f"{task}: empty structured output")
        return LLMResult(message.parsed, _usage(response))

    def chat(
        self, *, task: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> LLMResult[AssistantTurn]:
        try:
            response = self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="required",  # every turn must act; the run ends via the submit_reply tool
                parallel_tool_calls=False,  # one tool per turn keeps traces and permission checks simple
                **self._sampling(),
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"{task}: {type(exc).__name__}: {exc}") from exc
        message = response.choices[0].message
        calls = [
            ToolCall(c.id, c.function.name, c.function.arguments)
            for c in (message.tool_calls or [])
            if getattr(c, "function", None) is not None
        ]
        return LLMResult(AssistantTurn(message.content, calls), _usage(response))

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = [t.replace("\n", " ") for t in texts[start : start + _EMBED_BATCH]]
            try:
                response = self._client.embeddings.create(model=self.settings.EMBEDDING_MODEL, input=batch)
            except openai.OpenAIError as exc:
                raise LLMError(f"embeddings: {type(exc).__name__}: {exc}") from exc
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        for vector in vectors:
            if len(vector) != self.settings.EMBEDDING_DIMENSIONS:
                raise LLMError(
                    f"Embedding model returned {len(vector)} dimensions; "
                    f"EMBEDDING_DIMENSIONS is {self.settings.EMBEDDING_DIMENSIONS}."
                )
        return vectors
