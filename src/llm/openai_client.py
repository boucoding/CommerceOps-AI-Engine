"""
OpenAI implementation of the LLM interface (Structured Outputs + embeddings).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import openai
from openai import OpenAI

from src.config import Settings
from src.llm.base import LLMError, LLMResult, LLMUsage, SchemaT
from src.utils.logger import get_logger

logger = get_logger("llm.openai")

_EMBED_BATCH = 256


def _usage(response: Any) -> LLMUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return LLMUsage()
    return LLMUsage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )


class OpenAILLMClient:
    def __init__(self, settings: Settings, client: Optional[OpenAI] = None):
        self.settings = settings
        self.model = settings.LLM_MODEL
        self.embedding_model = f"openai:{settings.EMBEDDING_MODEL}"
        self._client = client or OpenAI(
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    def _sampling(self) -> Dict[str, Any]:
        if self.settings.LLM_TEMPERATURE is None:
            return {}
        return {"temperature": self.settings.LLM_TEMPERATURE}

    @staticmethod
    def _messages(system: str, user: str) -> List[Dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]:
        try:
            response = self._client.chat.completions.parse(
                model=self.model,
                messages=self._messages(system, user),  # type: ignore[arg-type]
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
        return LLMResult(value=message.parsed, usage=_usage(response))

    def complete(self, *, task: str, system: str, user: str) -> LLMResult[str]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages(system, user),  # type: ignore[arg-type]
                **self._sampling(),
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"{task}: {type(exc).__name__}: {exc}") from exc

        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise LLMError(f"{task}: empty completion")
        return LLMResult(value=content, usage=_usage(response))

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
