"""LLM client implementations and factory."""

from __future__ import annotations

from typing import Optional

from src.config import LLMMode, Settings
from src.llm.base import LLMClient, LLMError, LLMResult, LLMUsage, ReplayMissError
from src.utils.logger import get_logger

logger = get_logger("llm")

__all__ = ["LLMClient", "LLMError", "LLMResult", "LLMUsage", "ReplayMissError", "build_llm_client"]


def build_llm_client(settings: Settings, mode: Optional[LLMMode] = None) -> Optional[LLMClient]:
    """Returns the client for the resolved mode, or None for offline (rules-only) operation."""
    from src.llm.openai_client import OpenAILLMClient
    from src.llm.replay import RecordingLLMClient, ReplayLLMClient

    resolved = mode or settings.resolved_llm_mode()
    if resolved == LLMMode.OFFLINE:
        logger.info("LLM mode: offline (deterministic rules and templates, no model calls)")
        return None
    if resolved == LLMMode.REPLAY:
        replay = ReplayLLMClient(settings.REPLAY_FIXTURES_PATH, settings.EMBEDDING_DIMENSIONS)
        logger.info(
            "LLM mode: replay (%d %s answers for model %s from %s)",
            len(replay),
            replay.source,
            replay.model,
            settings.REPLAY_FIXTURES_PATH,
        )
        return replay
    live = OpenAILLMClient(settings)
    if resolved == LLMMode.RECORD:
        logger.info("LLM mode: record (live %s, saving to %s)", live.model, settings.REPLAY_FIXTURES_PATH)
        return RecordingLLMClient(live, settings.REPLAY_FIXTURES_PATH)
    logger.info("LLM mode: live (%s)", live.model)
    return live
