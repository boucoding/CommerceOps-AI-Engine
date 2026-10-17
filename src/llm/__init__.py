"""LLM clients and factory."""

from __future__ import annotations

from src.config import LLMMode, Settings
from src.llm.base import AssistantTurn, LLMClient, LLMError, LLMResult, LLMUsage, ReplayMissError, ToolCall
from src.utils.logger import get_logger

logger = get_logger("llm")

__all__ = [
    "AssistantTurn",
    "LLMClient",
    "LLMError",
    "LLMResult",
    "LLMUsage",
    "ReplayMissError",
    "ToolCall",
    "build_llm_client",
]


def build_llm_client(settings: Settings) -> LLMClient:
    from src.llm.openai_client import OpenAILLMClient
    from src.llm.replay import RecordingLLMClient, ReplayLLMClient

    settings.validate_llm_mode()
    if settings.LLM_MODE == LLMMode.REPLAY:
        replay = ReplayLLMClient(settings.REPLAY_FIXTURES_PATH)
        logger.info(
            "LLM mode: replay (%d recorded exchanges of %s, recorded %s)", len(replay), replay.model, replay.recorded_at
        )
        return replay
    live = OpenAILLMClient(settings)
    if settings.LLM_MODE == LLMMode.RECORD:
        logger.info("LLM mode: record (%s, saving to %s)", live.model, settings.REPLAY_FIXTURES_PATH)
        return RecordingLLMClient(live, settings.REPLAY_FIXTURES_PATH)
    logger.info("LLM mode: live (%s)", live.model)
    return live
