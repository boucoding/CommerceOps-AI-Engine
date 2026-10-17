"""
Builds fixtures/llm_replays.json from the hand-written answers in
fixtures/reference_answers.json.

The sample emails are processed by the real pipeline. Whenever a stage calls the
model, the matching hand-written answer (by email ID and task) is returned and
recorded under the exact request fingerprint, so `LLM_MODE=replay` replays it
byte-for-byte. Re-run this after changing prompts, the catalog or retrieval:

    python -m scripts.build_reference_fixtures

To replace the reference answers with real model output, run `make record` instead.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Type

from src.config import LLMMode, StorageBackend, get_settings
from src.engine import build_engine
from src.llm.base import LLMError, LLMResult, SchemaT
from src.llm.replay import RecordingLLMClient
from src.loaders import load_emails
from src.retrieval.embeddings import LOCAL_EMBEDDING_MODEL, LocalHashEmbedder
from src.utils.logger import configure_logging

ANSWERS_PATH = "fixtures/reference_answers.json"
_EMAIL_ID = re.compile(r"^Email ID: (\S+)", re.MULTILINE)


class AuthoredAnswersClient:
    """Serves hand-written answers as if they came from the model."""

    def __init__(self, answers: Dict[str, Dict[str, Any]], model: str, dimensions: int):
        self.answers = answers
        self.model = model
        self.embedding_model = LOCAL_EMBEDDING_MODEL
        self._local = LocalHashEmbedder(dimensions)
        self.missing: List[str] = []

    def _answer(self, task: str, user: str) -> Any:
        match = _EMAIL_ID.search(user)
        email_id = match.group(1) if match else "?"
        try:
            return self.answers[email_id][task]
        except KeyError:
            self.missing.append(f"{email_id}/{task}")
            raise LLMError(f"no reference answer for {email_id}/{task}") from None

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]:
        return LLMResult(value=schema.model_validate(self._answer(task, user)))

    def complete(self, *, task: str, system: str, user: str) -> LLMResult[str]:
        return LLMResult(value=str(self._answer(task, user)))

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._local.embed(texts)


def main() -> int:
    configure_logging("WARNING")
    settings = get_settings().model_copy(update={"STORAGE_BACKEND": StorageBackend.MEMORY, "BATCH_WORKERS": 1})
    with open(ANSWERS_PATH, encoding="utf-8") as fh:
        reference = json.load(fh)
    authored = AuthoredAnswersClient(reference["answers"], reference["model"], settings.EMBEDDING_DIMENSIONS)
    recorder = RecordingLLMClient(authored, settings.REPLAY_FIXTURES_PATH, source="hand-authored")

    engine = build_engine(settings, mode=LLMMode.LIVE, llm=recorder)
    outcomes = engine.orchestrator.process_batch(load_emails(settings.INBOUND_EMAILS_CSV_PATH))

    problems = [f"missing answer: {m}" for m in authored.missing]
    problems += [f"{o.email_id}: {r}" for o in outcomes for r in o.degraded_reasons]
    if problems:
        print("Reference answers do not cover the pipeline cleanly:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    count = recorder.save()
    print(f"Wrote {count} replay entries to {settings.REPLAY_FIXTURES_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
