"""
Shared fixtures.

Tests never call a real model: ``ScriptedLLM`` plays back a scripted plan and
tool-call sequence per email. Everything that touches storage runs against a real
PostgreSQL + pgvector database given by TEST_DATABASE_URL (tests are skipped
without it). That database is wiped.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
import pytest

from src.config import LLMMode, Settings
from src.engine import Engine, build_engine
from src.llm.base import AssistantTurn, LLMError, LLMResult, LLMUsage, SchemaT, ToolCall
from src.models.domain import CustomerEmail, Product
from src.storage.db import Database

Step = Union[Tuple[str, Dict[str, Any]], Exception, Callable[[List[Dict[str, Any]]], Tuple[str, Dict[str, Any]]]]


def hash_embedding(text: str, dims: int = 1536) -> List[float]:
    """Deterministic bag-of-words vector so similar texts get similar embeddings (test-only)."""
    vec = np.zeros(dims, dtype=np.float32)
    for word in text.lower().replace(",", " ").replace(".", " ").split():
        word = word.rstrip("s")
        digest = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
        vec[digest % dims] += 1.0
    norm = float(np.linalg.norm(vec)) or 1.0
    return (vec / norm).tolist()


class ScriptedLLM:
    """LLMClient test double. Plans and replans come from dicts; execution turns from per-email step lists."""

    model = "scripted-model"
    embedding_model = "test:hash"

    def __init__(
        self,
        plans: Optional[Dict[str, Any]] = None,
        steps: Optional[Dict[str, List[Step]]] = None,
        replans: Optional[Dict[str, List[Any]]] = None,
    ):
        self.plans = plans or {}
        self.steps = {k: list(v) for k, v in (steps or {}).items()}
        self.replans = {k: list(v) for k, v in (replans or {}).items()}
        self.calls: List[Tuple[str, str]] = []
        self._ids = itertools.count(1)

    @staticmethod
    def _email_id(text: str) -> str:
        return text.split("\n", 1)[0].removeprefix("Email ID: ").strip()

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]:
        email_id = self._email_id(user)
        self.calls.append((task, email_id))
        if task == "plan":
            answer = self.plans.get(email_id)
        else:
            queue = self.replans.get(email_id) or []
            answer = queue.pop(0) if queue else None
        if answer is None:
            raise LLMError(f"no scripted {task} for {email_id}")
        if isinstance(answer, Exception):
            raise answer
        return LLMResult(schema.model_validate(answer), LLMUsage(120, 40))

    def chat(
        self, *, task: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> LLMResult[AssistantTurn]:
        email_id = self._email_id(messages[1]["content"])
        self.calls.append((task, email_id))
        queue = self.steps.get(email_id) or []
        if not queue:
            raise LLMError(f"script for {email_id} is exhausted")
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        name, args = step(messages) if callable(step) else step
        call = ToolCall(f"call_{next(self._ids)}", name, json.dumps(args))
        return LLMResult(AssistantTurn(None, [call]), LLMUsage(300, 60))

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [hash_embedding(t) for t in texts]

    def tasks(self, email_id: str) -> List[str]:
        return [task for task, eid in self.calls if eid == email_id]


def plan(intent: str, *tools: str, name: Optional[str] = None, language: str = "en") -> Dict[str, Any]:
    return {
        "intent": intent,
        "customer_name": name,
        "language": language,
        "summary": "test",
        "steps": [{"tool": t, "purpose": f"use {t}"} for t in tools],
        "success_criteria": "reply sent",
    }


def last_tool_result(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    return json.loads(next(m for m in reversed(messages) if m["role"] == "tool")["content"])


def email(email_id: str, body: str, subject: str = "") -> CustomerEmail:
    return CustomerEmail(email_id, subject, body)


def make_settings(**overrides: Any) -> Settings:
    base: Dict[str, Any] = {
        "LLM_MODE": LLMMode.LIVE,
        "OPENAI_API_KEY": "",
        "BATCH_WORKERS": 1,
        "LOG_LEVEL": "WARNING",
        "DB_CONNECT_RETRIES": 1,
        "DB_POOL_MAX": 30,
    }
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        base["DATABASE_URL"] = url
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


PRODUCTS = [
    Product(
        "RSG8901",
        "Retro Sunglasses",
        "Accessories",
        "Vintage-inspired shades with a nostalgic vibe.",
        1,
        "Spring, Summer",
        26.99,
    ),
    Product(
        "SWL2345", "Sleek Wallet", "Accessories", "Minimalist leather wallet with card slots.", 5, "All seasons", 30.0
    ),
    Product(
        "CHN0987",
        "Chunky Knit Beanie",
        "Accessories",
        "Thick cozy yarn beanie for winter warmth.",
        2,
        "Fall, Winter",
        22.0,
    ),
    Product(
        "CLF2109", "Cable Knit Beanie", "Accessories", "Classic cable knit beanie in soft wool.", 2, "Winter", 16.0
    ),
    Product("FZZ1098", "Fuzzy Slippers", "Accessories", "Plush fuzzy slippers for lounging.", 2, "Fall, Winter", 29.0),
    Product(
        "CBT8901",
        "Chelsea Boots",
        "Men's Shoes",
        "Leather ankle boots with elastic panels and lug sole.",
        2,
        "Fall, Winter",
        65.0,
    ),
    Product(
        "LTH1098",
        "Leather Backpack",
        "Bags",
        "Leather backpack with a padded laptop sleeve and compartments.",
        7,
        "All seasons",
        43.99,
    ),
    Product(
        "LTH5432",
        "Leather Tote",
        "Bags",
        "Full-grain leather tote with a spacious interior and pockets.",
        6,
        "All seasons",
        28.0,
    ),
]


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


@pytest.fixture(scope="session")
def clean_database(database_url: str) -> str:
    db = Database(make_settings())
    with db.cursor() as cur:
        cur.execute(
            "DROP TABLE IF EXISTS sql_audit, order_lines, emails, products, lexical_stats, schema_migrations CASCADE"
        )
        cur.execute("DROP VIEW IF EXISTS product_catalog, order_history")
    db.close()
    return database_url


@pytest.fixture
def make_engine(clean_database: str) -> Iterator[Callable[..., Engine]]:
    """Factory: make_engine(llm, **settings) -> Engine on a reset database with the test catalog."""
    engines: List[Engine] = []

    def _make(llm: Optional[ScriptedLLM] = None, **overrides: Any) -> Engine:
        engine = build_engine(
            make_settings(**overrides),
            reset=True,
            products=[Product(**vars(p)) for p in PRODUCTS],
            llm=llm or ScriptedLLM(),
        )
        engines.append(engine)
        return engine

    yield _make
    for engine in engines:
        engine.store.close()
