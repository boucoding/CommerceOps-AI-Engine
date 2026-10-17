"""
Shared fixtures. Tests never touch the network: LLM behaviour is scripted with
``ScriptedLLM`` or replayed from fixtures; PostgreSQL tests are opt-in via
TEST_DATABASE_URL.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pytest

from src.config import LLMMode, Settings, StorageBackend
from src.engine import Engine, build_engine
from src.llm.base import LLMError, LLMResult, LLMUsage, SchemaT
from src.models.domain import CustomerEmail, Product
from src.retrieval.embeddings import LOCAL_EMBEDDING_MODEL, LocalHashEmbedder

Answer = Union[Dict[str, Any], str, Exception, Callable[[str, str], Any]]


def make_settings(**overrides: Any) -> Settings:
    base: Dict[str, Any] = {
        "LLM_MODE": LLMMode.OFFLINE,
        "STORAGE_BACKEND": StorageBackend.MEMORY,
        "OPENAI_API_KEY": "",
        "BATCH_WORKERS": 1,
        "LOG_LEVEL": "WARNING",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


class ScriptedLLM:
    """Test double implementing the LLMClient protocol. Answers are looked up by task (and optionally email id)."""

    model = "scripted-model"
    embedding_model = LOCAL_EMBEDDING_MODEL

    def __init__(self, answers: Optional[Dict[str, Answer]] = None, dimensions: int = 1536):
        self.answers: Dict[str, Answer] = answers or {}
        self.calls: List[Tuple[str, str, str]] = []
        self._local = LocalHashEmbedder(dimensions)

    def _answer(self, task: str, system: str, user: str) -> Any:
        self.calls.append((task, system, user))
        email_id = user.split("\n", 1)[0].removeprefix("Email ID: ").strip()
        answer = self.answers.get(f"{task}:{email_id}", self.answers.get(task))
        if answer is None:
            raise LLMError(f"no scripted answer for {task}")
        if callable(answer):
            answer = answer(system, user)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def parse(self, *, task: str, system: str, user: str, schema: Type[SchemaT]) -> LLMResult[SchemaT]:
        return LLMResult(schema.model_validate(self._answer(task, system, user)), LLMUsage(100, 20))

    def complete(self, *, task: str, system: str, user: str) -> LLMResult[str]:
        return LLMResult(str(self._answer(task, system, user)), LLMUsage(200, 80))

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._local.embed(texts)

    def tasks(self) -> List[str]:
        return [c[0] for c in self.calls]


def intent(category: str, name: Optional[str] = None, language: str = "en", confidence: float = 0.9) -> Dict[str, Any]:
    return {
        "category": category,
        "confidence": confidence,
        "customer_name": name,
        "language": language,
        "reasoning": "test",
    }


def order_item(
    pid: Optional[str],
    qty: Optional[int] = 1,
    all_available: bool = False,
    active: bool = True,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "product_id": pid,
        "product_name": name or pid,
        "quantity": qty,
        "wants_all_available": all_available,
        "is_active_order": active,
    }


@pytest.fixture
def products() -> List[Product]:
    return [
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
            "SWL2345",
            "Sleek Wallet",
            "Accessories",
            "Minimalist wallet with multiple card slots.",
            5,
            "All seasons",
            30.0,
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
        Product(
            "CGN2345",
            "Cargo Pants",
            "Women's Clothing",
            "Utility cargo pants with multiple pockets.",
            2,
            "Spring, Summer, Fall",
            80.0,
        ),
        Product(
            "VSC6789",
            "Versatile Scarf",
            "Accessories",
            "Lightweight scarf, shawl or headwrap.",
            6,
            "Spring, Fall",
            23.0,
        ),
        Product(
            "SLD7654",
            "Slide Sandals",
            "Men's Shoes",
            "Lightweight slide sandals with a cushioned footbed.",
            3,
            "Spring, Summer",
            22.0,
        ),
    ]


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def build(products: List[Product]) -> Callable[..., Engine]:
    """Factory: build(llm=None, **setting_overrides) -> Engine on a fresh in-memory store."""

    def _build(llm: Any = None, **overrides: Any) -> Engine:
        mode = LLMMode.OFFLINE if llm is None else LLMMode.LIVE
        return build_engine(
            make_settings(**overrides), mode=mode, products=[Product(**vars(p)) for p in products], llm=llm
        )

    return _build


def email(email_id: str, body: str, subject: str = "") -> CustomerEmail:
    return CustomerEmail(email_id, subject, body)


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url
