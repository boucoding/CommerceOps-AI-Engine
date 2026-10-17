"""
Centralized, validated configuration for CommerceOps AI Engine.

All values are read from environment variables (or a local ``.env`` file).
Components receive a ``Settings`` instance explicitly so tests and the API can
run with isolated configurations.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Values shipped in example files that must never be treated as a real key.
_PLACEHOLDER_KEYS = {"", "your_openai_api_key_here", "sk-...", "changeme", "dummy-key"}


class LLMMode(StrEnum):
    """How the engine talks to the language model."""

    AUTO = "auto"  # live when an API key is configured, otherwise replay, otherwise offline
    LIVE = "live"  # real OpenAI calls
    RECORD = "record"  # real OpenAI calls, saved to the replay fixture file
    REPLAY = "replay"  # deterministic answers from the replay fixture file
    OFFLINE = "offline"  # no LLM at all; deterministic rules and templates only


class StorageBackend(StrEnum):
    MEMORY = "memory"
    POSTGRES = "postgres"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    APP_NAME: str = "CommerceOps AI Engine"
    ENV: str = Field(default="development", description="development | staging | production")
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"

    # LLM
    LLM_MODE: LLMMode = LLMMode.AUTO
    OPENAI_API_KEY: SecretStr = SecretStr("")
    OPENAI_BASE_URL: Optional[str] = None
    LLM_MODEL: str = "gpt-6-luna"
    LLM_TEMPERATURE: Optional[float] = Field(
        default=None, description="Omitted from requests when unset (some models only accept the default)."
    )
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 3
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    REPLAY_FIXTURES_PATH: str = "fixtures/llm_replays.json"
    LLM_PRICE_INPUT_PER_1M: Optional[float] = Field(default=None, description="USD per 1M prompt tokens")
    LLM_PRICE_OUTPUT_PER_1M: Optional[float] = Field(default=None, description="USD per 1M completion tokens")

    # Storage
    STORAGE_BACKEND: StorageBackend = StorageBackend.MEMORY
    DATABASE_URL: Optional[str] = None
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "commerceops"
    POSTGRES_USER: str = "commerceops"
    POSTGRES_PASSWORD: SecretStr = SecretStr("")
    DB_POOL_MIN: int = 1
    DB_POOL_MAX: int = 10
    DB_CONNECT_TIMEOUT_SECONDS: int = 5
    DB_CONNECT_RETRIES: int = 10

    # Retrieval
    ORDER_CANDIDATE_TOP_K: int = Field(default=5, ge=1, le=50)
    INQUIRY_TOP_K: int = Field(default=4, ge=1, le=50)
    SKU_FUZZY_THRESHOLD: float = Field(default=80.0, ge=0, le=100)
    NAME_MATCH_THRESHOLD: float = Field(default=88.0, ge=0, le=100)
    RRF_K: int = 60

    # Processing
    BATCH_WORKERS: int = Field(default=4, ge=1, le=64)

    # Brand identity used in prompts, templates and signatures
    BRAND_NAME: str = "Aura Fashion Atelier"
    AGENT_NAME: str = "The Aura Concierge Team"
    SUPPORT_EMAIL: str = "concierge@aurafashion.com"
    PHONE_NUMBER: str = "+1 (800) 555-2872"

    # Batch input/output
    CATALOG_CSV_PATH: str = "data/catalog_products.csv"
    INBOUND_EMAILS_CSV_PATH: str = "data/inbound_customer_emails.csv"
    EXPORT_DIR: str = "exports"

    # HTTP API
    API_KEY: Optional[SecretStr] = Field(default=None, description="When set, requests must send X-API-Key")
    API_MAX_BATCH: int = 100

    @field_validator("OPENAI_BASE_URL", "DATABASE_URL", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def has_openai_key(self) -> bool:
        key = self.OPENAI_API_KEY.get_secret_value().strip()
        return key not in _PLACEHOLDER_KEYS and not key.startswith("<")

    @property
    def database_dsn(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        password = quote(self.POSTGRES_PASSWORD.get_secret_value(), safe="")
        auth = f"{self.POSTGRES_USER}:{password}" if password else self.POSTGRES_USER
        return f"postgresql://{auth}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    def resolved_llm_mode(self) -> LLMMode:
        """Resolves AUTO into a concrete mode and validates the explicit ones."""
        if self.LLM_MODE in (LLMMode.LIVE, LLMMode.RECORD) and not self.has_openai_key:
            raise ValueError(f"LLM_MODE={self.LLM_MODE.value} requires a real OPENAI_API_KEY.")
        if self.LLM_MODE == LLMMode.REPLAY and not Path(self.REPLAY_FIXTURES_PATH).exists():
            raise ValueError(f"LLM_MODE=replay but {self.REPLAY_FIXTURES_PATH} does not exist.")
        if self.LLM_MODE != LLMMode.AUTO:
            return self.LLM_MODE
        if self.has_openai_key:
            return LLMMode.LIVE
        if Path(self.REPLAY_FIXTURES_PATH).exists():
            return LLMMode.REPLAY
        return LLMMode.OFFLINE

    @property
    def signature_block(self) -> str:
        return f"{self.AGENT_NAME}\n{self.BRAND_NAME}\n{self.SUPPORT_EMAIL} | {self.PHONE_NUMBER}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
