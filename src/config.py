"""
Centralized, validated configuration for CommerceOps AI Engine.

Every value comes from environment variables (or a local ``.env`` file). Components
receive a ``Settings`` instance explicitly, so tests and the API run with isolated
configurations.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal, Optional
from urllib.parse import quote

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_PLACEHOLDER_KEYS = {"", "your_openai_api_key_here", "sk-...", "changeme"}


class LLMMode(StrEnum):
    LIVE = "live"  # call OpenAI
    RECORD = "record"  # call OpenAI and save every exchange to REPLAY_FIXTURES_PATH
    REPLAY = "replay"  # serve previously recorded exchanges (deterministic, no network)


class Settings(BaseSettings):
    # Blank values (e.g. "LLM_TEMPERATURE=" in .env) fall back to the defaults below.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)

    APP_NAME: str = "CommerceOps AI Engine"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"

    # LLM
    LLM_MODE: LLMMode = LLMMode.LIVE
    OPENAI_API_KEY: SecretStr = SecretStr("")
    OPENAI_BASE_URL: Optional[str] = None
    LLM_MODEL: str = "gpt-6-luna"
    LLM_TEMPERATURE: Optional[float] = Field(default=None, description="Only sent when set.")
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 3
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536  # must match the vector(1536) column
    REPLAY_FIXTURES_PATH: str = "fixtures/llm_recordings.json"
    LLM_PRICE_INPUT_PER_1M: Optional[float] = None
    LLM_PRICE_OUTPUT_PER_1M: Optional[float] = None

    # Agent budgets and policy
    AGENT_MAX_STEPS: int = Field(default=12, ge=2, le=50, description="Max executor turns per email")
    AGENT_MAX_REPLANS: int = Field(default=2, ge=0, le=10)
    AGENT_MAX_ORDER_QUANTITY: int = Field(default=50, ge=1, description="Larger requests go to human review")
    SEARCH_RESULT_LIMIT: int = Field(default=8, ge=1, le=25)

    # SQL sandbox
    SQL_STATEMENT_TIMEOUT_MS: int = Field(default=2000, ge=100)
    SQL_MAX_ROWS: int = Field(default=50, ge=1, le=1000)
    SQL_MAX_CELL_CHARS: int = Field(default=200, ge=10)

    # PostgreSQL
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

    # Batch processing. 1 keeps first-come-first-served stock allocation and reproducible runs;
    # higher values are safe (row locks prevent overselling) but allocation order becomes nondeterministic.
    BATCH_WORKERS: int = Field(default=1, ge=1, le=64)

    # Brand identity used in prompts and the reply signature
    BRAND_NAME: str = "Aura Fashion Atelier"
    AGENT_NAME: str = "The Aura Concierge Team"
    SUPPORT_EMAIL: str = "concierge@aurafashion.com"
    PHONE_NUMBER: str = "+1 (800) 555-2872"

    # Batch input/output
    CATALOG_CSV_PATH: str = "data/catalog_products.csv"
    INBOUND_EMAILS_CSV_PATH: str = "data/inbound_customer_emails.csv"
    EXPORT_DIR: str = "exports"

    # HTTP API
    API_KEY: Optional[SecretStr] = None
    API_MAX_BATCH: int = 50

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

    def validate_llm_mode(self) -> None:
        if self.LLM_MODE in (LLMMode.LIVE, LLMMode.RECORD) and not self.has_openai_key:
            raise ValueError(
                f"LLM_MODE={self.LLM_MODE.value} needs OPENAI_API_KEY. "
                "Set it in .env, or use LLM_MODE=replay with a recordings file."
            )

    @property
    def signature_block(self) -> str:
        return f"{self.AGENT_NAME}\n{self.BRAND_NAME}\n{self.SUPPORT_EMAIL} | {self.PHONE_NUMBER}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
