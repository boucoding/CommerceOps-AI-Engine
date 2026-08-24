"""
Centralized Configuration Settings for CommerceOps AI Engine.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Application & Environment
    APP_NAME: str = "CommerceOps AI Engine"
    ENV: str = Field(default="production", description="Environment: development, staging, production")
    LOG_LEVEL: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")

    # Database & Vector Store
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgrespassword@postgres-db:5432/commerceops",
        description="PostgreSQL Connection URL"
    )
    POSTGRES_HOST: str = Field(default="postgres-db")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="commerceops")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgrespassword")
    DB_POOL_MIN: int = Field(default=1)
    DB_POOL_MAX: int = Field(default=10)

    # LLM & Embedding Settings
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API Key loaded from environment variable OPENAI_API_KEY"
    )
    OPENAI_BASE_URL: Optional[str] = Field(
        default=None,
        description="Optional custom API Base URL loaded from OPENAI_BASE_URL"
    )
    LLM_MODEL: str = Field(default="gpt-4o")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = Field(default=1536)
    LLM_TEMPERATURE: float = Field(default=0.0)
    LLM_TIMEOUT_SECONDS: float = Field(default=30.0)

    # Retrieval & RAG Parameters
    RAG_TOP_K: int = Field(default=4, description="Number of products retrieved for inquiry responses")
    ORDER_CANDIDATE_TOP_K: int = Field(default=5, description="Number of candidate products for Stage 1 Order RAG")
    FUZZY_THRESHOLD: float = Field(default=75.0, description="Minimum score for fuzzy SKU matching")

    # Enterprise Brand Defaults
    BRAND_NAME: str = "Aura Fashion Atelier"
    BRAND_SIGNATURE: str = "Aura Customer Experience Concierge"
    SUPPORT_EMAIL: str = "concierge@aurafashion.com"
    PHONE_NUMBER: str = "+1 (800) 555-AURA"

    # Data & Export Paths
    CATALOG_CSV_PATH: str = "data/catalog_products.csv"
    INBOUND_EMAILS_CSV_PATH: str = "data/inbound_customer_emails.csv"
    EXCEL_OUTPUT_PATH: str = "exports/commerceops_operations_report.xlsx"
    CSV_OUTPUT_DIR: str = "exports/csv"
    
    # Legacy output fallbacks for backward compatibility
    LEGACY_EXCEL_PATH: str = "Solving_Business_Problems_Output.xlsx"
    LEGACY_CSV_DIR: str = "4 solution sheets"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
