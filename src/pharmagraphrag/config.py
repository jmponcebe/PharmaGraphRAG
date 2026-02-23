"""Centralized configuration using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "pharmagraphrag"

    # --- LLM ---
    gemini_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_provider: str = "gemini"  # "gemini" or "ollama"

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- App ---
    log_level: str = "INFO"

    # --- FAERS ---
    faers_quarters: list[str] = ["2024Q3", "2024Q4"]

    # --- DailyMed ---
    dailymed_top_n_drugs: int = 200


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
