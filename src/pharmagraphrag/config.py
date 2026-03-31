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

# ---------------------------------------------------------------------------
# Available LLM models — grouped by capability tier
# ---------------------------------------------------------------------------

FLASH_MODELS: list[dict[str, str]] = [
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash (preview)"},
    {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite"},
]

PRO_MODELS: list[dict[str, str]] = [
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro (preview)"},
]

DEFAULT_MODEL = "gemini-2.5-flash"


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

    # --- ChromaDB ---
    chroma_persist_dir: str = "./data/chroma"

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
