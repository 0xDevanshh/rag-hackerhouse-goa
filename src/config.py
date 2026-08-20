"""
Central configuration for the voice-rag pipeline.

Loads settings (API keys, model names, chunking parameters, index paths, etc.)
from environment variables / a .env file, and exposes them as a single typed
settings object that other modules import from.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """
    Typed application settings, populated from environment variables.

    Provider keys are optional at import time so /health remains available;
    generation readiness is checked separately by /ready and by the harness.
    """

    sarvam_api_key: str | None = None
    groq_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_provider: str = "groq"
    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    corpus_language: str = "hi"
    corpus_split: str = "validation"
    corpus_limit: int = Field(default=500, ge=1)
    corpus_include_english: bool = True
    rag_target_ms: float = Field(default=200.0, gt=0)
    allowed_origin: str = "http://localhost:3000"


def load_settings() -> Settings:
    """
    Load environment variables (via python-dotenv) and construct a Settings instance.

    Returns:
        Settings: the populated application configuration.
    """
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    return Settings(
        sarvam_api_key=os.getenv("SARVAM_API_KEY") or None,
        groq_api_key=os.getenv("GROQ_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        corpus_language=os.getenv("CORPUS_LANGUAGE", "hi"),
        corpus_split=os.getenv("CORPUS_SPLIT", "validation"),
        corpus_limit=int(os.getenv("CORPUS_LIMIT", "500")),
        corpus_include_english=os.getenv("CORPUS_INCLUDE_ENGLISH", "1") != "0",
        rag_target_ms=float(os.getenv("RAG_TARGET_MS", "200")),
        allowed_origin=os.getenv("ALLOWED_ORIGIN", "http://localhost:3000"),
    )
