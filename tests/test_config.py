"""Tests for typed environment configuration defaults."""

from src.config import Settings, load_settings


def test_settings_have_submission_safe_defaults():
    settings = Settings()
    assert settings.embedding_model_name == "paraphrase-multilingual-MiniLM-L12-v2"
    assert settings.corpus_language == "hi"
    assert settings.rag_target_ms == 200.0


def test_load_settings_reads_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "demo-token")
    settings = load_settings()
    assert settings.hf_token == "demo-token"