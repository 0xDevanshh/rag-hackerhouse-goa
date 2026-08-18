"""
Central configuration for the voice-rag pipeline.

Loads settings (API keys, model names, chunking parameters, index paths, etc.)
from environment variables / a .env file, and exposes them as a single typed
settings object that other modules import from.
"""

from pydantic import BaseModel


class Settings(BaseModel):
    """
    Typed application settings, populated from environment variables.

    Expected fields (to be implemented):
    - sarvam_api_key: API key for the Sarvam STT service.
    - anthropic_api_key: API key for the Anthropic generation service.
    - embedding_model_name: name/path of the sentence-transformers model used for embeddings.
    - generation_model_name: name of the Anthropic model used for answer generation.
    - chunk_size / chunk_overlap: parameters controlling document chunking.
    - vector_index_path: filesystem path where the FAISS index is persisted.
    - top_k: default number of chunks to retrieve per query.
    """

    pass


def load_settings() -> Settings:
    """
    Load environment variables (via python-dotenv) and construct a Settings instance.

    Returns:
        Settings: the populated application configuration.
    """
    raise NotImplementedError
