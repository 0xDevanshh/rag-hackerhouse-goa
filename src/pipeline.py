"""
Pipeline orchestration module.

Thin public facade over PipelineHarness and the FastAPI app in src/api.py.
The real orchestration logic lives in src/harness.py (PipelineHarness) and
the HTTP layer in src/api.py; this module re-exports them so callers have a
stable import path.
"""

from typing import Any

from src.harness import PipelineHarness, PipelineResult  # noqa: F401


class VoiceRAGPipeline:
    """
    End-to-end orchestrator: audio/text query -> transcription (if needed) ->
    guardrail checks -> retrieval -> generation -> guardrail checks -> answer.

    This is a thin wrapper over PipelineHarness. Prefer using PipelineHarness
    directly for programmatic access, or the FastAPI app (src/api.py) for HTTP
    access.
    """

    def __init__(self):
        """
        Initialize pipeline components from environment variables.

        Reads SARVAM_API_KEY, GROQ_API_KEY / ANTHROPIC_API_KEY / LLM_PROVIDER,
        EMBEDDING_MODEL, CORPUS_LANGUAGE, CORPUS_SPLIT, CORPUS_LIMIT from the
        environment. Uses src/api.py's get_harness() and _load_default_chunks()
        to build the same harness the HTTP API serves.
        """
        from src.api import _load_default_chunks, get_harness
        from src.vectorstore import VectorStore

        store = VectorStore()
        chunks = _load_default_chunks()
        self._harness = PipelineHarness(store=store, chunks=chunks)
        self._harness.build_index()

    def run(self, audio_path: str | None = None, query_text: str | None = None) -> dict[str, Any]:
        """
        Run the full pipeline for a single request.

        Args:
            audio_path: path to an audio file to transcribe, if input is voice.
            query_text: text query, if input is already text (bypasses STT).

        Returns:
            dict: PipelineResult serialized to a dict, including answer,
            sources, trace, guard_flags, degraded, and errors.
        """
        import asyncio

        if audio_path is None and query_text is None:
            raise ValueError("Provide either audio_path or query_text")
        if audio_path is not None and query_text is not None:
            raise ValueError("Provide only one of audio_path or query_text")

        input_value: str | bytes = audio_path if audio_path is not None else query_text
        result: PipelineResult = asyncio.run(self._harness.run(input_value))
        return result.model_dump()


def create_app():
    """
    Build and return the FastAPI application exposing the pipeline over HTTP.

    Returns:
        FastAPI: the configured application instance from src/api.py.
    """
    from src.api import app

    return app
