"""
Pipeline orchestration module.

Wires together STT, retrieval, guardrails, and generation into a single
end-to-end voice-RAG flow, and exposes it via a FastAPI app for serving.
"""

from typing import Any


class VoiceRAGPipeline:
    """
    End-to-end orchestrator: audio/text query -> transcription (if needed) ->
    guardrail checks -> retrieval -> generation -> guardrail checks -> answer.
    """

    def __init__(self):
        """
        Initialize pipeline components (config, vector store, etc.).
        """
        raise NotImplementedError

    def run(self, audio_path: str | None = None, query_text: str | None = None) -> dict[str, Any]:
        """
        Run the full pipeline for a single request, from either raw audio or
        pre-transcribed text input.

        Args:
            audio_path: path to an audio file to transcribe, if input is voice.
            query_text: text query, if input is already text (bypasses STT).

        Returns:
            dict: result payload including the answer, retrieved sources, and
            any guardrail metadata.
        """
        raise NotImplementedError


def create_app():
    """
    Build and return the FastAPI application exposing the pipeline over HTTP.

    Returns:
        FastAPI: the configured application instance.
    """
    raise NotImplementedError
