"""
Guardrails module.

Validates and filters both the incoming query and the generated answer:
input sanitization/relevance checks, retrieved-context sufficiency checks,
and output safety/grounding checks before an answer is returned to the user.
"""

from typing import Any


def check_input(query: str) -> tuple[bool, str | None]:
    """
    Validate an incoming (transcribed) user query before retrieval.

    Args:
        query: the text query to validate.

    Returns:
        tuple[bool, str | None]: (is_valid, rejection_reason). rejection_reason
        is None when is_valid is True.
    """
    raise NotImplementedError


def check_context_sufficiency(query: str, retrieved_chunks: list[dict[str, Any]]) -> bool:
    """
    Determine whether the retrieved chunks provide enough grounding to
    answer the query, to avoid hallucinated answers on empty/weak context.

    Args:
        query: the text query.
        retrieved_chunks: chunks returned by the retrieval step.

    Returns:
        bool: True if the context is sufficient to attempt generation.
    """
    raise NotImplementedError


def check_output(answer: str, retrieved_chunks: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """
    Validate a generated answer before it is returned to the user, e.g.
    checking for groundedness in the retrieved context and disallowed content.

    Args:
        answer: the generated answer text.
        retrieved_chunks: chunks the answer was generated from.

    Returns:
        tuple[bool, str | None]: (is_valid, rejection_reason). rejection_reason
        is None when is_valid is True.
    """
    raise NotImplementedError
