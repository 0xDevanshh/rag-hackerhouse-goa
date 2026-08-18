"""
Generation module.

Calls the Anthropic API to generate a grounded answer from a user query and
its retrieved context chunks, using tenacity-backed retries for resilience.
"""

from typing import Any


def build_prompt(query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Construct the prompt (system + context + query) sent to the generation model.

    Args:
        query: the text query to answer.
        retrieved_chunks: chunk records to include as grounding context.

    Returns:
        str: the fully assembled prompt.
    """
    raise NotImplementedError


def generate_answer(query: str, retrieved_chunks: list[dict[str, Any]], model: str) -> str:
    """
    Generate an answer to the query grounded in the retrieved chunks, via the
    Anthropic API. Retries on transient failures.

    Args:
        query: the text query to answer.
        retrieved_chunks: chunk records to include as grounding context.
        model: the Anthropic model name to use for generation.

    Returns:
        str: the generated answer text.
    """
    raise NotImplementedError
