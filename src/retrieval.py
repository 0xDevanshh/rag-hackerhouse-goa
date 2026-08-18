"""
Retrieval module.

Given a user query, retrieves and ranks the most relevant chunks from the
vector store, optionally applying re-ranking or filtering logic on top of
raw similarity search.
"""

from typing import Any

from src.vectorstore import VectorStore


def retrieve(query: str, store: VectorStore, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve the top_k most relevant chunks for a query from the vector store.

    Args:
        query: the text query (e.g. transcribed from voice input).
        store: an initialized VectorStore instance to search against.
        top_k: number of chunks to retrieve.

    Returns:
        list[dict]: retrieved chunk records with scores, ordered by relevance.
    """
    raise NotImplementedError


def rerank(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Re-rank a list of candidate chunks against the query using a secondary
    scoring signal (e.g. cross-encoder or heuristic).

    Args:
        query: the text query.
        candidates: chunk records to re-rank.

    Returns:
        list[dict]: re-ranked chunk records, most relevant first.
    """
    raise NotImplementedError
