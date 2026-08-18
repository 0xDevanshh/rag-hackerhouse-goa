"""
Document chunking module.

Splits raw source documents (e.g. from data/sample_corpus.json) into smaller,
overlapping text chunks suitable for embedding and retrieval.
"""

from typing import Any


def load_corpus(corpus_path: str) -> list[dict[str, Any]]:
    """
    Load raw documents from a corpus JSON file.

    Args:
        corpus_path: filesystem path to the corpus JSON file.

    Returns:
        list[dict]: raw document records (e.g. {"id", "title", "text", ...}).
    """
    raise NotImplementedError


def chunk_document(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Split a single document's text into overlapping chunks.

    Args:
        text: the full document text.
        chunk_size: target number of characters/tokens per chunk.
        chunk_overlap: number of characters/tokens shared between consecutive chunks.

    Returns:
        list[str]: ordered text chunks.
    """
    raise NotImplementedError


def chunk_corpus(
    documents: list[dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 50
) -> list[dict[str, Any]]:
    """
    Chunk every document in a corpus and attach chunk-level metadata.

    Args:
        documents: raw document records as returned by load_corpus.
        chunk_size: target number of characters/tokens per chunk.
        chunk_overlap: number of characters/tokens shared between consecutive chunks.

    Returns:
        list[dict]: chunk records (e.g. {"doc_id", "chunk_id", "text", "metadata", ...}).
    """
    raise NotImplementedError
