"""
Vector store module.

Embeds text chunks with a sentence-transformers model and indexes them in a
FAISS index for fast approximate nearest-neighbor similarity search.
"""

from typing import Any


class VectorStore:
    """
    Wraps a sentence-transformers embedding model and a FAISS index, and
    manages the mapping between vector index positions and chunk metadata.
    """

    def __init__(self, embedding_model_name: str):
        """
        Args:
            embedding_model_name: name/path of the sentence-transformers model to use.
        """
        raise NotImplementedError

    def build_index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Embed a list of chunk records and build the FAISS index from scratch.

        Args:
            chunks: chunk records as produced by chunking.chunk_corpus.
        """
        raise NotImplementedError

    def add(self, chunks: list[dict[str, Any]]) -> None:
        """
        Embed and add new chunk records to an existing index.

        Args:
            chunks: chunk records to add.
        """
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Embed a query and return the top_k most similar chunks.

        Args:
            query: the text query to search for.
            top_k: number of results to return.

        Returns:
            list[dict]: matching chunk records with similarity scores.
        """
        raise NotImplementedError

    def save(self, index_path: str) -> None:
        """
        Persist the FAISS index and associated chunk metadata to disk.

        Args:
            index_path: filesystem path/prefix to save the index files under.
        """
        raise NotImplementedError

    def load(self, index_path: str) -> None:
        """
        Load a previously persisted FAISS index and chunk metadata from disk.

        Args:
            index_path: filesystem path/prefix the index files were saved under.
        """
        raise NotImplementedError
