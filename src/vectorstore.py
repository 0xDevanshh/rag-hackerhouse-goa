"""
Vector store module.

Embeds text chunks with a sentence-transformers model and indexes them in a
FAISS index for fast approximate nearest-neighbor similarity search.
"""

import pickle
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk

DEFAULT_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    """
    Wraps a sentence-transformers embedding model for batch encoding into
    L2-normalized vectors. Defaults to "paraphrase-multilingual-MiniLM-L12-v2"
    for multilingual (e.g. Hindi/Tamil/Bengali) support.

    The underlying SentenceTransformer model is cached as a singleton per
    model name at the class level, so it is loaded once per process no
    matter how many Embedder instances are constructed.
    """

    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)
        self._model = self._model_cache[model_name]

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Batch-encode texts into L2-normalized embedding vectors.

        Args:
            texts: the texts to embed.

        Returns:
            np.ndarray: array of shape (len(texts), embedding_dim), where
            each row is L2-normalized.
        """
        return self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


class VectorStore:
    """
    Cosine-similarity vector store: embeds Chunk objects with an Embedder and
    indexes them in a faiss.IndexFlatIP. Since Embedder outputs are
    L2-normalized, inner product over those vectors is equivalent to cosine
    similarity.
    """

    def __init__(self, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME):
        """
        Args:
            embedding_model_name: name/path of the sentence-transformers model
                the internal Embedder uses to embed chunks and queries.
        """
        self.embedder = Embedder(embedding_model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """
        Embed the given chunks and build the FAISS index from scratch.

        Args:
            chunks: chunk records to index.
        """
        embeddings = np.asarray(self.embedder.encode([chunk.text for chunk in chunks]), dtype=np.float32)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = list(chunks)

    @staticmethod
    def _matches_filter(chunk: Chunk, metadata_filter: dict[str, Any]) -> bool:
        return all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Search for the top_k chunks most similar to query_embedding.

        When metadata_filter is given (e.g. {"language": "hi"}), it's
        applied as an exact-match filter on each candidate's chunk.metadata,
        post-search: top_k*4 candidates are over-fetched from FAISS first,
        then filtered, then truncated back down to top_k. Without a filter,
        exactly top_k candidates are fetched.

        Args:
            query_embedding: a single embedding vector, shape (dim,) or (1, dim).
            top_k: number of results to return.
            metadata_filter: optional exact-match filters on chunk metadata.

        Returns:
            list[tuple[Chunk, float]]: (chunk, similarity score) pairs, ordered
            by descending similarity.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        fetch_k = min(top_k * 4 if metadata_filter else top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            if metadata_filter and not self._matches_filter(chunk, metadata_filter):
                continue
            results.append((chunk, float(score)))

        return results[:top_k]

    def save(self, index_path: str) -> None:
        """
        Persist the FAISS index to "{index_path}.faiss" and the chunk
        metadata to a pickled sidecar at "{index_path}.meta.pkl".

        Args:
            index_path: filesystem path/prefix to save the files under.
        """
        if self.index is None:
            raise ValueError("cannot save an empty VectorStore; call build() first")
        faiss.write_index(self.index, f"{index_path}.faiss")
        with open(f"{index_path}.meta.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, index_path: str) -> None:
        """
        Load a previously persisted FAISS index and its chunk metadata sidecar.

        Args:
            index_path: filesystem path/prefix the files were saved under
                (the same prefix passed to save()).
        """
        self.index = faiss.read_index(f"{index_path}.faiss")
        with open(f"{index_path}.meta.pkl", "rb") as f:
            self.chunks = pickle.load(f)
