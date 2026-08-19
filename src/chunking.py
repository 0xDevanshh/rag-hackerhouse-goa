"""
Document chunking module.

Defines a common Chunker interface and four concrete chunking strategies —
FixedSizeChunker, SentenceSemanticChunker, MetadataAwareChunker, and
RecursiveChunker — plus a ChunkerRegistry for building any strategy by name
from a config dict.
"""

import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.text import split_sentences as _split_sentences


class Chunk(BaseModel):
    """A single chunk of text produced by a Chunker, with provenance metadata."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    strategy_name: str


class Chunker(ABC):
    """Common interface all chunking strategies implement."""

    @abstractmethod
    def chunk(self, doc: dict[str, Any]) -> list[Chunk]:
        """
        Split a single document into chunks.

        Args:
            doc: a raw document record. Shape depends on the strategy: generic
                strategies expect {"id", "text", ...}; MetadataAwareChunker
                expects the MSMARCO-XI query/passages shape (see its docstring).

        Returns:
            list[Chunk]: the chunks produced from this document.
        """
        raise NotImplementedError


class FixedSizeChunker(Chunker):
    """Sliding window over characters: fixed-size chunks with a fixed overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: dict[str, Any]) -> list[Chunk]:
        text = doc.get("text", "")
        doc_id = doc.get("id")
        step = self.chunk_size - self.overlap

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            piece = text[start:end]
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        metadata={"doc_id": doc_id, "start": start, "end": min(end, len(text))},
                        strategy_name="fixed_size",
                    )
                )
            if end >= len(text):
                break
            start += step
        return chunks


class SentenceSemanticChunker(Chunker):
    """
    Splits text into sentences, embeds each with sentence-transformers, and
    greedily merges adjacent sentences into a chunk as long as the next
    sentence's embedding stays similar enough to the running chunk's
    centroid embedding; a similarity drop below similarity_threshold starts
    a new chunk.
    """

    def __init__(self, similarity_threshold: float = 0.75, model_name: str = "all-MiniLM-L6-v2"):
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _make_chunk(self, sentences: list[str], doc_id: Any, chunk_index: int) -> Chunk:
        return Chunk(
            text=" ".join(sentences),
            metadata={"doc_id": doc_id, "chunk_index": chunk_index, "num_sentences": len(sentences)},
            strategy_name="sentence_semantic",
        )

    def chunk(self, doc: dict[str, Any]) -> list[Chunk]:
        text = doc.get("text", "")
        doc_id = doc.get("id")
        sentences = _split_sentences(text)
        if not sentences:
            return []

        embeddings = self.model.encode(sentences)
        chunks = []
        current_sentences = [sentences[0]]
        current_embeddings = [embeddings[0]]

        for sentence, embedding in zip(sentences[1:], embeddings[1:]):
            centroid = np.mean(current_embeddings, axis=0)
            similarity = self._cosine_similarity(centroid, embedding)
            if similarity >= self.similarity_threshold:
                current_sentences.append(sentence)
                current_embeddings.append(embedding)
            else:
                chunks.append(self._make_chunk(current_sentences, doc_id, len(chunks)))
                current_sentences = [sentence]
                current_embeddings = [embedding]

        chunks.append(self._make_chunk(current_sentences, doc_id, len(chunks)))
        return chunks


class MetadataAwareChunker(Chunker):
    """
    Treats each MSMARCO-XI passage as its own chunk (no further splitting),
    tagged with query_id, passage_id, language, is_selected, and query_text
    so retrieval can later boost/filter by that metadata.

    Expects `doc` shaped like a mapped QueryDoc (see src/data_loader.py):
    {"query_id", "query_text", "language", "passages": [{"text", "is_selected", ...}, ...]}.
    Falls back to treating the whole doc as a single passage when it has no
    "passages" field, so generic {"id", "title", "text"} documents (e.g.
    data/sample_corpus.json) can still be run through this strategy.
    """

    def chunk(self, doc: dict[str, Any]) -> list[Chunk]:
        query_id = doc.get("query_id", doc.get("id"))
        query_text = doc.get("query_text", doc.get("title", ""))
        language = doc.get("language", "unknown")
        passages = doc.get("passages")
        if passages is None:
            passages = [{"text": doc.get("text", ""), "is_selected": True}]

        chunks = []
        for passage_id, passage in enumerate(passages):
            chunks.append(
                Chunk(
                    text=passage.get("text", ""),
                    metadata={
                        "query_id": query_id,
                        "passage_id": passage_id,
                        "language": language,
                        "is_selected": bool(passage.get("is_selected", False)),
                        "query_text": query_text,
                    },
                    strategy_name="metadata_aware",
                )
            )
        return chunks


class RecursiveChunker(Chunker):
    """
    Splits by paragraph first. Any paragraph longer than max_chunk_size is
    recursively split into sentences; the resulting pieces (whole paragraphs
    and/or sentences) are then greedily packed back together, largest first,
    into chunks that stay within max_chunk_size.
    """

    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size

    def _pack_pieces(self, pieces: list[str]) -> list[str]:
        merged = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if len(candidate) <= self.max_chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = piece
        if current:
            merged.append(current)
        return merged

    def chunk(self, doc: dict[str, Any]) -> list[Chunk]:
        text = doc.get("text", "")
        doc_id = doc.get("id")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        pieces = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.max_chunk_size:
                pieces.append(paragraph)
            else:
                pieces.extend(_split_sentences(paragraph))

        merged = self._pack_pieces(pieces)
        return [
            Chunk(
                text=piece,
                metadata={"doc_id": doc_id, "chunk_index": i},
                strategy_name="recursive",
            )
            for i, piece in enumerate(merged)
        ]


class ChunkerRegistry:
    """Builds a Chunker instance by strategy name, e.g. from a config dict."""

    _strategies: dict[str, type[Chunker]] = {
        "fixed_size": FixedSizeChunker,
        "sentence_semantic": SentenceSemanticChunker,
        "metadata_aware": MetadataAwareChunker,
        "recursive": RecursiveChunker,
    }

    @classmethod
    def available_strategies(cls) -> list[str]:
        """List the registered strategy names."""
        return sorted(cls._strategies)

    @classmethod
    def build(cls, name: str, **kwargs: Any) -> Chunker:
        """
        Instantiate a Chunker by strategy name.

        Args:
            name: registered strategy name (see available_strategies()).
            **kwargs: constructor arguments forwarded to the strategy class.

        Returns:
            Chunker: the constructed chunker instance.
        """
        try:
            strategy_cls = cls._strategies[name]
        except KeyError:
            raise ValueError(
                f"Unknown chunking strategy: {name!r}. Available: {cls.available_strategies()}"
            ) from None
        return strategy_cls(**kwargs)

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> Chunker:
        """
        Instantiate a Chunker from a config dict of the form
        {"strategy": "<name>", ...strategy-specific kwargs}.

        Args:
            config: config dict; the "strategy" key selects the class, all
                other keys are forwarded as constructor kwargs.

        Returns:
            Chunker: the constructed chunker instance.
        """
        config = dict(config)
        name = config.pop("strategy")
        return cls.build(name, **config)
