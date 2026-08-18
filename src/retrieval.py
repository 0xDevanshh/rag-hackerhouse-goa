"""
Retrieval module.

Implements a Retriever that embeds a query, searches a VectorStore for a
candidate pool, reranks candidates with metadata-driven boosts, and falls
back to a rewritten query once when confidence is low.
"""

import re

from pydantic import BaseModel

from src.chunking import Chunk
from src.vectorstore import VectorStore

RERANK_POOL_SIZE = 20
TOP_N = 5
IS_SELECTED_BOOST = 0.1
LANGUAGE_MATCH_BOOST = 0.05
LOW_CONFIDENCE_THRESHOLD = 0.3

# Unicode script ranges used for heuristic query-language detection.
_SCRIPT_LANGUAGE_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "bn": (0x0980, 0x09FF),  # Bengali
    "ta": (0x0B80, 0x0BFF),  # Tamil
}

# Minimal English stopword list for the "simple" query-rewrite fallback.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at", "to",
    "for", "and", "or", "what", "how", "why", "who", "when", "where", "does",
    "do", "did", "with", "this", "that", "it", "as", "be", "by",
}

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)


class RetrievalResult(BaseModel):
    """The outcome of a Retriever.retrieve() call."""

    chunks: list[Chunk]
    scores: list[float]
    low_confidence: bool


def _detect_language(text: str) -> str:
    """
    Heuristic script-based language detection: counts characters falling in
    known Unicode script ranges (Devanagari, Bengali, Tamil) and returns the
    dominant script's language code, defaulting to "en" when none match
    (e.g. Latin-script text).
    """
    counts = dict.fromkeys(_SCRIPT_LANGUAGE_RANGES, 0)
    for ch in text:
        code = ord(ch)
        for lang, (start, end) in _SCRIPT_LANGUAGE_RANGES.items():
            if start <= code <= end:
                counts[lang] += 1
                break

    dominant_lang, dominant_count = max(counts.items(), key=lambda item: item[1])
    return dominant_lang if dominant_count > 0 else "en"


def _rewrite_query(query: str) -> str:
    """
    Simple query-rewrite fallback: strips punctuation and drops a small set
    of common English stopwords. Used as a single retry when initial
    retrieval confidence is too low.
    """
    stripped = _PUNCTUATION_RE.sub(" ", query)
    tokens = [tok for tok in stripped.split() if tok.lower() not in _STOPWORDS]
    rewritten = " ".join(tokens).strip()
    return rewritten if rewritten else query


class Retriever:
    """
    Retrieves and reranks chunks for a query against a VectorStore.

    retrieve() embeds the query, searches for a rerank_pool_size candidate
    pool, reranks by cosine score plus metadata boosts (is_selected,
    language match), and returns the top_n results. If the best reranked
    score is still below low_confidence_threshold, it rewrites the query
    (stopword/punctuation stripping) and retries once before giving up.
    """

    def __init__(
        self,
        store: VectorStore,
        rerank_pool_size: int = RERANK_POOL_SIZE,
        top_n: int = TOP_N,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    ):
        """
        Args:
            store: an initialized VectorStore to search against.
            rerank_pool_size: number of candidates to fetch from the vector
                store before reranking.
            top_n: number of reranked results to return.
            low_confidence_threshold: reranked top score below which the
                query-rewrite fallback is triggered.
        """
        self.store = store
        self.rerank_pool_size = rerank_pool_size
        self.top_n = top_n
        self.low_confidence_threshold = low_confidence_threshold

    def _search_and_rerank(self, query: str) -> list[tuple[Chunk, float]]:
        query_language = _detect_language(query)
        query_embedding = self.store.embedder.encode([query])[0]
        candidates = self.store.search(query_embedding, top_k=self.rerank_pool_size)

        reranked = []
        for chunk, cosine_score in candidates:
            score = cosine_score
            if chunk.metadata.get("is_selected"):
                score += IS_SELECTED_BOOST
            if chunk.metadata.get("language") == query_language:
                score += LANGUAGE_MATCH_BOOST
            reranked.append((chunk, score))

        reranked.sort(key=lambda pair: pair[1], reverse=True)
        return reranked

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Retrieve the top_n chunks most relevant to query, after reranking.

        Args:
            query: the text query (e.g. transcribed from voice input).

        Returns:
            RetrievalResult: the retrieved chunks, their reranked scores, and
            a low_confidence flag (True if even the best result, after the
            query-rewrite retry, stayed below low_confidence_threshold).
        """
        reranked = self._search_and_rerank(query)
        top_score = reranked[0][1] if reranked else 0.0

        if top_score < self.low_confidence_threshold:
            rewritten_query = _rewrite_query(query)
            if rewritten_query != query:
                retry_reranked = self._search_and_rerank(rewritten_query)
                retry_top_score = retry_reranked[0][1] if retry_reranked else 0.0
                if retry_top_score > top_score:
                    reranked = retry_reranked
                    top_score = retry_top_score

        top_results = reranked[: self.top_n]
        return RetrievalResult(
            chunks=[chunk for chunk, _ in top_results],
            scores=[score for _, score in top_results],
            low_confidence=top_score < self.low_confidence_threshold,
        )
