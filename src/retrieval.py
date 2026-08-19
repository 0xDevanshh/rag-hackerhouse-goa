"""
Retrieval module.

Implements a Retriever that embeds a query, searches a VectorStore for a
candidate pool, reranks candidates with metadata-driven boosts, and falls
back to a rewritten query once when confidence is low.
"""

import re
import time

from pydantic import BaseModel

from src.chunking import Chunk
from src.vectorstore import VectorStore

# 10 candidates is plenty for a fusion rerank that only adds small metadata
# boosts (+0.1/+0.05) on top of cosine score — a candidate ranked much below
# 10th place on raw similarity is vanishingly unlikely to be promoted into
# the top_n by those boosts alone. Lower this pool size = less rerank work
# and a smaller FAISS fetch, at negligible recall risk for this scoring model.
RERANK_POOL_SIZE = 10
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
    # FAISS index positions of `chunks`, parallel to it. Carried so downstream
    # consumers (GroundingGuardrail) can read each chunk's already-computed
    # embedding straight out of the index instead of re-encoding its text.
    # Empty when retrieval ran against a store that couldn't supply positions.
    positions: list[int] = []


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
        is_selected_boost: float = IS_SELECTED_BOOST,
        language_match_boost: float = LANGUAGE_MATCH_BOOST,
    ):
        """
        Args:
            store: an initialized VectorStore to search against.
            rerank_pool_size: number of candidates to fetch from the vector
                store before reranking.
            top_n: number of reranked results to return.
            low_confidence_threshold: reranked top score below which the
                query-rewrite fallback is triggered.
            is_selected_boost: score added to a chunk tagged
                is_selected. MUST be set to 0 when scoring retrieval against
                MSMARCO-XI's is_selected labels: that flag *is* the relevance
                label, so boosting by it leaks ground truth into the ranking
                and inflates every retrieval metric. See
                benchmarks/run_eval.py.
            language_match_boost: score added to a chunk whose language
                matches the query's detected script.
        """
        self.store = store
        self.rerank_pool_size = rerank_pool_size
        self.top_n = top_n
        self.low_confidence_threshold = low_confidence_threshold
        self.is_selected_boost = is_selected_boost
        self.language_match_boost = language_match_boost

    def _search_and_rerank(
        self, query: str, timing: dict[str, float] | None = None
    ) -> list[tuple[Chunk, float, int]]:
        query_language = _detect_language(query)

        # Embedder.encode writes embedding_cache_ms / embedding_compute_ms
        # into `timing` itself, so a cache hit and a real forward pass are
        # distinguishable in the trace rather than averaged into one number.
        query_embedding = self.store.embedder.encode([query], timing=timing)[0]

        t1 = time.perf_counter()
        candidates = self.store.search(query_embedding, top_k=self.rerank_pool_size)
        t2 = time.perf_counter()

        reranked = []
        for chunk, cosine_score, position in candidates:
            score = cosine_score
            if chunk.metadata.get("is_selected"):
                score += self.is_selected_boost
            if chunk.metadata.get("language") == query_language:
                score += self.language_match_boost
            reranked.append((chunk, score, position))
        reranked.sort(key=lambda triple: triple[1], reverse=True)
        t3 = time.perf_counter()

        if timing is not None:
            timing["vector_search_ms"] = timing.get("vector_search_ms", 0.0) + (t2 - t1) * 1000
            timing["reranking_ms"] = timing.get("reranking_ms", 0.0) + (t3 - t2) * 1000
            timing["search_passes"] = timing.get("search_passes", 0) + 1

        return reranked

    @staticmethod
    def _dedupe(reranked: list[tuple[Chunk, float, int]]) -> list[tuple[Chunk, float, int]]:
        """Drop exact-duplicate chunk texts, keeping the first (highest-scored) occurrence."""
        seen: set[str] = set()
        deduped = []
        for chunk, score, position in reranked:
            if chunk.text in seen:
                continue
            seen.add(chunk.text)
            deduped.append((chunk, score, position))
        return deduped

    def retrieve(self, query: str, timing: dict[str, float] | None = None) -> RetrievalResult:
        """
        Retrieve the top_n chunks most relevant to query, after reranking.

        Args:
            query: the text query (e.g. transcribed from voice input).
            timing: optional dict to accumulate embedding_cache_ms /
                embedding_compute_ms / vector_search_ms / reranking_ms and a
                search_passes count into (for latency instrumentation). Unused
                by default — pass a dict to collect these sub-timings.

        Returns:
            RetrievalResult: the retrieved chunks, their reranked scores, their
            FAISS index positions, and a low_confidence flag (True if even the
            best result, after the query-rewrite retry, stayed below
            low_confidence_threshold).
        """
        reranked = self._search_and_rerank(query, timing)
        top_score = reranked[0][1] if reranked else 0.0

        if top_score < self.low_confidence_threshold:
            rewritten_query = _rewrite_query(query)
            if rewritten_query != query:
                # A second pass costs a second query embedding. That was
                # expensive when a single encode could spike past 300ms; with
                # the embedder pinned to CPU and warmed it costs ~5ms, so the
                # retry is kept — it buys real recall on low-confidence
                # queries, and search_passes in the trace makes its cost
                # visible rather than hidden inside one embedding_ms total.
                retry_reranked = self._search_and_rerank(rewritten_query, timing)
                retry_top_score = retry_reranked[0][1] if retry_reranked else 0.0
                if retry_top_score > top_score:
                    reranked = retry_reranked
                    top_score = retry_top_score

        top_results = self._dedupe(reranked)[: self.top_n]
        return RetrievalResult(
            chunks=[chunk for chunk, _, _ in top_results],
            scores=[score for _, score, _ in top_results],
            positions=[position for _, _, position in top_results],
            low_confidence=top_score < self.low_confidence_threshold,
        )
