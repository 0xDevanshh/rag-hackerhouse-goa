"""
Guardrails module.

Validates and filters both the incoming query and the generated answer:
InputGuardrail screens the raw query before retrieval, RelevanceGuardrail
refuses to generate when retrieval confidence is too low, and
GroundingGuardrail checks a generated answer's sentences are actually
supported by the retrieved context before it reaches the user.
"""

import re
import unicodedata
from collections import Counter

import numpy as np
from pydantic import BaseModel

from src.chunking import Chunk
from src.retrieval import LOW_CONFIDENCE_THRESHOLD, RetrievalResult
from src.text import split_sentences as _split_sentences
from src.vectorstore import Embedder

REFUSAL_RESPONSE = "I don't have enough information in the dataset to answer that."
MAX_QUERY_CHARS = 256
MAX_QUERY_TOKENS = 64
_QUERY_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_query_input(text: str) -> str:
    """Normalize and bound user text before any model or retrieval work."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    tokens = _QUERY_TOKEN_RE.findall(normalized)
    if len(tokens) > MAX_QUERY_TOKENS:
        normalized = " ".join(tokens[:MAX_QUERY_TOKENS])
    return normalized[:MAX_QUERY_CHARS].rstrip()


def _is_repetitive_query(text: str) -> bool:
    tokens = _QUERY_TOKEN_RE.findall(text)
    if len(tokens) < 24:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    repeated_terms = Counter(tokens).most_common(3)
    return unique_ratio < 0.28 or (repeated_terms and repeated_terms[0][1] / len(tokens) > 0.35)


class GuardResult(BaseModel):
    """The outcome of a guardrail check."""

    allowed: bool
    reason: str
    response_override: str | None = None


def _max_similarities(sentence_vectors: np.ndarray, chunk_vectors: np.ndarray) -> np.ndarray:
    """
    For each sentence vector, its greatest cosine similarity to any chunk
    vector — as one matrix product instead of a Python loop over every
    (sentence, chunk) pair, which is what this replaced.

    Both operand sets are row-normalized first, so the product is cosine
    rather than raw dot; that keeps the result identical to a pairwise cosine
    even if a caller hands in unnormalized vectors.
    """
    if sentence_vectors.size == 0 or chunk_vectors.size == 0:
        return np.zeros(len(sentence_vectors), dtype=np.float32)

    def unit(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)

    return (unit(sentence_vectors) @ unit(chunk_vectors).T).max(axis=1)


class InputGuardrail:
    """
    Screens an incoming (transcribed) user query before it reaches
    retrieval: rejects empty input, likely gibberish, and basic
    unsafe-content patterns.
    """

    MIN_LENGTH = 2
    MIN_ALPHA_RATIO = 0.3

    UNSAFE_PATTERNS = [
        re.compile(r"\bhow (?:do i|to) (?:make|build|synthesize)\b.*\b(?:bomb|explosive|nerve agent)\b", re.IGNORECASE),
        re.compile(r"\bkill (?:myself|yourself|someone)\b", re.IGNORECASE),
        re.compile(r"\b(?:child|minor)s?\b.{0,20}\b(?:sexual|explicit)\b", re.IGNORECASE),
    ]

    def check(self, query: str) -> GuardResult:
        """
        Validate a raw query before retrieval.

        Args:
            query: the text query to validate.

        Returns:
            GuardResult: allowed=True with reason "ok" if the query passes
            all checks; otherwise allowed=False with a reason code and a
            canned response_override.
        """
        stripped = normalize_query_input(query)

        if not stripped or len(stripped) < self.MIN_LENGTH:
            return GuardResult(allowed=False, reason="empty_query", response_override=REFUSAL_RESPONSE)

        if _is_repetitive_query(stripped):
            return GuardResult(allowed=False, reason="repetitive_query", response_override=REFUSAL_RESPONSE)

        for pattern in self.UNSAFE_PATTERNS:
            if pattern.search(stripped):
                return GuardResult(allowed=False, reason="unsafe_content", response_override=REFUSAL_RESPONSE)

        if self._is_gibberish(stripped):
            return GuardResult(allowed=False, reason="gibberish_query", response_override=REFUSAL_RESPONSE)

        return GuardResult(allowed=True, reason="ok")

    def _is_gibberish(self, text: str) -> bool:
        non_space = [ch for ch in text if not ch.isspace()]
        if not non_space:
            return True
        alpha_ratio = sum(1 for ch in non_space if ch.isalpha()) / len(non_space)
        return alpha_ratio < self.MIN_ALPHA_RATIO


class RelevanceGuardrail:
    """
    Refuses to generate an answer when retrieval confidence is too low:
    either RetrievalResult.low_confidence is set, or the top result's score
    falls below relevance_threshold.
    """

    def __init__(self, relevance_threshold: float = LOW_CONFIDENCE_THRESHOLD):
        """
        Args:
            relevance_threshold: minimum top retrieval score required to
                proceed to generation.
        """
        self.relevance_threshold = relevance_threshold

    def check(self, retrieval_result: RetrievalResult) -> GuardResult:
        """
        Decide whether retrieval was strong enough to attempt generation.

        Args:
            retrieval_result: the RetrievalResult from Retriever.retrieve().

        Returns:
            GuardResult: allowed=True with reason "ok" if retrieval is
            strong enough; otherwise allowed=False with a canned
            response_override instead of calling generation.
        """
        if retrieval_result.low_confidence:
            return GuardResult(allowed=False, reason="low_confidence_retrieval", response_override=REFUSAL_RESPONSE)

        top_score = retrieval_result.scores[0] if retrieval_result.scores else 0.0
        if top_score < self.relevance_threshold:
            return GuardResult(allowed=False, reason="top_score_below_threshold", response_override=REFUSAL_RESPONSE)

        return GuardResult(allowed=True, reason="ok")


class GroundingGuardrail:
    """
    Checks that a generated answer is actually supported by the retrieved
    context: each answer sentence is embedded and compared (max cosine
    similarity) against the retrieved chunks' embeddings. If more than
    max_unsupported_ratio of sentences fall below grounding_threshold, the
    whole answer is discarded in favor of the refusal response.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        grounding_threshold: float = 0.5,
        max_unsupported_ratio: float = 0.3,
    ):
        """
        Args:
            embedder: Embedder used to encode answer sentences and chunk
                texts. Defaults to a new Embedder() (real model) if omitted.
            grounding_threshold: minimum max-cosine-similarity to a
                retrieved chunk for a sentence to count as supported.
            max_unsupported_ratio: fraction of unsupported sentences above
                which the whole answer is discarded.
        """
        self.embedder = embedder or Embedder()
        self.grounding_threshold = grounding_threshold
        self.max_unsupported_ratio = max_unsupported_ratio

    def _chunk_vectors(
        self,
        retrieved_chunks: list[Chunk],
        chunk_embeddings: np.ndarray | None,
        timing: dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Vectors to ground against: the caller's precomputed ones if supplied,
        otherwise encoded from the chunk texts.

        Callers should supply them. Every retrieved chunk was already embedded
        at index-build time, so re-encoding its text here is pure duplicated
        work — measured at 17-23ms per request for a top-5 of
        MSMARCO-length passages, all of it invisible in the old latency
        breakdown because it happened inside an untraced guardrail stage.
        VectorStore.embeddings_for() reads the same vectors back out of the
        FAISS index for the cost of a memory copy.
        """
        if chunk_embeddings is not None and len(chunk_embeddings):
            return np.asarray(chunk_embeddings, dtype=np.float32)
        return self.embedder.encode([chunk.text for chunk in retrieved_chunks], timing=timing)

    def check(
        self,
        answer_text: str,
        retrieved_chunks: list[Chunk],
        chunk_embeddings: np.ndarray | None = None,
        timing: dict[str, float] | None = None,
    ) -> GuardResult:
        """
        Validate that answer_text is grounded in retrieved_chunks.

        Args:
            answer_text: the generated answer to check.
            retrieved_chunks: the chunks the answer was generated from.
            chunk_embeddings: precomputed embeddings for retrieved_chunks,
                row-aligned with it — e.g. from
                VectorStore.embeddings_for(retrieval_result.positions). Skips
                re-encoding the chunk texts; see _chunk_vectors.
            timing: optional sink for embedding sub-timings, forwarded to
                Embedder.encode.

        Returns:
            GuardResult: allowed=True with reason "ok" if the answer is
            sufficiently grounded; otherwise allowed=False with the
            unsupported-sentence ratio in reason and a canned
            response_override.
        """
        sentences = _split_sentences(answer_text)
        if not sentences:
            return GuardResult(allowed=False, reason="empty_answer", response_override=REFUSAL_RESPONSE)
        if not retrieved_chunks:
            return GuardResult(allowed=False, reason="no_context_to_ground_against", response_override=REFUSAL_RESPONSE)

        # The local extractive provider returns a literal prefix of the top
        # passage followed by its citation. Exact evidence needs no second
        # embedding pass; preserving this check keeps the fast path grounded.
        cited_evidence = re.sub(r"\s*\[passage_id:\s*[^\]]+\]", "", answer_text).strip()
        if re.search(r"\[passage_id:\s*[^\]]+\]", answer_text) and cited_evidence:
            evidence_sentences = _split_sentences(cited_evidence)
            if evidence_sentences and all(
                any(sentence.strip() in chunk.text for chunk in retrieved_chunks)
                for sentence in evidence_sentences
            ):
                return GuardResult(allowed=True, reason="exact_retrieved_evidence")

        chunk_vectors = self._chunk_vectors(retrieved_chunks, chunk_embeddings, timing)
        sentence_embeddings = self.embedder.encode(sentences, timing=timing)

        similarities = _max_similarities(sentence_embeddings, chunk_vectors)
        unsupported_count = int((similarities < self.grounding_threshold).sum())

        unsupported_ratio = unsupported_count / len(sentences)
        if unsupported_ratio > self.max_unsupported_ratio:
            return GuardResult(
                allowed=False,
                reason=f"ungrounded_answer:{unsupported_ratio:.2f}",
                response_override=REFUSAL_RESPONSE,
            )

        return GuardResult(allowed=True, reason="ok")

    def is_sentence_grounded(
        self,
        sentence: str,
        retrieved_chunks: list[Chunk],
        chunk_embeddings: np.ndarray | None = None,
        timing: dict[str, float] | None = None,
    ) -> bool:
        """
        Check a single sentence against retrieved_chunks using the same
        max-cosine-similarity test as check(), without the whole-answer
        unsupported-ratio aggregation. Used for incremental (streaming)
        grounding checks, where each sentence is validated as soon as it's
        generated instead of waiting for the complete answer.

        Args:
            sentence: a single generated sentence.
            retrieved_chunks: the chunks the answer was generated from.
            chunk_embeddings: precomputed embeddings for retrieved_chunks. On
                the streaming path this matters more than on check(): without
                it, the chunk texts are re-encoded once per generated
                sentence, so a five-sentence answer paid that 17-23ms five
                times over.
            timing: optional sink for embedding sub-timings.

        Returns:
            bool: True if the sentence's max similarity to any retrieved
            chunk meets grounding_threshold.
        """
        if not sentence.strip() or not retrieved_chunks:
            return False
        chunk_vectors = self._chunk_vectors(retrieved_chunks, chunk_embeddings, timing)
        sentence_embedding = self.embedder.encode([sentence], timing=timing)
        return bool(_max_similarities(sentence_embedding, chunk_vectors)[0] >= self.grounding_threshold)
