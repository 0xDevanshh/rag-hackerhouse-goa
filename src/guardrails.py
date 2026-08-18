"""
Guardrails module.

Validates and filters both the incoming query and the generated answer:
InputGuardrail screens the raw query before retrieval, RelevanceGuardrail
refuses to generate when retrieval confidence is too low, and
GroundingGuardrail checks a generated answer's sentences are actually
supported by the retrieved context before it reaches the user.
"""

import re

import numpy as np
from pydantic import BaseModel

from src.chunking import Chunk
from src.retrieval import LOW_CONFIDENCE_THRESHOLD, RetrievalResult
from src.vectorstore import Embedder

REFUSAL_RESPONSE = "I don't have enough information in the dataset to answer that."

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class GuardResult(BaseModel):
    """The outcome of a guardrail check."""

    allowed: bool
    reason: str
    response_override: str | None = None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '?', '!' followed by whitespace."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


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
        stripped = query.strip()

        if not stripped or len(stripped) < self.MIN_LENGTH:
            return GuardResult(allowed=False, reason="empty_query", response_override=REFUSAL_RESPONSE)

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

    def check(self, answer_text: str, retrieved_chunks: list[Chunk]) -> GuardResult:
        """
        Validate that answer_text is grounded in retrieved_chunks.

        Args:
            answer_text: the generated answer to check.
            retrieved_chunks: the chunks the answer was generated from.

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

        chunk_embeddings = self.embedder.encode([chunk.text for chunk in retrieved_chunks])
        sentence_embeddings = self.embedder.encode(sentences)

        unsupported_count = 0
        for sentence_embedding in sentence_embeddings:
            max_similarity = max(
                _cosine_similarity(sentence_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings
            )
            if max_similarity < self.grounding_threshold:
                unsupported_count += 1

        unsupported_ratio = unsupported_count / len(sentences)
        if unsupported_ratio > self.max_unsupported_ratio:
            return GuardResult(
                allowed=False,
                reason=f"ungrounded_answer:{unsupported_ratio:.2f}",
                response_override=REFUSAL_RESPONSE,
            )

        return GuardResult(allowed=True, reason="ok")
