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
from src.text import NO_CONTEXT_RESPONSE
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


# The citation form SYSTEM_PROMPT instructs the model to emit. Stripped before
# embedding, since it is boilerplate this pipeline asked for rather than
# content: leaving it in dilutes a short sentence's vector and depresses its
# similarity to the passage that actually supports it.
_CITATION_RE = re.compile(r"\s*\[passage_id:\s*[^\]]*\]", re.IGNORECASE)


def _is_declination(answer_text: str) -> bool:
    """
    Is this the model saying the context doesn't cover the question, rather
    than an attempted answer?

    Compared against text.NO_CONTEXT_RESPONSE, the exact sentence
    SYSTEM_PROMPT tells the model to emit in that case. Matching the shared
    constant rather than a copy of the string means a reword of the prompt
    can't silently desynchronise the two. Citations, whitespace, case, and a
    trailing period are normalised away, since the model reproduces the
    sentence closely but not always byte-exactly.
    """
    normalized = " ".join(_CITATION_RE.sub(" ", answer_text).split()).strip().rstrip(".").casefold()
    expected = " ".join(NO_CONTEXT_RESPONSE.split()).strip().rstrip(".").casefold()
    return normalized == expected


def _strip_citations(sentences: list[str]) -> list[str]:
    """
    Drop citation markers from each sentence for embedding purposes, keeping
    the original text if a sentence turns out to be nothing but a citation
    (embedding an empty string would score as unsupported against everything).
    """
    stripped = []
    for sentence in sentences:
        without = _CITATION_RE.sub(" ", sentence).strip()
        stripped.append(" ".join(without.split()) if without else sentence)
    return stripped


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
        max_drift_ratio: float = 0.5,
    ):
        """
        Args:
            embedder: Embedder used to encode answer sentences and chunk
                texts. Defaults to a new Embedder() (real model) if omitted.
            grounding_threshold: minimum max-cosine-similarity to a
                retrieved chunk for a sentence to count as supported.
            max_unsupported_ratio: retained for callers that construct this
                explicitly, and still the threshold is_sentence_grounded's
                per-sentence policy is derived from. No longer used to discard
                a whole answer — see check(), which now drops the unsupported
                sentences instead.
            max_drift_ratio: fraction of unsupported sentences above which the
                answer is refused wholesale rather than filtered. Above this,
                enough of the answer was unsupported that returning the
                remainder would mislead by omission.
        """
        self.embedder = embedder or Embedder()
        self.grounding_threshold = grounding_threshold
        self.max_unsupported_ratio = max_unsupported_ratio
        self.max_drift_ratio = max_drift_ratio

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

    def _similarities(
        self,
        sentences: list[str],
        retrieved_chunks: list[Chunk],
        chunk_vectors: np.ndarray,
        timing: dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Best support score for each answer sentence, measured in two passes so
        accuracy and latency don't have to be traded against each other.

        Pass 1 compares each sentence against the *whole-chunk* vectors, which
        the caller gets free out of the FAISS index. Cheap, but systematically
        pessimistic: a one-sentence paraphrase of one sentence inside a
        300-character passage is diluted by everything else in that passage.
        Measured on a correct, fully-grounded three-sentence answer, pass 1
        alone scored 0.600 / 0.702 / 0.505 against a 0.5 threshold — passing,
        but by so little that any wording drift tips a sentence under and the
        *entire* answer is discarded. That is the intermittent
        "ungrounded_answer" refusal on questions whose answer was sitting in
        the retrieved context.

        Pass 2 re-scores only the sentences that failed pass 1, against the
        individual sentences of the retrieved chunks. Same three sentences
        score 0.721 / 0.890 / 1.000 that way. Crucially this sharpens the
        measurement rather than relaxing the policy: hallucinated sentences
        ("RAG was invented at Stanford in 1998", "Napoleon was crowned
        Emperor") score 0.133 / 0.395 / 0.047 under pass 2 and are still
        refused, so the grounded/ungrounded separation widens in both
        directions.

        Because pass 2 runs only for sentences that would otherwise be
        rejected, a comfortably-grounded answer still costs zero extra
        embedding work.
        """
        similarities = _max_similarities(self.embedder.encode(sentences, timing=timing), chunk_vectors)

        weak = [i for i, score in enumerate(similarities) if score < self.grounding_threshold]
        if not weak:
            return similarities

        chunk_sentences = [
            sentence for chunk in retrieved_chunks for sentence in _split_sentences(chunk.text)
        ]
        # Only worth a second pass if the chunks actually decompose into more
        # than what pass 1 already compared against.
        if len(chunk_sentences) <= len(retrieved_chunks):
            return similarities

        chunk_sentence_vectors = self.embedder.encode(chunk_sentences, timing=timing)
        weak_vectors = self.embedder.encode([sentences[i] for i in weak], timing=timing)
        rescored = _max_similarities(weak_vectors, chunk_sentence_vectors)
        for index, score in zip(weak, rescored):
            similarities[index] = max(similarities[index], score)
        return similarities

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

        # The model declining, per SYSTEM_PROMPT's instruction to answer with
        # exactly NO_CONTEXT_RESPONSE when the context doesn't cover the
        # question. Recognized here so it isn't run through the hallucination
        # check, which it can only ever fail: it is a statement *about* the
        # context rather than a claim drawn from it, so it scores near-zero
        # against every passage and gets reported as
        # "ungrounded_answer:1.00" — indistinguishable in the logs from
        # catching a genuine fabrication, when the model in fact did exactly
        # the right thing. The user-facing text is still the shared canned
        # refusal, so which layer refused stays invisible to them.
        if _is_declination(answer_text):
            return GuardResult(
                allowed=False, reason="model_declined", response_override=REFUSAL_RESPONSE
            )

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
        similarities = self._similarities(
            _strip_citations(sentences), retrieved_chunks, chunk_vectors, timing
        )
        supported = similarities >= self.grounding_threshold
        unsupported_count = int((~supported).sum())
        unsupported_ratio = unsupported_count / len(sentences)

        # Nothing grounded at all, or the model drifted for most of the answer:
        # refuse outright, since what little remains can mislead by omission.
        if unsupported_count == len(sentences) or unsupported_ratio > self.max_drift_ratio:
            return GuardResult(
                allowed=False,
                reason=f"ungrounded_answer:{unsupported_ratio:.2f}",
                response_override=REFUSAL_RESPONSE,
            )

        # Otherwise drop just the unsupported sentences and keep the rest.
        #
        # This replaces an all-or-nothing whole-answer verdict, which was
        # degenerate for short answers: at max_unsupported_ratio=0.3 a
        # three-sentence answer tolerated *zero* unsupported sentences
        # (1/3 = 0.33 > 0.3), so one connective or summarising sentence
        # discarded two perfectly grounded ones and the user was told the
        # dataset had no answer.
        #
        # Dropping is the stricter guarantee on what actually reaches the user
        # — no ungrounded sentence is returned at all, where the old rule
        # passed through up to 30% of them whenever the answer was long enough
        # — and it is the rule PipelineHarness.run_streaming already applies
        # per sentence. The cost is that removing a sentence can leave the
        # remainder reading a little abruptly.
        if unsupported_count:
            kept = [sentence for sentence, ok in zip(sentences, supported) if ok]
            return GuardResult(
                allowed=True,
                reason=f"dropped_{unsupported_count}_ungrounded_sentence(s)",
                response_override=" ".join(kept),
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
        # Same two-pass measurement as check(), so the streaming path applies
        # the same standard. It stays stricter in *policy* — every sentence
        # must pass here, versus check()'s 30% tolerance — but it must not be
        # stricter by accident, through a coarser similarity measure that
        # rejects correctly-grounded sentences check() would have kept.
        similarities = self._similarities(
            _strip_citations([sentence]), retrieved_chunks, chunk_vectors, timing
        )
        return bool(similarities[0] >= self.grounding_threshold)
