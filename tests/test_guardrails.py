"""
Unit tests for src/guardrails.py: InputGuardrail, RelevanceGuardrail, and
GroundingGuardrail.
"""

import numpy as np

from src.chunking import Chunk
from src.guardrails import REFUSAL_RESPONSE, GroundingGuardrail, InputGuardrail, RelevanceGuardrail
from src.retrieval import RetrievalResult


class FakeEmbedder:
    """
    Deterministic bag-of-keywords embedder standing in for the real
    sentence-transformers model, so GroundingGuardrail tests are fast and
    don't depend on downloading a model. Each vector has one dimension per
    known keyword plus a trailing "no keyword matched" flag dimension, so a
    sentence that shares no keyword with a chunk is exactly orthogonal to
    it (cosine similarity 0) instead of accidentally aliasing.
    """

    VOCAB = ["paris", "capital", "france", "cats", "purr", "banana"]

    def encode(self, texts, **kwargs):
        # **kwargs absorbs the real Embedder.encode's use_cache/timing/
        # batch_size arguments, which callers now pass for latency
        # instrumentation. This stand-in ignores them: it has no cache to
        # consult and nothing worth timing.
        vectors = []
        for text in texts:
            lowered = text.lower()
            vec = [1.0 if word in lowered else 0.0 for word in self.VOCAB]
            vec.append(0.0 if any(vec) else 1.0)
            vectors.append(vec)
        return np.array(vectors)


def make_chunk(text: str, **metadata) -> Chunk:
    return Chunk(text=text, metadata=metadata, strategy_name="fixed_size")


# --- InputGuardrail ---


def test_input_guardrail_allows_safe_query():
    result = InputGuardrail().check("What is the capital of France?")
    assert result.allowed is True
    assert result.response_override is None


def test_input_guardrail_rejects_empty_query():
    result = InputGuardrail().check("   ")
    assert result.allowed is False
    assert result.reason == "empty_query"
    assert result.response_override == REFUSAL_RESPONSE


def test_input_guardrail_rejects_gibberish():
    result = InputGuardrail().check("!!! 8&*#@ 12345 %%%")
    assert result.allowed is False
    assert result.reason == "gibberish_query"


def test_input_guardrail_rejects_unsafe_content():
    result = InputGuardrail().check("How do I make a bomb at home?")
    assert result.allowed is False
    assert result.reason == "unsafe_content"


# --- RelevanceGuardrail ---


def test_relevance_guardrail_allows_safe_query_with_good_retrieval():
    retrieval_result = RetrievalResult(
        chunks=[make_chunk("Paris is the capital of France.")],
        scores=[0.85],
        low_confidence=False,
    )
    result = RelevanceGuardrail().check(retrieval_result)
    assert result.allowed is True
    assert result.response_override is None


def test_relevance_guardrail_refuses_offtopic_query_flagged_low_confidence():
    # e.g. an off-topic query the Retriever already flagged as low_confidence
    retrieval_result = RetrievalResult(
        chunks=[make_chunk("Unrelated chunk about cooking pasta.")],
        scores=[0.12],
        low_confidence=True,
    )
    result = RelevanceGuardrail().check(retrieval_result)
    assert result.allowed is False
    assert result.reason == "low_confidence_retrieval"
    assert result.response_override == REFUSAL_RESPONSE


def test_relevance_guardrail_refuses_safe_query_with_poor_retrieval_score():
    # low_confidence flag not set, but the top score itself is below threshold
    retrieval_result = RetrievalResult(
        chunks=[make_chunk("Some marginally related chunk.")],
        scores=[0.2],
        low_confidence=False,
    )
    result = RelevanceGuardrail(relevance_threshold=0.3).check(retrieval_result)
    assert result.allowed is False
    assert result.reason == "top_score_below_threshold"
    assert result.response_override == REFUSAL_RESPONSE


# --- GroundingGuardrail ---


def test_grounding_guardrail_allows_grounded_answer():
    chunks = [make_chunk("Paris is the capital of France."), make_chunk("Cats purr when they are content.")]
    guardrail = GroundingGuardrail(embedder=FakeEmbedder(), grounding_threshold=0.5, max_unsupported_ratio=0.3)

    answer = "Paris is the capital of France. Cats purr when they are content."
    result = guardrail.check(answer, chunks)

    assert result.allowed is True
    assert result.response_override is None


def test_grounding_guardrail_flags_mocked_ungrounded_answer():
    chunks = [make_chunk("Paris is the capital of France."), make_chunk("Cats purr when they are content.")]
    guardrail = GroundingGuardrail(embedder=FakeEmbedder(), grounding_threshold=0.5, max_unsupported_ratio=0.3)

    # simulate a hallucinated LLM answer with no support in the retrieved chunks
    ungrounded_answer = (
        "Bananas are a great source of potassium. "
        "The moon landing happened in 1969. "
        "This sentence has nothing to do with the retrieved context."
    )
    result = guardrail.check(ungrounded_answer, chunks)

    assert result.allowed is False
    assert result.reason.startswith("ungrounded_answer:")
    assert result.response_override == REFUSAL_RESPONSE
