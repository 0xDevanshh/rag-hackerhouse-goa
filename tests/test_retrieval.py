"""Focused tests for deterministic query routing and lexical retrieval."""

from src.chunking import Chunk
from src.retrieval import BM25Index, QueryRouter, normalize_query


def chunk(text: str, identifier: str) -> Chunk:
    return Chunk(text=text, metadata={"document_id": identifier}, strategy_name="test")


def test_normalize_query_handles_unicode_and_whitespace():
    assert normalize_query("  WHAT\u00a0is  Paris? ") == "what is paris?"


def test_query_router_uses_cheap_question_heuristics():
    router = QueryRouter()
    assert router.classify("What is Paris?") == "FACTUAL"
    assert router.classify("Why does rain happen?") == "CONTEXTUAL"
    assert router.classify("find documents about Paris") == "KEYWORD"


def test_bm25_prefers_exact_lexical_match():
    index = BM25Index(
        [
            chunk("Paris is the capital of France", "paris"),
            chunk("Cats purr when content", "cats"),
        ]
    )
    results = index.search("capital France", top_k=2)
    assert results
    assert results[0][0].metadata["document_id"] == "paris"
    assert results[0][1] > 0
