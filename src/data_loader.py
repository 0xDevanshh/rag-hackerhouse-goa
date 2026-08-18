"""
Data loading module.

Loads the ai4bharat/MSMARCO-XI dataset (streamed, to avoid downloading the
full ~55GB corpus) and maps raw examples into typed QueryDoc records. Pulled
examples are cached locally under data/msmarco_xi_cache/ so repeat runs with
the same (language, split, limit) don't re-hit the network.
"""

import itertools
from pathlib import Path

from datasets import load_dataset
from pydantic import BaseModel

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "msmarco_xi_cache"


class Passage(BaseModel):
    """A single retrieval passage for a query."""

    text: str
    is_selected: bool
    english_text: str


class QueryDoc(BaseModel):
    """A query and its candidate passages, mapped from a raw MSMARCO-XI example."""

    query_id: int
    query_text: str
    query_type: str
    language: str
    passages: list[Passage]


def _cache_path(language: str, split: str, limit: int) -> Path:
    """Build the cache file path for a given (language, split, limit) combination."""
    return CACHE_DIR / f"{language}_{split}_{limit}.jsonl"


def _example_to_query_doc(example: dict, language: str) -> QueryDoc:
    """
    Map a single raw MSMARCO-XI example into a QueryDoc.

    example["passages"] holds parallel arrays (is_selected, Translated_passages,
    English_passages), NOT a list of dicts, so they are zipped together into
    individual Passage objects.

    Args:
        example: a raw example dict as yielded by the streamed dataset.
        language: the MSMARCO-XI language config the example was loaded under
            (passed through as-is, since the dataset itself carries no short
            language code field per example).

    Returns:
        QueryDoc: the mapped record.
    """
    passages_raw = example["passages"]
    passages = [
        Passage(text=text, is_selected=bool(is_selected), english_text=english_text)
        for is_selected, text, english_text in zip(
            passages_raw["is_selected"],
            passages_raw["Translated_passages"],
            passages_raw["English_passages"],
        )
    ]
    return QueryDoc(
        query_id=example["query_id"],
        query_text=example["query"],
        query_type=example["query_type"],
        language=language,
        passages=passages,
    )


def load_corpus(language: str = "hi", split: str = "validation", limit: int = 500) -> list[QueryDoc]:
    """
    Load `limit` examples of the ai4bharat/MSMARCO-XI dataset for the given
    language config and split, mapped into QueryDoc records.

    Uses `load_dataset(..., streaming=True)` so only `limit` examples are
    pulled over the network via itertools.islice, instead of downloading the
    full corpus. Results are cached to
    data/msmarco_xi_cache/{language}_{split}_{limit}.jsonl on the first run;
    subsequent calls with the same arguments read from that cache file
    instead of streaming from the network again.

    Args:
        language: MSMARCO-XI language config name (e.g. "hi").
        split: dataset split to load (e.g. "validation").
        limit: maximum number of examples to pull and cache.

    Returns:
        list[QueryDoc]: the loaded query documents.
    """
    cache_path = _cache_path(language, split, limit)

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return [QueryDoc.model_validate_json(line) for line in f if line.strip()]

    dataset = load_dataset("ai4bharat/MSMARCO-XI", language, split=split, streaming=True)
    examples = itertools.islice(dataset, limit)
    query_docs = [_example_to_query_doc(example, language) for example in examples]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for doc in query_docs:
            f.write(doc.model_dump_json() + "\n")

    return query_docs
