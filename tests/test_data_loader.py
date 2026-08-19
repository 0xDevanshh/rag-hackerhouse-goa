"""
Unit tests for src/data_loader.py: parquet path resolution and the mapping
from raw MSMARCO-XI examples into QueryDoc / chunker docs / eval cases.

All tests run offline against synthetic examples shaped like the real ones
(verified against validation/hinval.parquet), so nothing here touches the Hub.
"""

import pytest

from src.data_loader import (
    ENGLISH_LANGUAGE,
    QueryDoc,
    _data_file,
    _example_to_query_doc,
)


def make_example(
    n_passages: int = 3,
    selected: tuple[int, ...] = (1,),
    answer: str = "निगम एक कानूनी संस्था है।",
    english_answer: str = "A corporation is a legal entity.",
) -> dict:
    """Build a raw example dict with the field names and parallel-array shape the dataset uses."""
    return {
        "query_id": 1102432,
        "query": "कॉर्पोरेशन क्या है?",
        "Eng_Query": "what is a corporation?",
        "Answer": answer,
        "Eng_Answer": english_answer,
        "query_type": "DESCRIPTION",
        "target_lang": "hin_Deva",
        "source_lang": "eng_Latn",
        "passages": {
            "is_selected": [1 if i in selected else 0 for i in range(n_passages)],
            "Translated_passages": [f"हिंदी अनुच्छेद {i}।" for i in range(n_passages)],
            "English_passages": [f"English passage {i}." for i in range(n_passages)],
        },
    }


# --- _data_file ---


@pytest.mark.parametrize(
    ("language", "split", "expected"),
    [
        ("hi", "validation", "validation/hinval.parquet"),
        ("hi", "train", "train/hintrain.parquet"),
        ("bn", "validation", "validation/benval.parquet"),
        ("ta", "train", "train/tamtrain.parquet"),
        ("te", "validation", "validation/telval.parquet"),
    ],
)
def test_data_file_resolves_language_and_split(language, split, expected):
    assert _data_file(language, split) == expected


def test_data_file_rejects_unknown_language():
    # Guards the original bug: the 3-letter file prefix is not the language
    # code, and the dataset has no per-language builder config at all.
    with pytest.raises(ValueError, match="Unknown language"):
        _data_file("hin", "validation")


def test_data_file_rejects_unknown_split():
    with pytest.raises(ValueError, match="Unknown split"):
        _data_file("hi", "test")


def test_data_file_rejects_telugu_train_which_has_no_parquet():
    with pytest.raises(ValueError, match="no 'train' split"):
        _data_file("te", "train")


# --- _example_to_query_doc ---


def test_example_maps_parallel_arrays_into_passages():
    doc = _example_to_query_doc(make_example(n_passages=3, selected=(1,)), "hi")

    assert doc.query_id == 1102432
    assert doc.language == "hi"
    assert doc.target_lang == "hin_Deva"
    assert doc.query_text == "कॉर्पोरेशन क्या है?"
    assert doc.english_query_text == "what is a corporation?"
    assert len(doc.passages) == 3
    assert doc.passages[1].is_selected is True
    assert doc.passages[0].is_selected is False
    assert doc.passages[2].english_text == "English passage 2."


def test_example_maps_is_selected_to_relevant_passage_ids():
    doc = _example_to_query_doc(make_example(n_passages=5, selected=(0, 3)), "hi")
    assert doc.relevant_passage_ids == [0, 3]


def test_ragged_passage_arrays_truncate_to_shortest_and_warn(caplog):
    example = make_example(n_passages=3)
    example["passages"]["English_passages"] = ["only one"]

    with caplog.at_level("WARNING"):
        doc = _example_to_query_doc(example, "hi")

    assert len(doc.passages) == 1
    assert "ragged passage arrays" in caplog.text


# --- to_chunker_docs ---


def test_to_chunker_docs_emits_one_document_per_language():
    doc = _example_to_query_doc(make_example(n_passages=3, selected=(2,)), "hi")
    chunker_docs = doc.to_chunker_docs()

    assert [d["language"] for d in chunker_docs] == ["hi", ENGLISH_LANGUAGE]
    assert chunker_docs[0]["query_text"] == "कॉर्पोरेशन क्या है?"
    assert chunker_docs[1]["query_text"] == "what is a corporation?"
    assert chunker_docs[0]["passages"][0]["text"] == "हिंदी अनुच्छेद 0।"
    assert chunker_docs[1]["passages"][0]["text"] == "English passage 0."


def test_to_chunker_docs_keeps_passage_order_aligned_across_languages():
    # passage_id is positional, and the eval set's relevant_passage_ids are
    # scored against it, so the two languages must not diverge in order.
    doc = _example_to_query_doc(make_example(n_passages=4, selected=(2,)), "hi")
    hi_doc, en_doc = doc.to_chunker_docs()

    hi_selected = [i for i, p in enumerate(hi_doc["passages"]) if p["is_selected"]]
    en_selected = [i for i, p in enumerate(en_doc["passages"]) if p["is_selected"]]
    assert hi_selected == en_selected == [2]


def test_to_chunker_docs_can_omit_english():
    doc = _example_to_query_doc(make_example(), "hi")
    chunker_docs = doc.to_chunker_docs(include_english=False)
    assert [d["language"] for d in chunker_docs] == ["hi"]


def test_chunker_docs_feed_metadata_aware_chunker():
    from src.chunking import ChunkerRegistry

    doc = _example_to_query_doc(make_example(n_passages=3, selected=(1,)), "hi")
    chunker = ChunkerRegistry.build("metadata_aware")
    chunks = [c for d in doc.to_chunker_docs() for c in chunker.chunk(d)]

    assert len(chunks) == 6  # 3 passages x 2 languages
    assert {c.metadata["language"] for c in chunks} == {"hi", "en"}
    # The same passage_id refers to the same passage in both languages.
    selected = {(c.metadata["language"], c.metadata["passage_id"]) for c in chunks if c.metadata["is_selected"]}
    assert selected == {("hi", 1), ("en", 1)}


# --- to_eval_cases ---


def test_to_eval_cases_emits_one_case_per_language():
    doc = _example_to_query_doc(make_example(n_passages=3, selected=(1,)), "hi")
    cases = doc.to_eval_cases()

    assert [c["language"] for c in cases] == ["hi", ENGLISH_LANGUAGE]
    assert all(c["relevant_passage_ids"] == [1] for c in cases)
    assert cases[0]["expected_answer"] == "निगम एक कानूनी संस्था है।"
    assert cases[1]["expected_answer"] == "A corporation is a legal entity."


def test_to_eval_cases_skips_rows_with_no_relevance_label():
    # Roughly half the Hindi validation split has no is_selected passage;
    # scoring those would count an unlabeled query as a retrieval miss.
    doc = _example_to_query_doc(make_example(n_passages=3, selected=()), "hi")
    assert doc.relevant_passage_ids == []
    assert doc.to_eval_cases() == []


def test_to_eval_cases_nulls_out_no_answer_present_marker():
    doc = _example_to_query_doc(
        make_example(selected=(0,), answer="No Answer Present.", english_answer="No Answer Present."),
        "hi",
    )
    cases = doc.to_eval_cases()
    assert all(c["expected_answer"] is None for c in cases)
    # Still a usable retrieval case — only the answer reference is missing.
    assert all(c["relevant_passage_ids"] == [0] for c in cases)


def test_query_doc_round_trips_through_json_cache():
    # load_corpus caches as one JSON line per doc, so the model must survive it.
    doc = _example_to_query_doc(make_example(n_passages=3, selected=(1,)), "hi")
    restored = QueryDoc.model_validate_json(doc.model_dump_json())
    assert restored == doc
