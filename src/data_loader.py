"""
Data loading module.

Loads the ai4bharat/MSMARCO-XI dataset (streamed, to avoid downloading the
full ~55GB corpus) and maps raw examples into typed QueryDoc records. Pulled
examples are cached locally under data/msmarco_xi_cache/ so repeat runs with
the same (language, split, limit) don't re-hit the network.

Language selection is by *file*, not by config. The dataset exposes exactly
one builder config ("default"), and stores one parquet per language, named
with a 3-letter code: validation/hinval.parquet, train/hintrain.parquet, etc.
Passing a language as the config name (load_dataset(..., "hi", ...)) raises
`ValueError: BuilderConfig 'hi' not found. Available: ['default']`, and
passing no config at all silently concatenates every language in file order
— so the first examples come back Assamese, not Hindi. Both failure modes
are why this module addresses the parquet directly via `data_files`.
"""

import itertools
import logging
from pathlib import Path

from datasets import load_dataset
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DATASET_ID = "ai4bharat/MSMARCO-XI"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "msmarco_xi_cache"

# Short (ISO 639-1) language code -> the 3-letter prefix used in the dataset's
# parquet filenames. Only these languages exist in the repo.
LANGUAGE_FILE_PREFIXES = {
    "as": "asm",  # Assamese
    "bn": "ben",  # Bengali
    "gu": "guj",  # Gujarati
    "hi": "hin",  # Hindi
    "kn": "kan",  # Kannada
    "ml": "mal",  # Malayalam
    "mr": "mar",  # Marathi
    "ne": "nep",  # Nepali
    "or": "ori",  # Odia
    "pa": "pan",  # Punjabi
    "sa": "san",  # Sanskrit
    "ta": "tam",  # Tamil
    "te": "tel",  # Telugu — validation only, see TRAIN_ONLY_MISSING
    "ur": "urd",  # Urdu
}

# The filename suffix differs per split: "hinval.parquet" vs "hintrain.parquet".
SPLIT_FILE_SUFFIXES = {"validation": "val", "train": "train"}

# Telugu ships a validation parquet but no train parquet, so ("te", "train")
# has no file to read and must fail loudly rather than 404 mid-stream.
MISSING_SPLIT_LANGUAGES = {("te", "train")}

# The language code used for the original (untranslated) English side of each
# example, carried in Eng_Query / Eng_Answer / passages.English_passages.
ENGLISH_LANGUAGE = "en"

# MS MARCO marks unanswerable queries with this literal answer string rather
# than an empty field; such rows are useless as answer-quality references.
NO_ANSWER_MARKER = "no answer present."


class Passage(BaseModel):
    """A single retrieval passage for a query, in both the target language and English."""

    text: str
    is_selected: bool
    english_text: str


class QueryDoc(BaseModel):
    """A query and its candidate passages, mapped from a raw MSMARCO-XI example."""

    query_id: int
    query_text: str
    english_query_text: str
    answer: str
    english_answer: str
    query_type: str
    language: str
    target_lang: str
    passages: list[Passage]

    @property
    def relevant_passage_ids(self) -> list[int]:
        """Indices of passages flagged is_selected — the dataset's relevance labels."""
        return [i for i, passage in enumerate(self.passages) if passage.is_selected]

    def _passage_dicts(self, language: str) -> list[dict]:
        attr = "english_text" if language == ENGLISH_LANGUAGE else "text"
        return [
            {"text": getattr(passage, attr), "is_selected": passage.is_selected}
            for passage in self.passages
        ]

    def to_chunker_docs(self, include_english: bool = True) -> list[dict]:
        """
        Flatten this record into per-language documents shaped for
        MetadataAwareChunker (see src/chunking.py), which emits one chunk per
        passage tagged with query_id/passage_id/language/is_selected.

        Emitting one document per language (rather than one bilingual
        document) keeps MetadataAwareChunker unchanged, and makes the
        `language` tag on each chunk accurate — which is what
        Retriever's language-match boost and VectorStore's metadata filter
        key off. Passage order is identical across languages, so a chunk's
        passage_id means the same thing in both and lines up with
        relevant_passage_ids.

        Args:
            include_english: also emit the original English side as its own
                document, tagged language="en".

        Returns:
            list[dict]: one document per indexed language.
        """
        languages = [self.language] + ([ENGLISH_LANGUAGE] if include_english else [])
        query_texts = {self.language: self.query_text, ENGLISH_LANGUAGE: self.english_query_text}
        return [
            {
                "query_id": self.query_id,
                "query_text": query_texts[language],
                "language": language,
                "passages": self._passage_dicts(language),
            }
            for language in languages
        ]

    def to_eval_cases(self, include_english: bool = True) -> list[dict]:
        """
        Build labeled evaluation cases from this record, using is_selected as
        retrieval ground truth and Answer/Eng_Answer as the reference answer.

        Rows with no is_selected passage carry no retrieval label, so they're
        skipped entirely — scoring them would count an unanswerable query as
        a retrieval miss. `expected_answer` is None for MS MARCO's
        "No Answer Present." rows, which are still valid retrieval cases.

        Args:
            include_english: also emit the English-side query as its own case.

        Returns:
            list[dict]: eval cases, empty if this record has no relevance label.
        """
        relevant = self.relevant_passage_ids
        if not relevant:
            return []

        languages = [self.language] + ([ENGLISH_LANGUAGE] if include_english else [])
        queries = {self.language: self.query_text, ENGLISH_LANGUAGE: self.english_query_text}
        answers = {self.language: self.answer, ENGLISH_LANGUAGE: self.english_answer}

        cases = []
        for language in languages:
            answer = answers[language].strip()
            cases.append(
                {
                    "query_id": self.query_id,
                    "query": queries[language],
                    "language": language,
                    "query_type": self.query_type,
                    "expected_answer": None if answer.lower() == NO_ANSWER_MARKER else answer,
                    "relevant_passage_ids": relevant,
                }
            )
        return cases


def _data_file(language: str, split: str) -> str:
    """
    Resolve the in-repo parquet path for a (language, split) pair, e.g.
    ("hi", "validation") -> "validation/hinval.parquet".

    Raises:
        ValueError: if the language or split is unknown, or the pair has no
            parquet in the dataset (e.g. Telugu has no train split).
    """
    if language not in LANGUAGE_FILE_PREFIXES:
        raise ValueError(
            f"Unknown language {language!r}. Available: {sorted(LANGUAGE_FILE_PREFIXES)}"
        )
    if split not in SPLIT_FILE_SUFFIXES:
        raise ValueError(f"Unknown split {split!r}. Available: {sorted(SPLIT_FILE_SUFFIXES)}")
    if (language, split) in MISSING_SPLIT_LANGUAGES:
        raise ValueError(f"{DATASET_ID} has no {split!r} split for language {language!r}")

    return f"{split}/{LANGUAGE_FILE_PREFIXES[language]}{SPLIT_FILE_SUFFIXES[split]}.parquet"


def _cache_path(language: str, split: str, limit: int) -> Path:
    """Build the cache file path for a given (language, split, limit) combination."""
    return CACHE_DIR / f"{language}_{split}_{limit}.jsonl"


def _example_to_query_doc(example: dict, language: str) -> QueryDoc:
    """
    Map a single raw MSMARCO-XI example into a QueryDoc.

    example["passages"] holds parallel arrays (is_selected,
    Translated_passages, English_passages), NOT a list of dicts, so they are
    zipped together into individual Passage objects. The three arrays are
    expected to be the same length; a ragged example is truncated to the
    shortest (zip's default) and logged, since dropping the unpaired tail is
    preferable to crashing a long stream over one malformed row.

    Args:
        example: a raw example dict as yielded by the streamed dataset.
        language: the short language code the example was loaded for. The
            example's own `target_lang` field carries a FLORES-style code
            (e.g. "hin_Deva"), which is kept separately on the QueryDoc.

    Returns:
        QueryDoc: the mapped record.
    """
    passages_raw = example["passages"]
    is_selected = passages_raw["is_selected"]
    translated = passages_raw["Translated_passages"]
    english = passages_raw["English_passages"]

    if not len(is_selected) == len(translated) == len(english):
        logger.warning(
            "query_id=%s has ragged passage arrays (is_selected=%d, translated=%d, english=%d); "
            "truncating to the shortest",
            example["query_id"],
            len(is_selected),
            len(translated),
            len(english),
        )

    passages = [
        Passage(text=text, is_selected=bool(selected), english_text=english_text)
        for selected, text, english_text in zip(is_selected, translated, english)
    ]
    return QueryDoc(
        query_id=example["query_id"],
        query_text=example["query"],
        english_query_text=example["Eng_Query"],
        answer=example["Answer"],
        english_answer=example["Eng_Answer"],
        query_type=example["query_type"],
        language=language,
        target_lang=example["target_lang"],
        passages=passages,
    )


def load_corpus(language: str = "hi", split: str = "validation", limit: int = 500) -> list[QueryDoc]:
    """
    Load `limit` examples of the ai4bharat/MSMARCO-XI dataset for the given
    language and split, mapped into QueryDoc records.

    The language's parquet is addressed directly via `data_files` (see the
    module docstring for why language-as-config does not work), and read with
    `streaming=True` so only `limit` examples cross the network via
    itertools.islice instead of downloading the full corpus. Results are
    cached to data/msmarco_xi_cache/{language}_{split}_{limit}.jsonl on the
    first run; subsequent calls with the same arguments read that file
    instead of streaming again.

    Args:
        language: short language code, e.g. "hi" (see LANGUAGE_FILE_PREFIXES).
        split: "validation" or "train".
        limit: maximum number of examples to pull and cache.

    Returns:
        list[QueryDoc]: the loaded query documents.

    Raises:
        ValueError: if (language, split) has no parquet in the dataset.
    """
    data_file = _data_file(language, split)
    cache_path = _cache_path(language, split, limit)

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return [QueryDoc.model_validate_json(line) for line in f if line.strip()]

    logger.info("streaming %d examples from %s:%s", limit, DATASET_ID, data_file)
    dataset = load_dataset(
        DATASET_ID,
        data_files={split: data_file},
        split=split,
        streaming=True,
    )
    examples = itertools.islice(dataset, limit)
    query_docs = [_example_to_query_doc(example, language) for example in examples]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for doc in query_docs:
            f.write(doc.model_dump_json() + "\n")

    return query_docs


def load_chunker_docs(
    language: str = "hi",
    split: str = "validation",
    limit: int = 500,
    include_english: bool = True,
) -> list[dict]:
    """
    Convenience wrapper: load_corpus() flattened into per-language documents
    ready for MetadataAwareChunker. See QueryDoc.to_chunker_docs.
    """
    docs = load_corpus(language=language, split=split, limit=limit)
    return [doc for query_doc in docs for doc in query_doc.to_chunker_docs(include_english)]


def load_eval_cases(
    language: str = "hi",
    split: str = "validation",
    limit: int = 500,
    include_english: bool = True,
) -> list[dict]:
    """
    Convenience wrapper: load_corpus() flattened into labeled eval cases.
    See QueryDoc.to_eval_cases.
    """
    docs = load_corpus(language=language, split=split, limit=limit)
    return [case for query_doc in docs for case in query_doc.to_eval_cases(include_english)]
