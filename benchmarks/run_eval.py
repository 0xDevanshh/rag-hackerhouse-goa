"""
Retrieval-quality evaluation against ai4bharat/MSMARCO-XI's own labels.

Builds the same index the API serves (MSMARCO-XI passages, chunked one per
passage in both the target language and English), then scores retrieval
against the dataset's `is_selected` relevance labels: hit rate@k, recall@k,
precision@k, and MRR, broken down by query language.

This complements benchmarks/run_benchmark.py rather than replacing it:
run_benchmark.py measures end-to-end *latency* (and needs STT/LLM keys to be
fully meaningful), while this measures *retrieval accuracy* and needs neither
— no API keys, no network beyond the one-time dataset cache, deterministic
output.

Writes data/eval_set.json (the labeled cases, so a run is reproducible and
the eval set is reviewable) and benchmarks/eval_report.md.

    python benchmarks/run_eval.py
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data_loader  # noqa: E402
from src.chunking import ChunkerRegistry  # noqa: E402
from src.harness import DEFAULT_K_VALUES, load_benchmark, report_results, run_benchmark  # noqa: E402
from src.retrieval import Retriever  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402

EVAL_SET_PATH = ROOT / "data" / "eval_set.json"
REPORT_PATH = ROOT / "benchmarks" / "eval_report.md"

LANGUAGE = "hi"
SPLIT = "validation"
LIMIT = 500
INCLUDE_ENGLISH = True

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
# basicConfig sets the *root* level, which also flips httpx/datasets/hub loggers
# to INFO — that buries this script's own output under hundreds of per-request
# lines while the dataset streams and the model loads.
for _noisy in ("httpx", "datasets", "huggingface_hub", "filelock", "sentence_transformers", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("run_eval")


def build_eval_set() -> list[dict]:
    """
    Generate the labeled eval set from MSMARCO-XI and write it to
    data/eval_set.json, returning the cases.

    Queries with no is_selected passage are dropped by
    QueryDoc.to_eval_cases — they carry no retrieval label, and scoring them
    would count an unlabeled query as a retrieval miss. In the Hindi
    validation slice that is roughly half of all rows, so the eval set is
    substantially smaller than `limit`.
    """
    cases = data_loader.load_eval_cases(
        language=LANGUAGE, split=SPLIT, limit=LIMIT, include_english=INCLUDE_ENGLISH
    )
    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_SET_PATH.open("w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    logger.info("wrote %d eval cases to %s", len(cases), EVAL_SET_PATH)
    return cases


def build_index() -> VectorStore:
    """
    Build the FAISS index over the full corpus slice.

    Every loaded query's passages are indexed, including those excluded from
    the eval set for lacking a label: they are legitimate passages and serve
    as distractors, so retrieval is scored against a realistically sized
    corpus rather than only labeled material.
    """
    documents = data_loader.load_chunker_docs(
        language=LANGUAGE, split=SPLIT, limit=LIMIT, include_english=INCLUDE_ENGLISH
    )
    chunker = ChunkerRegistry.build("metadata_aware")
    chunks = [chunk for doc in documents for chunk in chunker.chunk(doc)]

    logger.info("embedding and indexing %d chunks (this is the slow part)...", len(chunks))
    store = VectorStore()
    store.build(chunks)
    logger.info("index built: %d vectors", store.index.ntotal)
    return store


def main() -> None:
    build_eval_set()
    cases = load_benchmark(str(EVAL_SET_PATH))
    store = build_index()

    # top_n must cover the largest cutoff, and the rerank pool must be wider
    # still so metadata boosts have candidates to promote. Both are scaled up
    # because each passage is indexed once per language, so a pool of N chunks
    # holds only about N/2 distinct passages. Note this pool is wider than
    # the production default (RERANK_POOL_SIZE=10), so ranking at small k can
    # differ slightly from what the live API produces.
    max_k = max(DEFAULT_K_VALUES)
    pool_size, top_n = max_k * 4, max_k * 3

    # Headline numbers: is_selected_boost disabled, because that flag is the
    # ground-truth label being scored.
    honest = Retriever(store, rerank_pool_size=pool_size, top_n=top_n, is_selected_boost=0.0)
    results = run_benchmark(cases, honest, k_values=DEFAULT_K_VALUES)

    # Same eval with the boost left at its production default, to quantify how
    # much the label leak is worth rather than just asserting it leaks.
    leaky = Retriever(store, rerank_pool_size=pool_size, top_n=top_n)
    leaked = run_benchmark(cases, leaky, k_values=DEFAULT_K_VALUES)
    results["leaked_comparison"] = leaked["overall"]

    report_results(results, report_path=REPORT_PATH)
    logger.info("wrote %s", REPORT_PATH)


if __name__ == "__main__":
    main()
