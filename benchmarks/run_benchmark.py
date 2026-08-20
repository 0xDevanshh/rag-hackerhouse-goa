"""
Latency benchmark for the voice-rag pipeline.

Runs a mix of real MSMARCO-XI queries and adversarial/off-topic queries
through PipelineHarness with MockSTT substituted for real speech-to-text, so
the recorded latencies isolate in-process compute (embedding, FAISS
retrieval, guardrail checks, LLM generation) from network-bound STT.
Separately runs a handful of queries through the real SarvamSTT to report
its network latency on its own terms. Writes benchmarks/latency_report.md.

Indexes the same corpus src/api.py actually serves (MSMARCO-XI, not the
2-document demo corpus) — retrieval latency depends on corpus size, so
benchmarking against a 2-chunk toy corpus would say nothing about production.
"""

import asyncio
import io
import random
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src import data_loader, generation  # noqa: E402
from src.chunking import ChunkerRegistry  # noqa: E402
from src.generation import ExtractiveProvider, Generator  # noqa: E402
from src.harness import PipelineHarness, PipelineResult  # noqa: E402
from src.stt import MockSTT, SarvamSTT  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402

REPORT_PATH = ROOT / "benchmarks" / "latency_report.md"
RETRIEVAL_TARGET_MS = 200
POST_STT_TARGET_MS = 200
STT_SAMPLE_COUNT = 5

# Same corpus src/api.py serves by default — see its CORPUS_LANGUAGE/
# CORPUS_SPLIT/CORPUS_LIMIT/CORPUS_INCLUDE_ENGLISH env vars.
CORPUS_LANGUAGE = "hi"
CORPUS_SPLIT = "validation"
CORPUS_LIMIT = 500
CORPUS_INCLUDE_ENGLISH = True

NUM_IN_DOMAIN_QUERIES = 100

# Off-topic / adversarial: exercise the guardrail short-circuit paths.
# Corpus-independent, so these stay fixed regardless of which corpus is indexed.
ADVERSARIAL_QUERIES = [
    "What's a good recipe for chocolate chip cookies?",
    "What is the weather like in Paris today?",
    "Who won the football match last night?",
    "asdkjfh qwoeiru 12345 %%%&&&",
    "How do I make a bomb at home?",
    "Tell me a joke about cats.",
    "What's the capital of Australia?",
    "the a an in on of is are",
]


def _load_in_domain_queries(n: int = NUM_IN_DOMAIN_QUERIES) -> list[str]:
    """
    Pull n real, deduplicated query strings from MSMARCO-XI's own labeled
    cases — authentic user queries rather than hand-written ones, and
    guaranteed to be answerable against the corpus being indexed below.
    """
    cases = data_loader.load_eval_cases(
        language=CORPUS_LANGUAGE, split=CORPUS_SPLIT, limit=CORPUS_LIMIT, include_english=CORPUS_INCLUDE_ENGLISH
    )
    seen: set[str] = set()
    queries = []
    for case in cases:
        query = case["query"].strip()
        if query and query not in seen:
            seen.add(query)
            queries.append(query)
    random.Random(0).shuffle(queries)  # fixed seed: reproducible sample, not just the dataset's own ordering
    return queries[:n]


def _build_generator() -> tuple[Generator, str]:
    """Build the configured hosted provider or the local grounded provider."""
    provider = generation.get_provider()
    mode = "local-extractive" if isinstance(provider, ExtractiveProvider) else "hosted"
    return Generator(provider=provider), mode


def _load_corpus_chunks():
    """Chunk the same corpus src/api.py serves — one chunk per passage, matching production."""
    documents = data_loader.load_chunker_docs(
        language=CORPUS_LANGUAGE, split=CORPUS_SPLIT, limit=CORPUS_LIMIT, include_english=CORPUS_INCLUDE_ENGLISH
    )
    chunker = ChunkerRegistry.build("metadata_aware")
    return [chunk for doc in documents for chunk in chunker.chunk(doc)]


def _percentiles(values: list[float]) -> dict | None:
    """P50/P70/P100 (max) via linear interpolation on the sorted sample."""
    if not values:
        return None
    ordered = sorted(values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        k = (len(ordered) - 1) * (p / 100)
        floor_i, ceil_i = int(k), min(int(k) + 1, len(ordered) - 1)
        if floor_i == ceil_i:
            return ordered[floor_i]
        return ordered[floor_i] + (ordered[ceil_i] - ordered[floor_i]) * (k - floor_i)

    return {"p50": pct(50), "p70": pct(70), "p100": pct(100), "n": len(ordered)}


# Migrated from the old LatencyTrace to src/latency.RequestTrace. The stage
# names changed with it: 'stt' -> 'stt_network', 'input_guardrail' ->
# 'query_preprocessing', 'relevance_guardrail' -> 'relevance_guard', and the
# single 'retrieval' stage split into the four sub-spans below plus
# 'retrieval_overhead'. The old 'generation' stage likewise split into
# llm_network / llm_client_wait / llm_generation.
_RETRIEVAL_SPANS = (
    "embedding_cache",
    "embedding_compute",
    "vector_search",
    "bm25",
    "fusion",
    "reranking",
    "retrieval_overhead",
)
_GENERATION_SPANS = ("llm_network", "llm_client_wait", "llm_generation")
_RETRIEVAL_PIPELINE_SPANS = ("query_preprocessing", *_RETRIEVAL_SPANS, "relevance_guard")


def _span_sum(result: PipelineResult, names: tuple[str, ...]) -> float | None:
    """Total of the named spans, or None if none of them ran."""
    values = [v for v in (result.trace.get(name) for name in names) if v is not None]
    return sum(values) if values else None


def _stage_duration(result: PipelineResult, stage: str) -> float | None:
    if stage == "retrieval":
        return _span_sum(result, _RETRIEVAL_SPANS)
    if stage == "generation":
        return _span_sum(result, _GENERATION_SPANS)
    return result.trace.get(stage)


def _stage_failed(result: PipelineResult, stage: str) -> bool:
    return any(e.stage == stage for e in result.errors)


def _total_in_process_ms(result: PipelineResult) -> float:
    """
    "Post-STT total": every recorded span except the STT round trip, i.e.
    everything a caller waits on after transcription — retrieval, guardrails,
    AND generation. This is the number to check against an end-to-end
    post-STT latency target.
    """
    return sum(s.duration_ms for s in result.trace.spans if s.name != "stt_network")


def _retrieval_pipeline_ms(result: PipelineResult) -> float:
    """
    "Retrieval pipeline": input validation + embedding/FAISS search/rerank +
    the relevance check — i.e. everything up to the decision of whether to
    call the LLM at all. Deliberately excludes 'index_build' (one-time index
    build, not a per-query cost) and the LLM/grounding spans (the
    network-bound call and its post-hoc check).
    """
    return _span_sum(result, _RETRIEVAL_PIPELINE_SPANS) or 0.0


def _generate_silent_wav(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Build a minimal silent mono WAV clip in memory — no external audio asset needed."""
    buf = io.BytesIO()
    n_frames = int(duration_seconds * sample_rate)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


async def run_in_process_benchmark() -> dict:
    generator, generation_mode = _build_generator()
    store = VectorStore()
    mock_stt = MockSTT()
    corpus_chunks = _load_corpus_chunks()
    harness = PipelineHarness(store=store, chunks=corpus_chunks, generator=generator, stt_client=mock_stt)

    in_domain_queries = _load_in_domain_queries()
    all_queries = [(q, "in_domain") for q in in_domain_queries] + [(q, "adversarial") for q in ADVERSARIAL_QUERIES]

    # Warm-up run: absorbs one-time corpus indexing + model-loading cold
    # start, discarded from the measured stats below.
    mock_stt.transcripts["unknown"] = all_queries[0][0]
    await harness.run(b"warmup-audio")

    retrieval_durations, retrieval_pipeline_durations, generation_durations, post_stt_total_durations = [], [], [], []
    rows = []

    for i, (query_text, category) in enumerate(all_queries):
        mock_stt.transcripts["unknown"] = query_text
        result = await harness.run(b"dummy-audio-bytes")

        retrieval_ms = None if _stage_failed(result, "retrieval") else _stage_duration(result, "retrieval")
        generation_ms = None if _stage_failed(result, "generation") else _stage_duration(result, "generation")
        retrieval_pipeline_ms = _retrieval_pipeline_ms(result)
        post_stt_total_ms = _total_in_process_ms(result)

        if retrieval_ms is not None:
            retrieval_durations.append(retrieval_ms)
        if generation_ms is not None:
            generation_durations.append(generation_ms)
        retrieval_pipeline_durations.append(retrieval_pipeline_ms)
        post_stt_total_durations.append(post_stt_total_ms)

        rows.append(
            {
                "query": query_text,
                "category": category,
                "degraded": result.degraded,
                "guard_flags": {k: v.allowed for k, v in result.guard_flags.items()},
                "retrieval_pipeline_ms": retrieval_pipeline_ms,
                "post_stt_total_ms": post_stt_total_ms,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(all_queries)} queries done", file=sys.stderr)

    return {
        "generation_mode": generation_mode,
        "corpus_size": len(corpus_chunks),
        "retrieval": _percentiles(retrieval_durations),
        "generation": _percentiles(generation_durations),
        "retrieval_pipeline": _percentiles(retrieval_pipeline_durations),
        "post_stt_total": _percentiles(post_stt_total_durations),
        "rows": rows,
        "num_queries": len(all_queries),
    }


def run_stt_benchmark() -> dict:
    try:
        stt_client = SarvamSTT()
    except ValueError as exc:
        return {"available": False, "reason": str(exc)}

    audio_bytes = _generate_silent_wav()
    durations, errors = [], []
    for _ in range(STT_SAMPLE_COUNT):
        start = time.monotonic()
        try:
            stt_client.transcribe(audio_bytes, language_code="unknown")
            durations.append((time.monotonic() - start) * 1000)
        except Exception as exc:
            errors.append(str(exc))

    if not durations:
        reason = f"all {STT_SAMPLE_COUNT} calls failed: {errors[0] if errors else 'unknown error'}"
        return {"available": False, "reason": reason}

    return {"available": True, "percentiles": _percentiles(durations), "num_errors": len(errors)}


def _format_pct_row(label: str, pct: dict | None) -> str:
    if pct is None:
        return f"| {label} | n/a | n/a | n/a | 0 |"
    return f"| {label} | {pct['p50']:.1f} ms | {pct['p70']:.1f} ms | {pct['p100']:.1f} ms | {pct['n']} |"


def _verdict(pct: dict | None, target_ms: float) -> str:
    if pct is None:
        return "n/a"
    return "**PASS**" if pct["p100"] < target_ms else "**FAIL**"


def write_report(in_process: dict, stt: dict) -> None:
    n_domain = sum(1 for r in in_process["rows"] if r["category"] == "in_domain")
    n_adversarial = sum(1 for r in in_process["rows"] if r["category"] == "adversarial")

    lines = [
        "# Voice RAG Latency Benchmark",
        "",
        f"Ran {in_process['num_queries']} queries ({n_domain} real MSMARCO-XI queries, {n_adversarial} "
        "adversarial/off-topic) through `PipelineHarness` against the same corpus `src/api.py` serves "
        f"({in_process['corpus_size']} indexed chunks), with `MockSTT` substituted for real speech-to-text so "
        "the numbers below isolate in-process compute (embedding, FAISS retrieval, guardrail checks, LLM "
        "generation) from network-bound STT.",
        "",
    ]

    if in_process["generation_mode"] == "local-extractive":
        lines += [
            "> Generation used the local extractive provider. It performs no network call and returns only "
            "the highest-ranked retrieved passage with a citation, so these are honest local pipeline "
            "measurements rather than simulated LLM timings.",
            "",
        ]

    lines += [
        "## Verdict",
        "",
        "| Target | P100 (max) | Result |",
        "|---|---|---|",
        f"| Retrieval pipeline < {RETRIEVAL_TARGET_MS}ms (input guard + embed + FAISS + rerank + relevance guard, "
        f"excl. generation) | {in_process['retrieval_pipeline']['p100']:.1f}ms | "
        f"{_verdict(in_process['retrieval_pipeline'], RETRIEVAL_TARGET_MS)} |",
        f"| Post-STT total < {POST_STT_TARGET_MS}ms (everything after STT, **including** LLM generation) | "
        f"{in_process['post_stt_total']['p100']:.1f}ms | {_verdict(in_process['post_stt_total'], POST_STT_TARGET_MS)} |",
        "",
        "These are two different, deliberately separate claims — see \"Why two targets\" below before reading "
        "one as a substitute for the other.",
        "",
        "## Stage breakdown",
        "",
        "| Stage | P50 | P70 | P100 (max) | Samples |",
        "|---|---|---|---|---|",
        _format_pct_row("Retrieval (embed + FAISS + rerank) only", in_process["retrieval"]),
        _format_pct_row("Generation only", in_process["generation"]),
        _format_pct_row(f"**Retrieval pipeline** (target < {RETRIEVAL_TARGET_MS}ms)", in_process["retrieval_pipeline"]),
        _format_pct_row(f"**Post-STT total** (target < {POST_STT_TARGET_MS}ms)", in_process["post_stt_total"]),
        "",
        f"Sample counts below {in_process['num_queries']} for the \"only\" rows are expected: some adversarial "
        "queries are correctly short-circuited by InputGuardrail or RelevanceGuardrail before reaching "
        "retrieval or generation at all, so those stages simply weren't attempted for them (their duration is "
        "excluded, not counted as 0ms or as a failure). \"Retrieval pipeline\" and \"Post-STT total\" always "
        f"have the full {in_process['num_queries']} samples, since every query spends *some* time in the stages "
        "each one covers even when short-circuited early.",
        "",
        "## Speech-to-text (Sarvam API) — network-bound, separate from both targets",
        "",
    ]

    if not stt["available"]:
        lines += [f"STT benchmark skipped: {stt['reason']}", ""]
    else:
        pct = stt["percentiles"]
        lines += [
            f"Ran {STT_SAMPLE_COUNT} real calls to Sarvam's speech-to-text API on a 1-second silent WAV "
            f"clip ({stt['num_errors']} failed and were excluded from the percentiles below).",
            "",
            "| Metric | P50 | P100 (max) | Samples |",
            "|---|---|---|---|",
            f"| STT round-trip | {pct['p50']:.1f} ms | {pct['p100']:.1f} ms | {pct['n']} |",
            "",
        ]

    lines += [
        "## Why two targets",
        "",
        "\"Retrieval pipeline\" covers what this codebase actually controls: query validation, embedding, "
        "FAISS search, reranking, and the relevance check. This is the number an ANN index, a smaller "
        "embedding model, or a smarter cache would move — and the one that's realistic to hold under "
        f"{RETRIEVAL_TARGET_MS}ms regardless of corpus size (within reason).",
        "",
        "\"Post-STT total\" additionally includes LLM generation — a real network round trip to a third-party "
        "hosted model. No local optimization changes how long a remote GPU takes to generate tokens; "
        f"holding this under {POST_STT_TARGET_MS}ms is a claim about the LLM provider's latency, not about "
        "this codebase. If the verdict table above shows this target failing, that is expected and does not "
        "indicate a retrieval-side regression — check the retrieval pipeline verdict independently.",
        "",
        "STT is excluded from both because it precedes this pipeline entirely (the harness only starts timing "
        "after a query is already transcribed) — `MockSTT` removes it from these numbers by design, not as "
        "an oversight, and its real latency is reported separately above.",
        "",
        "## Per-query detail",
        "",
        "| Category | Degraded | Retrieval pipeline (ms) | Post-STT total (ms) | Query |",
        "|---|---|---|---|---|",
    ]
    for row in in_process["rows"]:
        query_display = row["query"].replace("|", "\\|")
        lines.append(
            f"| {row['category']} | {row['degraded']} | {row['retrieval_pipeline_ms']:.1f} | "
            f"{row['post_stt_total_ms']:.1f} | {query_display} |"
        )
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    in_process = await run_in_process_benchmark()
    stt = run_stt_benchmark()
    write_report(in_process, stt)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
