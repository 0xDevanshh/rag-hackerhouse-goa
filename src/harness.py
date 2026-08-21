"""
Harness module.

Two responsibilities:
1. PipelineHarness: orchestrates one end-to-end voice-RAG request (STT ->
   InputGuardrail -> chunking/retrieval -> RelevanceGuardrail ->
   Generator.answer -> GroundingGuardrail -> final response), with
   structured per-stage error handling, retries on network-bound stages,
   latency tracing, and a graceful degraded fallback instead of crashing.
2. Benchmark evaluation helpers (load_benchmark/run_benchmark/report_results):
   run the pipeline over a labeled benchmark dataset and report aggregate
   metrics such as retrieval accuracy, groundedness, and latency.
"""

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import numpy as np
import requests
from pydantic import BaseModel, Field
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from src import stt
from src.chunking import Chunk
from src.generation import DEFAULT_MAX_TOKENS, SYSTEM_PROMPT, Generator, _build_user_message
from src.guardrails import (
    REFUSAL_RESPONSE,
    GroundingGuardrail,
    GuardResult,
    InputGuardrail,
    RelevanceGuardrail,
    normalize_query_input,
)
from src.latency import RequestTrace
from src.retrieval import Retriever, RetrievalResult
from src.text import SENTENCE_BOUNDARY_RE as _SENTENCE_BOUNDARY_RE
from src.vectorstore import DEFAULT_EMBEDDING_MODEL_NAME, VectorStore, text_fingerprint

logger = logging.getLogger(__name__)

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
_RESULT_CACHE_MAX_SIZE = int(os.environ.get("RESULT_CACHE_MAX_SIZE", "512"))

# Bumped by hand whenever SYSTEM_PROMPT or the context-block format changes in
# a way that would make a previously cached answer wrong to serve. Part of the
# answer cache key: without it, editing the prompt leaves every prior answer
# in the cache being served as if it had been generated under the new prompt.
PROMPT_VERSION = "1"

# Retrieval sub-timings that Retriever.retrieve reports and that the trace
# records as flat, non-overlapping spans.
_RETRIEVAL_SPANS = (
    "embedding_cache",
    "embedding_compute",
    "vector_search",
    "bm25",
    "fusion",
    "reranking",
)


# Note on normalization: the answer cache keys off text_fingerprint(), which
# applies src.vectorstore.normalize_text() before hashing. That is deliberate
# sharing rather than duplication — the answer cache and the embedding cache
# must agree on what counts as "the same query", or a query could hit one and
# miss the other.


def _is_transient_stt_error(exc: BaseException) -> bool:
    """Network hiccups and rate limits/server errors are worth retrying; other failures aren't."""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        return status is not None and (status == 429 or status >= 500)
    return False


def _is_audio_path(value: str) -> bool:
    """Heuristic: does this string look like a path to an audio file?"""
    _, ext = os.path.splitext(value)
    return ext.lower() in _AUDIO_EXTENSIONS


class StageError(BaseModel):
    """Structured error captured when a pipeline stage fails."""

    stage: str
    error_type: str
    message: str


class PipelineResult(BaseModel):
    """The outcome of one PipelineHarness.run() call."""

    answer: str
    query_text: str = ""
    sources: list[Chunk] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    # Full-lifecycle wall-clock trace. Replaces the previous LatencyTrace,
    # whose total_ms was defined as the sum of the stages it happened to
    # measure — an arithmetic that could not, even in principle, surface the
    # ~958ms of real request time that was going unattributed. See
    # src/latency.py.
    trace: RequestTrace = Field(default_factory=RequestTrace)
    guard_flags: dict[str, GuardResult] = Field(default_factory=dict)
    degraded: bool = False
    errors: list[StageError] = Field(default_factory=list)
    # True when this response came from the answer cache rather than from a
    # fresh retrieval + generation.
    cached: bool = False


class StreamSentenceEvent(BaseModel):
    """One incrementally-generated, individually grounding-checked sentence."""

    event: Literal["sentence"] = "sentence"
    text: str


class StreamDoneEvent(BaseModel):
    """Final event of a PipelineHarness.run_streaming() call, carrying the completed PipelineResult."""

    event: Literal["done"] = "done"
    result: PipelineResult


class PipelineHarness:
    """
    Orchestrates one end-to-end voice-RAG request.

    Stages, in order: STT (only if the input is audio) -> InputGuardrail ->
    chunking/retrieval (chunks are indexed into `store` once, then skipped
    on subsequent calls) -> RelevanceGuardrail -> Generator.answer ->
    GroundingGuardrail -> final response. Every stage is wrapped in
    try/except and timed; a guardrail refusal short-circuits the remaining
    stages and returns its canned response (not a degraded result — that's
    the guardrail working as designed). A stage that fails technically
    (after its retry budget, for STT and generation) produces a degraded
    fallback response instead of propagating the exception.

    Generation retries are handled inside Generator/LLMProvider itself
    (see src/generation.py) via GenerationError; this harness does not wrap
    Generator.answer in a second retry layer, to avoid multiplying retries
    across two independent layers. STT has no such internal retry, so this
    harness owns STT's retry policy directly.
    """

    def __init__(
        self,
        store: VectorStore,
        chunks: list[Chunk] | None = None,
        generator: Generator | None = None,
        stt_client: stt.SarvamSTT | stt.MockSTT | None = None,
        input_guardrail: InputGuardrail | None = None,
        relevance_guardrail: RelevanceGuardrail | None = None,
        grounding_guardrail: GroundingGuardrail | None = None,
        max_retries: int = 2,
        retry_wait_multiplier: float = 1.0,
        scope: str = "public",
    ):
        """
        Args:
            store: the VectorStore to retrieve against. If it isn't already
                indexed (ntotal == 0), `chunks` is indexed into it on the
                first run() call.
            chunks: chunks to index into `store` if it's empty. Unused (and
                may be omitted) if `store` is already indexed.
            generator: the Generator to use. Defaults to Generator() (reads
                LLM_PROVIDER from the environment).
            stt_client: a SarvamSTT or MockSTT instance. If omitted, a
                SarvamSTT() is constructed lazily the first time audio input
                actually needs transcribing (so text-only usage never
                requires a SARVAM_API_KEY). Inject a MockSTT for tests.
            input_guardrail, relevance_guardrail, grounding_guardrail:
                guardrail instances to use. Default to their own defaults.
            max_retries: retries (beyond the first attempt) for the STT
                stage, e.g. max_retries=2 means up to 3 total attempts.
            retry_wait_multiplier: exponential-backoff base multiplier for
                STT retries, in seconds. Lower this in tests to avoid slow
                sleeps.
            scope: isolation scope mixed into every answer cache key. This
                project has no authentication or multi-tenancy, so the
                default single "public" scope is the honest description of
                what it serves — but the parameter exists so that adding a
                per-user, per-workspace, or per-repository scope later is a
                matter of passing it in, not of retrofitting isolation onto
                keys that never had it. See _result_cache_key.
        """
        self.store = store
        self.chunks = chunks or []
        self.generator = generator or Generator()
        self.stt_client = stt_client
        self.input_guardrail = input_guardrail or InputGuardrail()
        self.relevance_guardrail = relevance_guardrail or RelevanceGuardrail()
        self.grounding_guardrail = grounding_guardrail or GroundingGuardrail()
        self.retriever = Retriever(store)
        self.max_retries = max_retries
        self.retry_wait_multiplier = retry_wait_multiplier
        self.scope = scope
        # Exact-query answer cache: skips retrieval+guardrails+generation
        # entirely for a repeated (already-transcribed) query text.
        #
        # Never caches degraded results (transient technical failures should
        # be retried, not replayed). LRU rather than FIFO, so a genuinely hot
        # query isn't evicted on schedule by a stream of one-off ones.
        #
        # Process-local: this project has no Redis. That is a real limitation
        # for a multi-instance deployment (each instance misses on the others'
        # hits, and the effective hit rate divides by the instance count), and
        # the fix is a shared store, not a bigger local dict. The key is
        # already fully explicit — see _result_cache_key — so it ports as-is.
        self._result_cache: "OrderedDict[tuple, PipelineResult]" = OrderedDict()
        self._index_version = 0

    def _result_cache_key(self, query_text: str, scope: str | None = None) -> tuple:
        """
        Fully-qualified answer cache key.

        Every component is something that, if it changed, would make a
        previously cached answer wrong to serve:

        - scope: isolation boundary. Guarantees one scope can never be served
          another's cached answer, even for a byte-identical query.
        - index_version: a monotonic counter bumped on every (re)index. A
          counter rather than id(self.store.index) because a freed object's
          id() can be reused by a later object at the same address, which
          would risk a stale hit against a different corpus.
        - embedding model: different vectors mean different retrieval, which
          means a different grounded answer for the same words.
        - generation model + prompt version + max tokens: the answer text is a
          function of these, so a model swap or prompt edit must invalidate.
        - query fingerprint: a digest of the normalized query, so key size
          doesn't grow with query length.
        """
        return (
            scope if scope is not None else self.scope,
            self._index_version,
            getattr(self.store.embedder, "model_name", DEFAULT_EMBEDDING_MODEL_NAME),
            getattr(getattr(self.generator, "provider", None), "model", "unknown"),
            PROMPT_VERSION,
            DEFAULT_MAX_TOKENS,
            text_fingerprint(query_text),
        )

    def build_index(self, chunks: list[Chunk] | None = None) -> int:
        """
        Index `chunks` (defaulting to self.chunks) and return the new index
        version.

        The single place indexing happens, so the version counter cannot be
        bypassed. It previously could: startup called `harness.store.build()`
        directly, which skipped the counter entirely, so the answer cache key
        claimed index_version=0 while a fully built index sat behind it. That
        was harmless only because a fresh process starts with an empty cache —
        any runtime re-index would have kept serving answers from the previous
        corpus.

        Both the vector index and the lexical (BM25) index are rebuilt
        together; leaving one stale would make hybrid retrieval score a query
        against two different corpora.

        Args:
            chunks: chunks to index. Defaults to the harness's own `chunks`.

        Returns:
            int: the incremented index version now embedded in cache keys.
        """
        to_index = chunks if chunks is not None else self.chunks
        if not to_index:
            raise RuntimeError("VectorStore is empty and no chunks were provided to index")
        if chunks is not None:
            self.chunks = list(chunks)
        self.store.build(to_index)
        self.retriever.prepare_index()
        self._index_version += 1
        logger.info("indexed %d chunks (index_version=%d)", len(to_index), self._index_version)
        return self._index_version

    async def prewarm(self) -> dict[str, float]:
        """
        Pay every one-time initialization cost at process startup instead of
        inside the first user request, and report what each cost.

        Covers the three cold-start costs measured in this pipeline:
        the embedding model's lazy initialization (~690ms on the first
        encode, plus a fresh cost per novel input shape on MPS), the LLM
        provider's DNS/TCP/TLS handshake (~80ms), and the STT provider's
        (~100ms). Together that is roughly 0.9s that the first request after
        every deploy would otherwise absorb.

        Safe to call more than once; each step is idempotent.
        """
        timings: dict[str, float] = {}
        timings["embedder_warmup_ms"] = await asyncio.to_thread(self.store.embedder.warmup)
        # The grounding guardrail may hold a different Embedder instance (and
        # therefore a different model/device pair); warm that one too rather
        # than assume they share.
        guard_embedder = getattr(self.grounding_guardrail, "embedder", None)
        if guard_embedder is not None and guard_embedder is not self.store.embedder:
            timings["grounding_embedder_warmup_ms"] = await asyncio.to_thread(guard_embedder.warmup)
        try:
            # Provider SDKs can wait indefinitely during DNS or connection
            # setup. Prewarming is an optimization, so it must never prevent
            # the health endpoint and API from starting.
            prewarm_timeout = float(os.environ.get("LLM_PREWARM_TIMEOUT_SECONDS", "5"))
            timings["llm_prewarm_ms"] = await asyncio.wait_for(
                self.generator.prewarm(), timeout=prewarm_timeout
            )
        except Exception as exc:
            logger.warning("LLM prewarm skipped (%s)", exc)
        if self.stt_client is not None:
            timings["stt_prewarm_ms"] = await asyncio.to_thread(self.stt_client.prewarm)
        logger.info("prewarm: %s", " ".join(f"{k}={v:.1f}" for k, v in timings.items()))
        return timings

    def _run_stage(self, stage: str, trace: RequestTrace, errors: list[StageError], fn, *args):
        """Run a synchronous, non-retried stage; record its timing; capture failures as a StageError."""
        try:
            with trace.span(stage):
                return fn(*args)
        except Exception as exc:
            errors.append(StageError(stage=stage, error_type=type(exc).__name__, message=str(exc)))
            return None

    def _transcribe(
        self,
        audio_input: str | bytes,
        language_code: str = "unknown",
        filename: str = "audio.wav",
        content_type: str | None = None,
    ) -> str:
        if self.stt_client is None:
            self.stt_client = stt.SarvamSTT()
        if isinstance(audio_input, (bytes, bytearray)):
            audio_bytes = bytes(audio_input)
        else:
            filename = Path(audio_input).name
            content_type = None
            with open(audio_input, "rb") as f:
                audio_bytes = f.read()
        return self.stt_client.transcribe(
            audio_bytes, language_code, filename=filename, content_type=content_type
        ).transcript

    def _transcribe_with_retry(
        self, audio_input: str | bytes, filename: str = "audio.wav", content_type: str | None = None
    ) -> str:
        retrying = Retrying(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=self.retry_wait_multiplier, min=1, max=10),
            retry=retry_if_exception(_is_transient_stt_error),
            reraise=True,
        )
        return retrying(self._transcribe, audio_input, "unknown", filename, content_type)

    async def _transcribe_stage(
        self,
        audio_input: str | bytes,
        filename: str,
        content_type: str | None,
        trace: RequestTrace,
        errors: list[StageError],
    ) -> str | None:
        """
        Transcribe audio via the STT provider.

        Recorded as "stt_network" rather than "stt" because that is what it
        is: a round trip to api.sarvam.ai, measured at 250-525ms on a
        one-second clip and longer on real speech. This single span is the
        bulk of the ~958ms that the previous breakdown left unexplained — it
        was timed all along, and then simply omitted from the log line while
        still being counted into total_ms.
        """
        try:
            with trace.span("stt_network"):
                return await asyncio.to_thread(self._transcribe_with_retry, audio_input, filename, content_type)
        except Exception as exc:
            errors.append(StageError(stage="stt", error_type=type(exc).__name__, message=str(exc)))
            return None

    async def _retrieve_stage(
        self,
        query_text: str,
        trace: RequestTrace,
        errors: list[StageError],
    ) -> RetrievalResult | None:
        """
        Embed the query, search, and rerank — off the event loop.

        Runs in a worker thread via asyncio.to_thread. Embedding is a
        CPU-bound torch forward pass and FAISS search holds the GIL; calling
        them directly from an async handler blocks the whole event loop, so
        under any concurrency every other in-flight request stalls behind
        them. That did not show up in a one-request-at-a-time trace, which is
        exactly why it survived.

        Sub-timings arrive as flat, non-overlapping spans
        (embedding_cache / embedding_compute / vector_search / reranking), and
        whatever wall clock the stage spent outside them — thread dispatch,
        result marshalling — is recorded as retrieval_overhead rather than
        being silently dropped into the residual.
        """
        timing: dict[str, float] = {}
        stage_ms = 0.0
        try:
            if self.store.index is None or self.store.index.ntotal == 0:
                with trace.span("index_build"):
                    await asyncio.to_thread(self.build_index)

            started = time.perf_counter()
            try:
                return await asyncio.to_thread(self.retriever.retrieve, query_text, timing)
            finally:
                stage_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            errors.append(StageError(stage="retrieval", error_type=type(exc).__name__, message=str(exc)))
            return None
        finally:
            accounted = 0.0
            for name in _RETRIEVAL_SPANS:
                value = timing.get(f"{name}_ms")
                if value is not None:
                    trace.add(name, value)
                    accounted += value
            # Whatever the stage spent outside its own sub-timings: thread
            # dispatch, language detection, result marshalling. Recorded
            # explicitly so it shows up as retrieval overhead rather than
            # inflating the request-level unaccounted_ms, where it would be
            # indistinguishable from genuinely un-instrumented work.
            overhead = stage_ms - accounted
            if overhead > 0:
                trace.add("retrieval_overhead", overhead)
            passes = timing.get("search_passes")
            if passes:
                trace.label("retrieval_passes", int(passes))

    async def _chunk_vectors_for(self, retrieval_result: RetrievalResult) -> np.ndarray | None:
        """
        Embeddings for the retrieved chunks, read back out of the FAISS index.

        These are the vectors GroundingGuardrail needs. They were computed
        once when the index was built, so recovering them is a memory copy —
        versus 17-23ms of redundant model inference per request (and once per
        *sentence* on the streaming path) to re-encode text whose vector
        already exists. Returns None if positions aren't available, in which
        case the guardrail falls back to encoding as before.
        """
        if not retrieval_result.positions:
            return None
        try:
            return await asyncio.to_thread(self.store.embeddings_for, retrieval_result.positions)
        except Exception as exc:
            logger.warning("could not reconstruct chunk embeddings (%s); falling back to re-encoding", exc)
            return None

    async def _generate_stage(
        self,
        query_text: str,
        retrieval_result: RetrievalResult,
        trace: RequestTrace,
        errors: list[StageError],
    ) -> str | None:
        """
        Build the prompt and generate, recording context_build / llm_network /
        llm_client_wait / llm_generation as separate spans.

        Generation goes over the streaming transport even here on the
        non-streaming route, so llm_ttft_ms is always observable — see
        LLMProvider.answer_streamed for why that costs nothing measurable.
        """
        retrieved_chunks = list(zip(retrieval_result.chunks, retrieval_result.scores))
        with trace.span("context_build"):
            user_message = _build_user_message(query_text, retrieved_chunks)
            trace.label("prompt_chars", len(SYSTEM_PROMPT) + len(user_message))

        timing: dict[str, float] = {}
        try:
            return await self.generator.answer_streamed(query_text, retrieved_chunks, timing=timing)
        except Exception as exc:
            errors.append(StageError(stage="generation", error_type=type(exc).__name__, message=str(exc)))
            return None
        finally:
            self._record_llm_spans(trace, timing)

    @staticmethod
    def _record_llm_spans(trace: RequestTrace, timing: dict[str, Any]) -> None:
        """
        Move one LLM call's timings onto the trace: the consecutive slices as
        spans, the overlapping aggregates as details, and the non-timing facts
        (attempt count, provider finish reason) as labels.

        llm_finish_reason matters for diagnosis rather than latency: a
        `length` finish means the model was cut off at max_tokens, which for a
        reasoning model can mean its reasoning consumed the budget and no
        answer was emitted at all. That failure otherwise looks identical to a
        legitimate "not enough information" refusal by the time it reaches the
        guardrail.
        """
        for name in ("llm_network", "llm_client_wait", "llm_generation", "llm_retry_wait"):
            value = timing.get(f"{name}_ms")
            if value is not None:
                trace.add(name, value)
        for name in ("llm_ttft_ms", "llm_total_ms"):
            if name in timing:
                trace.detail(name[:-3], timing[name])
        if timing.get("llm_attempts", 1) > 1:
            trace.label("llm_attempts", int(timing["llm_attempts"]))
        if timing.get("llm_finish_reason"):
            trace.label("llm_finish_reason", str(timing["llm_finish_reason"]))

    def _degraded(
        self,
        answer: str,
        trace: RequestTrace,
        guard_flags: dict[str, GuardResult],
        errors: list[StageError],
        query_text: str = "",
        sources: list[Chunk] | None = None,
        scores: list[float] | None = None,
    ) -> PipelineResult:
        return PipelineResult(
            answer=answer,
            query_text=query_text,
            sources=sources or [],
            scores=scores or [],
            trace=trace,
            guard_flags=guard_flags,
            degraded=True,
            errors=errors,
        )

    async def run(
        self,
        audio_or_text_input: str | bytes,
        filename: str = "audio.wav",
        content_type: str | None = None,
        trace: RequestTrace | None = None,
        scope: str | None = None,
    ) -> PipelineResult:
        """
        Run one request through the full pipeline.

        Args:
            audio_or_text_input: either raw audio bytes, a path to an audio
                file (detected by extension), or an already-transcribed
                text query.
            filename: filename (with a real audio extension, e.g.
                "recording.webm") describing raw audio bytes' format. Only
                used when audio_or_text_input is bytes — Sarvam needs this
                to know the codec, since bytes alone carry no format info.
                Ignored for str inputs (a path's own extension is used) and
                for text queries.
            content_type: the audio's real MIME type (e.g. "audio/webm"),
                when audio_or_text_input is bytes. Sarvam rejects requests
                with no resolvable Content-Type, so prefer passing this
                (e.g. the browser upload's real Content-Type) over relying
                on a guess from `filename`'s extension alone.
            trace: an in-progress RequestTrace to record into — passed down by
                the ASGI middleware so the trace's total_ms covers the whole
                HTTP request (arrival, body parse, serialization, flush), not
                just the part inside this method. One is created here if
                omitted, e.g. when called from a benchmark.
            scope: overrides the harness-level cache isolation scope for this
                request.

        Returns:
            PipelineResult: the final answer, its source chunks, a full
            lifecycle trace, per-guardrail results, and whether the response
            is a degraded fallback (a stage technically failed) rather than a
            normal answer or guardrail refusal.
        """
        owns_trace = trace is None
        if trace is None:
            trace = RequestTrace()
            trace.start()
        errors: list[StageError] = []
        guard_flags: dict[str, GuardResult] = {}

        def finish(result: PipelineResult) -> PipelineResult:
            if owns_trace:
                trace.finish()
            logger.info("%s", trace.render())
            return result

        # Stage 1: STT (only if the input looks like audio)
        is_audio = isinstance(audio_or_text_input, (bytes, bytearray)) or (
            isinstance(audio_or_text_input, str) and _is_audio_path(audio_or_text_input)
        )
        trace.label("input", "audio" if is_audio else "text")
        if is_audio:
            query_text = await self._transcribe_stage(audio_or_text_input, filename, content_type, trace, errors)
            if query_text is None:
                return finish(
                    self._degraded(
                        "I couldn't understand the audio after a couple of tries. Could you type your question instead?",
                        trace,
                        guard_flags,
                        errors,
                    )
                )
        else:
            query_text = normalize_query_input(audio_or_text_input)

        query_text = normalize_query_input(query_text)

        # Fast path: the answer cache, consulted before embedding, retrieval,
        # or generation. For a text query this is the whole request; for audio
        # it can only ever start after STT, since the query isn't known until
        # the audio has been transcribed.
        with trace.span("cache_lookup"):
            cache_key = self._result_cache_key(query_text, scope)
            cached = self._result_cache.get(cache_key)
            if cached is not None:
                self._result_cache.move_to_end(cache_key)  # LRU touch
        if cached is not None:
            trace.label("cache", "hit")
            return finish(cached.model_copy(update={"trace": trace, "cached": True}))
        trace.label("cache", "miss")

        # Stage 2: InputGuardrail
        input_result = self._run_stage("query_preprocessing", trace, errors, self.input_guardrail.check, query_text)
        if input_result is None:
            return finish(self._degraded(REFUSAL_RESPONSE, trace, guard_flags, errors, query_text=query_text))
        guard_flags["input"] = input_result
        if not input_result.allowed:
            result = PipelineResult(
                answer=input_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=[],
                trace=trace,
                guard_flags=guard_flags,
                degraded=False,
                errors=errors,
            )
            self._cache_result(cache_key, result)
            return finish(result)

        # Stage 3: chunking/retrieval (chunking is skipped once the store is indexed)
        retrieval_result = await self._retrieve_stage(query_text, trace, errors)
        if retrieval_result is None:
            return finish(self._degraded(REFUSAL_RESPONSE, trace, guard_flags, errors, query_text=query_text))

        # Stage 4: RelevanceGuardrail
        relevance_result = self._run_stage(
            "relevance_guard", trace, errors, self.relevance_guardrail.check, retrieval_result
        )
        if relevance_result is None:
            return finish(
                self._degraded(
                    REFUSAL_RESPONSE, trace, guard_flags, errors,
                    query_text=query_text, sources=retrieval_result.chunks, scores=retrieval_result.scores,
                )
            )
        guard_flags["relevance"] = relevance_result
        if not relevance_result.allowed:
            result = PipelineResult(
                answer=relevance_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=retrieval_result.chunks,
                scores=retrieval_result.scores,
                trace=trace,
                guard_flags=guard_flags,
                degraded=False,
                errors=errors,
            )
            self._cache_result(cache_key, result)
            return finish(result)

        # Stage 5: generation, concurrent with grounding-vector preparation.
        #
        # The vectors the grounding guardrail will need depend only on which
        # chunks were retrieved — not on the answer — so there is no reason to
        # wait for the LLM before preparing them. asyncio.gather overlaps the
        # two. (With FAISS reconstruction the prep is now near-instant, so the
        # overlap saves little in absolute terms; it is kept because it makes
        # the dependency structure explicit, and because the fallback path
        # inside _chunk_vectors_for — re-encoding chunk texts when positions
        # aren't available — is the 17-23ms case this actually hides.)
        answer_text, chunk_vectors = await asyncio.gather(
            self._generate_stage(query_text, retrieval_result, trace, errors),
            self._chunk_vectors_for(retrieval_result),
        )
        if answer_text is None:
            return finish(
                self._degraded(
                    "I'm having trouble generating an answer right now. Please try again shortly.",
                    trace,
                    guard_flags,
                    errors,
                    query_text=query_text,
                    sources=retrieval_result.chunks,
                    scores=retrieval_result.scores,
                )
            )

        # Stage 6: GroundingGuardrail
        grounding_result = self._run_stage(
            "grounding_guard",
            trace,
            errors,
            self.grounding_guardrail.check,
            answer_text,
            retrieval_result.chunks,
            chunk_vectors,
        )
        if grounding_result is None:
            return finish(
                self._degraded(
                    REFUSAL_RESPONSE, trace, guard_flags, errors,
                    query_text=query_text, sources=retrieval_result.chunks, scores=retrieval_result.scores,
                )
            )
        guard_flags["grounding"] = grounding_result
        if not grounding_result.allowed:
            result = PipelineResult(
                answer=grounding_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=retrieval_result.chunks,
                scores=retrieval_result.scores,
                trace=trace,
                guard_flags=guard_flags,
                degraded=False,
                errors=errors,
            )
            self._cache_result(cache_key, result)
            return finish(result)

        # Final response.
        #
        # When GroundingGuardrail allows the answer but sets a
        # response_override, it has dropped individual ungrounded sentences and
        # the override is the surviving text — so the override must win here,
        # not just on the refusal branch above. Returning answer_text would
        # hand back the sentences the guardrail just rejected.
        result = PipelineResult(
            answer=grounding_result.response_override or answer_text,
            query_text=query_text,
            sources=retrieval_result.chunks,
            scores=retrieval_result.scores,
            trace=trace,
            guard_flags=guard_flags,
            degraded=False,
            errors=errors,
        )
        self._cache_result(cache_key, result)
        return finish(result)

    def _cache_result(self, cache_key: tuple, result: PipelineResult) -> None:
        """
        Store a result under `cache_key`, evicting least-recently-used entries
        first. Degraded results are never cached: they represent transient
        technical failures, which should be retried rather than replayed.
        """
        if result.degraded:
            return
        while len(self._result_cache) >= _RESULT_CACHE_MAX_SIZE:
            self._result_cache.popitem(last=False)
        self._result_cache[cache_key] = result

    async def run_streaming(
        self,
        audio_or_text_input: str | bytes,
        filename: str = "audio.wav",
        content_type: str | None = None,
        trace: RequestTrace | None = None,
        scope: str | None = None,
    ) -> AsyncIterator[StreamSentenceEvent | StreamDoneEvent]:
        """
        Streaming counterpart to run(). STT, InputGuardrail,
        chunking/retrieval, and RelevanceGuardrail run exactly as in run()
        — none of them benefit from streaming. For generation, text deltas
        are accumulated into sentences as the LLM streams; each complete
        sentence is checked individually via
        GroundingGuardrail.is_sentence_grounded() and only yielded if it
        passes, so an ungrounded sentence is silently dropped rather than
        ever reaching the caller. This is a *stricter* guarantee than
        run()'s 30%-tolerant whole-answer check — every individual sentence
        must pass here, not just enough of them in aggregate.

        Yields a StreamSentenceEvent for each accepted sentence as it's
        ready, followed by exactly one StreamDoneEvent carrying the final
        PipelineResult (built from only the accepted sentences).
        """
        owns_trace = trace is None
        if trace is None:
            trace = RequestTrace()
            trace.start()
        errors: list[StageError] = []
        guard_flags: dict[str, GuardResult] = {}

        def log(result: PipelineResult) -> None:
            if owns_trace:
                trace.finish()
            logger.info("%s", trace.render())

        # Stage 1: STT
        is_audio = isinstance(audio_or_text_input, (bytes, bytearray)) or (
            isinstance(audio_or_text_input, str) and _is_audio_path(audio_or_text_input)
        )
        trace.label("input", "audio" if is_audio else "text")
        if is_audio:
            query_text = await self._transcribe_stage(audio_or_text_input, filename, content_type, trace, errors)
            if query_text is None:
                result = self._degraded(
                    "I couldn't understand the audio after a couple of tries. Could you type your question instead?",
                    trace,
                    guard_flags,
                    errors,
                )
                log(result)
                yield StreamDoneEvent(result=result)
                return
        else:
            query_text = audio_or_text_input

        with trace.span("cache_lookup"):
            cache_key = self._result_cache_key(query_text, scope)
            cached = self._result_cache.get(cache_key)
            if cached is not None:
                self._result_cache.move_to_end(cache_key)
        if cached is not None:
            trace.label("cache", "hit")
            result = cached.model_copy(update={"trace": trace, "cached": True})
            log(result)
            if result.answer:
                yield StreamSentenceEvent(text=result.answer)
            yield StreamDoneEvent(result=result)
            return
        trace.label("cache", "miss")

        # Stage 2: InputGuardrail
        input_result = self._run_stage("query_preprocessing", trace, errors, self.input_guardrail.check, query_text)
        if input_result is None:
            result = self._degraded(REFUSAL_RESPONSE, trace, guard_flags, errors, query_text=query_text)
            log(result)
            yield StreamDoneEvent(result=result)
            return
        guard_flags["input"] = input_result
        if not input_result.allowed:
            result = PipelineResult(
                answer=input_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=[],
                trace=trace,
                guard_flags=guard_flags,
                degraded=False,
                errors=errors,
            )
            self._cache_result(cache_key, result)
            log(result)
            yield StreamDoneEvent(result=result)
            return

        # Stage 3: chunking/retrieval
        retrieval_result = await self._retrieve_stage(query_text, trace, errors)
        if retrieval_result is None:
            result = self._degraded(REFUSAL_RESPONSE, trace, guard_flags, errors, query_text=query_text)
            log(result)
            yield StreamDoneEvent(result=result)
            return

        # Stage 4: RelevanceGuardrail
        relevance_result = self._run_stage(
            "relevance_guard", trace, errors, self.relevance_guardrail.check, retrieval_result
        )
        if relevance_result is None:
            result = self._degraded(
                REFUSAL_RESPONSE,
                trace,
                guard_flags,
                errors,
                query_text=query_text,
                sources=retrieval_result.chunks,
                scores=retrieval_result.scores,
            )
            log(result)
            yield StreamDoneEvent(result=result)
            return
        guard_flags["relevance"] = relevance_result
        if not relevance_result.allowed:
            result = PipelineResult(
                answer=relevance_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=retrieval_result.chunks,
                scores=retrieval_result.scores,
                trace=trace,
                guard_flags=guard_flags,
                degraded=False,
                errors=errors,
            )
            self._cache_result(cache_key, result)
            log(result)
            yield StreamDoneEvent(result=result)
            return

        # Stage 5: streaming generation, with per-sentence grounding as each one completes
        retrieved_chunks = list(zip(retrieval_result.chunks, retrieval_result.scores))
        with trace.span("context_build"):
            user_message = _build_user_message(query_text, retrieved_chunks)
            trace.label("prompt_chars", len(SYSTEM_PROMPT) + len(user_message))

        # Prepared once, before the stream opens, and reused for every
        # sentence check below. Previously each is_sentence_grounded() call
        # re-encoded all five chunk texts, so a five-sentence answer paid that
        # ~17-23ms five times — and paid it *inside* the stream loop, delaying
        # each sentence's delivery to the client.
        chunk_vectors = await self._chunk_vectors_for(retrieval_result)

        accepted_sentences: list[str] = []
        dropped_count = 0
        buffer = ""
        llm_timing: dict[str, float] = {}
        grounding_ms = 0.0

        async def grounded(sentence: str) -> bool:
            """Grounding-check one sentence off the event loop, accumulating its cost."""
            nonlocal grounding_ms
            started = time.perf_counter()
            try:
                return await asyncio.to_thread(
                    self.grounding_guardrail.is_sentence_grounded,
                    sentence,
                    retrieval_result.chunks,
                    chunk_vectors,
                )
            finally:
                grounding_ms += (time.perf_counter() - started) * 1000

        try:
            async for delta in self.generator.stream_answer(query_text, retrieved_chunks, timing=llm_timing):
                buffer += delta
                while True:
                    match = _SENTENCE_BOUNDARY_RE.search(buffer)
                    if not match:
                        break
                    sentence, buffer = buffer[: match.end()].strip(), buffer[match.end() :]
                    if not sentence:
                        continue
                    if await grounded(sentence):
                        accepted_sentences.append(sentence)
                        yield StreamSentenceEvent(text=sentence)
                    else:
                        dropped_count += 1
            trailing = buffer.strip()
            if trailing:
                if await grounded(trailing):
                    accepted_sentences.append(trailing)
                    yield StreamSentenceEvent(text=trailing)
                else:
                    dropped_count += 1
        except Exception as exc:
            errors.append(StageError(stage="generation", error_type=type(exc).__name__, message=str(exc)))
        finally:
            self._record_llm_spans(trace, llm_timing)
            if grounding_ms:
                # Per-sentence grounding is interleaved with generation here,
                # not sequential after it: the loop above blocks the stream
                # read while it checks each completed sentence. So the
                # grounding cost is *inside* the llm_generation wall clock, and
                # recording both at face value would overlap them and push the
                # request residual negative by exactly this much. Splitting it
                # out of llm_generation keeps the spans flat and the residual
                # meaningful.
                trace.add("grounding_guard", grounding_ms)
                for span in trace.spans:
                    if span.name == "llm_generation":
                        span.duration_ms = max(0.0, span.duration_ms - grounding_ms)

        if not accepted_sentences:
            fallback = (
                "I'm having trouble generating an answer right now. Please try again shortly."
                if errors and errors[-1].stage == "generation"
                else REFUSAL_RESPONSE
            )
            result = self._degraded(
                fallback,
                trace,
                guard_flags,
                errors,
                query_text=query_text,
                sources=retrieval_result.chunks,
                scores=retrieval_result.scores,
            )
            log(result)
            yield StreamDoneEvent(result=result)
            return

        guard_flags["grounding"] = GuardResult(
            allowed=True,
            reason="ok" if dropped_count == 0 else f"dropped_{dropped_count}_ungrounded_sentence(s)",
        )
        final_result = PipelineResult(
            answer=" ".join(accepted_sentences),
            query_text=query_text,
            sources=retrieval_result.chunks,
            scores=retrieval_result.scores,
            trace=trace,
            guard_flags=guard_flags,
            degraded=False,
            errors=errors,
        )
        self._cache_result(cache_key, final_result)
        log(final_result)
        yield StreamDoneEvent(result=final_result)

    # ------------------------------------------------------------------
    # Overlapped STT + retrieval path
    # ------------------------------------------------------------------

    async def run_overlapped(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        content_type: str | None = None,
        trace: RequestTrace | None = None,
        scope: str | None = None,
        realtime_stt_client=None,
    ) -> PipelineResult:
        """
        Lowest-achievable-latency audio path: start retrieval on the first
        *stable* partial transcript while STT is still running, so retrieval
        latency is hidden inside the STT round-trip instead of sitting on
        top of it.

        Critical path with this architecture::

            ┌─── STT streaming ───────────────────────────┐
            │  partial₁  partial₂  stable_partial  final  │
            └─────────────────────────────────────────────┘
                                   │
                                   └── retrieval starts here (concurrent)
                                                │
                                                └── retrieval done
                                                            │
                              final transcript arrives ──── ┘
                                                            │
                                              guardrail + fast answer
                                                            │
                                                         DONE

        The wall clock stops when the response is returned, and total_ms is
        measured end-to-end from ASGI arrival (via the existing
        LatencyTraceMiddleware) — STT time is never subtracted, hidden, or
        excluded.  The ``stt_overlap_savings`` span records how many ms of
        retrieval ran concurrently with STT, which is the genuine reduction
        in total_ms this architecture achieves.

        If retrieval finishes *after* the final transcript (the partial was
        unstable or retrieval was unusually slow), the path degrades
        gracefully: the pipeline waits for retrieval to complete and
        proceeds normally — no answer is skipped, no result is fabricated.

        Uses the fast grounded (ExtractiveProvider / local) answer path by
        default.  The remote LLM path (700-1300ms) is categorically
        incompatible with any sub-second target; keeping it optional via the
        existing generator avoids hard-coding the choice here.

        Args:
            audio_bytes: raw audio file content.
            filename: audio filename hint passed to Sarvam.
            content_type: MIME type hint passed to Sarvam.
            trace: in-progress RequestTrace (from ASGI middleware).
            scope: answer-cache isolation scope override.
            realtime_stt_client: a SarvamRealtimeSTT or MockRealtimeSTT
                instance.  If None, one is constructed lazily from the env.

        Returns:
            PipelineResult with full latency breakdown including
            ``stt_to_first_partial``, ``retrieval_on_partial``,
            ``stt_final``, and ``stt_overlap_savings`` spans.
        """
        from src.stt import SarvamRealtimeSTT, PartialTranscriptEvent, FinalTranscriptEvent

        owns_trace = trace is None
        if trace is None:
            trace = RequestTrace()
            trace.start()
        errors: list[StageError] = []
        guard_flags: dict[str, GuardResult] = {}

        def finish(result: PipelineResult) -> PipelineResult:
            if owns_trace:
                trace.finish()
            logger.info("%s", trace.render())
            return result

        trace.label("input", "audio_overlapped")

        # Lazy-build a realtime STT client once per harness.
        if realtime_stt_client is None:
            if not hasattr(self, "_realtime_stt_client") or self._realtime_stt_client is None:
                try:
                    self._realtime_stt_client = SarvamRealtimeSTT()
                except Exception as exc:
                    errors.append(StageError(stage="stt", error_type=type(exc).__name__, message=str(exc)))
                    return finish(self._degraded(
                        "Speech-to-text is not configured (SARVAM_API_KEY missing).",
                        trace, guard_flags, errors,
                    ))
            realtime_stt_client = self._realtime_stt_client

        # ── Phase 1: STT streaming + concurrent retrieval ──────────────

        stt_stream_started = time.perf_counter()
        first_partial_at: float | None = None
        stable_partial_at: float | None = None
        final_transcript_at: float | None = None
        stable_partial_text: str = ""
        final_text: str = ""
        partial_changes: int = 0

        # Retrieval future: created as soon as a stable partial arrives.
        retrieval_task: asyncio.Task | None = None
        retrieval_result: RetrievalResult | None = None
        retrieval_query: str = ""  # the query retrieval ran on

        async def _kick_retrieval(query: str) -> RetrievalResult | None:
            """Fire retrieval off-event-loop and return the result."""
            return await self._retrieve_stage(query, trace, errors)

        try:
            async for event in realtime_stt_client.stream(
                audio_bytes, filename=filename, content_type=content_type
            ):
                if isinstance(event, PartialTranscriptEvent):
                    partial_changes += 1
                    if first_partial_at is None:
                        first_partial_at = event.received_at

                    if event.is_stable and retrieval_task is None:
                        # First stable partial — start retrieval immediately,
                        # don't wait for STT to finish.
                        stable_partial_at = event.received_at
                        stable_partial_text = normalize_query_input(event.text)
                        retrieval_query = stable_partial_text
                        # Ensure index is ready before spawning retrieval.
                        if self.store.index is None or self.store.index.ntotal == 0:
                            with trace.span("index_build"):
                                await asyncio.to_thread(self.build_index)
                        retrieval_task = asyncio.create_task(_kick_retrieval(retrieval_query))

                elif isinstance(event, FinalTranscriptEvent):
                    final_text = event.text
                    final_transcript_at = event.received_at
                    break

        except Exception as exc:
            errors.append(StageError(stage="stt", error_type=type(exc).__name__, message=str(exc)))
            return finish(self._degraded(
                "I couldn't understand the audio after a couple of tries. Could you type your question instead?",
                trace, guard_flags, errors,
            ))

        if not final_text:
            errors.append(StageError(stage="stt", error_type="EmptyTranscript", message="STT returned no text"))
            return finish(self._degraded(
                "I couldn't understand the audio. Could you type your question instead?",
                trace, guard_flags, errors,
            ))

        # ── Record STT spans ────────────────────────────────────────────
        stt_stream_end = time.perf_counter()

        if first_partial_at is not None:
            trace.add("stt_to_first_partial", (first_partial_at - stt_stream_started) * 1000)

        # stt_final = total STT wall clock (stream open → final transcript).
        trace.add("stt_final", (stt_stream_end - stt_stream_started) * 1000)

        # ── Normalize and cache-check the final query ──────────────────
        query_text = normalize_query_input(final_text)
        query_text = normalize_query_input(query_text)

        with trace.span("cache_lookup"):
            cache_key = self._result_cache_key(query_text, scope)
            cached = self._result_cache.get(cache_key)
            if cached is not None:
                self._result_cache.move_to_end(cache_key)
        if cached is not None:
            trace.label("cache", "hit")
            if retrieval_task is not None:
                retrieval_task.cancel()
            return finish(cached.model_copy(update={"trace": trace, "cached": True}))
        trace.label("cache", "miss")

        # ── Phase 2: If the final query differs materially from the
        #    partial we retrieved on, discard and re-retrieve. ──────────
        queries_match = (
            retrieval_query
            and _queries_close_enough(query_text, retrieval_query)
        )

        if retrieval_task is not None and not queries_match:
            # Final differed from the stable partial — cancel speculative
            # retrieval and fall back to serial retrieval on the correct query.
            retrieval_task.cancel()
            try:
                await retrieval_task
            except (asyncio.CancelledError, Exception):
                pass
            retrieval_task = None
            trace.label("retrieval_rerun", True)

        retrieval_wait_started = time.perf_counter()

        if retrieval_task is not None:
            # Await the already-running retrieval task.  If it finished during
            # STT, this returns immediately (zero additional wait).
            try:
                retrieval_result = await retrieval_task
            except Exception as exc:
                errors.append(StageError(stage="retrieval", error_type=type(exc).__name__, message=str(exc)))
                retrieval_result = None
        else:
            # No speculative retrieval ran (no stable partial, or re-run needed).
            retrieval_result = await self._retrieve_stage(query_text, trace, errors)

        retrieval_wait_end = time.perf_counter()

        # ── Compute overlap savings ─────────────────────────────────────
        if retrieval_task is not None and stable_partial_at is not None:
            # How much of retrieval ran concurrently with STT?
            # = final_transcript_time - stable_partial_time
            #   (the window during which retrieval was running while STT
            #    was still receiving audio)
            # Capped at actual retrieval duration — can't save more than
            # retrieval cost.
            retrieval_span = trace.get("retrieval_overhead")  # rough proxy
            concurrent_window_ms = (
                (final_transcript_at - stable_partial_at) * 1000
                if final_transcript_at is not None
                else 0.0
            )
            # More precisely: how long retrieval actually ran during STT.
            # retrieval_wait is the part that waited *after* STT finished.
            actual_wait_ms = (retrieval_wait_end - retrieval_wait_started) * 1000
            retrieval_duration_ms = sum(
                trace.get(s) or 0.0
                for s in ("embedding_cache", "embedding_compute", "vector_search",
                          "bm25", "fusion", "reranking", "retrieval_overhead")
            )
            savings_ms = max(0.0, retrieval_duration_ms - actual_wait_ms)
            if savings_ms > 0:
                trace.add("stt_overlap_savings", savings_ms)
            trace.label("overlap_concurrent_window_ms", int(concurrent_window_ms))

        # ── Retrieval-on-partial span ──────────────────────────────────
        # This is an *overlapping* wall-clock interval (it spans the same
        # clock ticks as the flat retrieval sub-spans), so it must be
        # recorded as a detail rather than a flat span.  Recording it as a
        # span would double-count those ms and push unaccounted_ms negative.
        if stable_partial_at is not None:
            trace.detail(
                "retrieval_on_partial",
                (retrieval_wait_end - stable_partial_at) * 1000,
            )

        trace.label("partial_changes", partial_changes)
        trace.label("queries_matched", queries_match)

        if retrieval_result is None:
            return finish(self._degraded(REFUSAL_RESPONSE, trace, guard_flags, errors, query_text=query_text))

        # ── Stage 2: InputGuardrail ────────────────────────────────────
        input_result = self._run_stage(
            "query_preprocessing", trace, errors, self.input_guardrail.check, query_text
        )
        if input_result is None:
            return finish(self._degraded(REFUSAL_RESPONSE, trace, guard_flags, errors, query_text=query_text))
        guard_flags["input"] = input_result
        if not input_result.allowed:
            result = PipelineResult(
                answer=input_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text, sources=[], trace=trace,
                guard_flags=guard_flags, degraded=False, errors=errors,
            )
            self._cache_result(cache_key, result)
            return finish(result)

        # ── Stage 4: RelevanceGuardrail ────────────────────────────────
        relevance_result = self._run_stage(
            "relevance_guard", trace, errors, self.relevance_guardrail.check, retrieval_result
        )
        if relevance_result is None:
            return finish(self._degraded(
                REFUSAL_RESPONSE, trace, guard_flags, errors,
                query_text=query_text,
                sources=retrieval_result.chunks, scores=retrieval_result.scores,
            ))
        guard_flags["relevance"] = relevance_result
        if not relevance_result.allowed:
            result = PipelineResult(
                answer=relevance_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=retrieval_result.chunks, scores=retrieval_result.scores,
                trace=trace, guard_flags=guard_flags, degraded=False, errors=errors,
            )
            self._cache_result(cache_key, result)
            return finish(result)

        # ── Stage 5: Generation + grounding-vector prep (concurrent) ───
        answer_text, chunk_vectors = await asyncio.gather(
            self._generate_stage(query_text, retrieval_result, trace, errors),
            self._chunk_vectors_for(retrieval_result),
        )
        if answer_text is None:
            return finish(self._degraded(
                "I'm having trouble generating an answer right now. Please try again shortly.",
                trace, guard_flags, errors,
                query_text=query_text,
                sources=retrieval_result.chunks, scores=retrieval_result.scores,
            ))

        # ── Stage 6: GroundingGuardrail ────────────────────────────────
        grounding_result = self._run_stage(
            "grounding_guard", trace, errors,
            self.grounding_guardrail.check, answer_text,
            retrieval_result.chunks, chunk_vectors,
        )
        if grounding_result is None:
            return finish(self._degraded(
                REFUSAL_RESPONSE, trace, guard_flags, errors,
                query_text=query_text,
                sources=retrieval_result.chunks, scores=retrieval_result.scores,
            ))
        guard_flags["grounding"] = grounding_result
        if not grounding_result.allowed:
            result = PipelineResult(
                answer=grounding_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=retrieval_result.chunks, scores=retrieval_result.scores,
                trace=trace, guard_flags=guard_flags, degraded=False, errors=errors,
            )
            self._cache_result(cache_key, result)
            return finish(result)

        result = PipelineResult(
            answer=grounding_result.response_override or answer_text,
            query_text=query_text,
            sources=retrieval_result.chunks, scores=retrieval_result.scores,
            trace=trace, guard_flags=guard_flags, degraded=False, errors=errors,
        )
        self._cache_result(cache_key, result)
        return finish(result)


def _queries_close_enough(a: str, b: str, threshold: float = 0.7) -> bool:
    """
    Heuristic: are two query strings close enough that retrieval on `b`
    is valid for answering `a`?

    Uses token overlap (Jaccard similarity on word sets).  A score >= threshold
    means we trust the speculative retrieval result; below it we re-retrieve.
    Simple and fast — no model needed, runs in microseconds.

    threshold=0.7 means at least 70% of the token union must be shared.
    """
    import re
    def tokens(s: str) -> set:
        return set(re.findall(r"\w+", s.casefold()))
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return True
    intersection = len(ta & tb)
    union = len(ta | tb)
    return (intersection / union) >= threshold if union else True


DEFAULT_K_VALUES = (1, 3, 5, 10)


def load_benchmark(benchmark_path: str) -> list[dict[str, Any]]:
    """
    Load a benchmark dataset of labeled retrieval cases.

    Each case is a dict as produced by QueryDoc.to_eval_cases (see
    src/data_loader.py): {"query_id", "query", "language", "query_type",
    "expected_answer", "relevant_passage_ids"}, where relevant_passage_ids
    comes from MSMARCO-XI's own is_selected labels.

    Args:
        benchmark_path: filesystem path to the benchmark JSON file.

    Returns:
        list[dict]: benchmark case records.
    """
    with open(benchmark_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ranked_passages(chunks: list[Chunk]) -> list[tuple[Any, Any]]:
    """
    Reduce a ranked chunk list to a ranked list of distinct passage keys.

    Ranking is scored over passages, not chunks. The corpus indexes each
    passage once per language (e.g. Hindi and English), so the same
    (query_id, passage_id) legitimately appears twice in one result; counting
    both would inflate recall and let a single passage retrieved in two
    languages masquerade as two distinct hits. The highest-ranked occurrence
    of each passage is kept.

    Args:
        chunks: retrieved chunks, best first.

    Returns:
        list[tuple]: (query_id, passage_id) keys, best first, deduped.
    """
    seen: set[tuple[Any, Any]] = set()
    ranked = []
    for chunk in chunks:
        key = (chunk.metadata.get("query_id"), chunk.metadata.get("passage_id"))
        if key in seen:
            continue
        seen.add(key)
        ranked.append(key)
    return ranked


def run_benchmark(
    cases: list[dict[str, Any]],
    retriever: Retriever,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> dict[str, Any]:
    """
    Run each benchmark case through retrieval and compute aggregate metrics.

    Deliberately scoped to retrieval, not the full pipeline: hit rate,
    recall, precision, and MRR are all computable from is_selected labels
    alone, so this runs with no API keys, no STT, and no LLM spend, and its
    numbers are deterministic. End-to-end answer quality and latency are
    measured separately by benchmarks/run_benchmark.py.

    Note this takes an explicit `retriever` rather than building one: metrics
    are only meaningful against the same corpus the cases were generated
    from, so the caller owns constructing and indexing that store. The
    retriever's top_n must be at least max(k_values) for the larger k's to
    mean anything.

    Args:
        cases: benchmark case records as returned by load_benchmark.
        retriever: a Retriever over an already-indexed VectorStore.
        k_values: cutoffs to report metrics at.

    Returns:
        dict: aggregate metrics overall and per language, plus per-case rows.
    """
    max_k = max(k_values)
    if retriever.top_n < max_k:
        logger.warning(
            "retriever.top_n=%d is below max(k_values)=%d; metrics at k>%d are truncated "
            "and will understate retrieval quality",
            retriever.top_n,
            max_k,
            retriever.top_n,
        )

    rows = []
    for case in cases:
        result = retriever.retrieve(case["query"])
        ranked = _ranked_passages(result.chunks)
        relevant = {(case["query_id"], pid) for pid in case["relevant_passage_ids"]}

        first_rank = next(
            (rank for rank, key in enumerate(ranked, start=1) if key in relevant), None
        )
        per_k = {}
        for k in k_values:
            found = sum(1 for key in ranked[:k] if key in relevant)
            per_k[k] = {
                "hit": found > 0,
                "recall": found / len(relevant),
                "precision": found / k,
            }

        rows.append(
            {
                "query_id": case["query_id"],
                "language": case["language"],
                "query": case["query"],
                "first_relevant_rank": first_rank,
                "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
                "low_confidence": result.low_confidence,
                # True when retrieval returned fewer distinct passages than the
                # largest cutoff, so metrics at that k are bounded by how much
                # the retriever returned rather than by its ranking quality.
                # Tracked because top_n counts chunks while ranking counts
                # passages — with each passage indexed per language, a top_n
                # comfortably above max_k can still yield too few passages.
                "truncated": len(ranked) < max_k,
                "per_k": per_k,
            }
        )

    def aggregate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {}
        n = len(subset)
        return {
            "n": n,
            "mrr": sum(r["reciprocal_rank"] for r in subset) / n,
            "low_confidence_rate": sum(1 for r in subset if r["low_confidence"]) / n,
            "truncated_rate": sum(1 for r in subset if r["truncated"]) / n,
            **{
                f"hit_rate@{k}": sum(1 for r in subset if r["per_k"][k]["hit"]) / n
                for k in k_values
            },
            **{f"recall@{k}": sum(r["per_k"][k]["recall"] for r in subset) / n for k in k_values},
            **{
                f"precision@{k}": sum(r["per_k"][k]["precision"] for r in subset) / n
                for k in k_values
            },
        }

    languages = sorted({row["language"] for row in rows})
    return {
        "k_values": list(k_values),
        "overall": aggregate(rows),
        "per_language": {
            lang: aggregate([r for r in rows if r["language"] == lang]) for lang in languages
        },
        "cases": rows,
    }


def report_results(results: dict[str, Any], report_path: str | Path | None = None) -> str:
    """
    Render a human-readable Markdown summary of benchmark results.

    Args:
        results: results dict as returned by run_benchmark.
        report_path: if given, the report is also written to this path.

    Returns:
        str: the rendered Markdown report.
    """
    k_values = results["k_values"]
    overall = results["overall"]

    lines = [
        "# Retrieval evaluation — ai4bharat/MSMARCO-XI",
        "",
        "Retrieval-only metrics scored against the dataset's own `is_selected` relevance",
        "labels. A passage counts as retrieved once, regardless of how many languages it",
        "was indexed in. No LLM is involved, so these numbers are deterministic.",
        "",
        f"Cases: **{overall.get('n', 0)}** | MRR: **{overall.get('mrr', 0):.3f}** | "
        f"low-confidence rate: **{overall.get('low_confidence_rate', 0):.1%}**",
        "",
        f"Coverage: {overall.get('truncated_rate', 0):.1%} of cases returned fewer than "
        f"{max(k_values)} distinct passages, so their metrics at the largest k are bounded by "
        "retrieval depth rather than ranking quality.",
        "",
        "## Overall",
        "",
        "| k | hit rate@k | recall@k | precision@k |",
        "| --- | --- | --- | --- |",
    ]
    for k in k_values:
        lines.append(
            f"| {k} | {overall.get(f'hit_rate@{k}', 0):.1%} | "
            f"{overall.get(f'recall@{k}', 0):.1%} | {overall.get(f'precision@{k}', 0):.1%} |"
        )

    leaked = results.get("leaked_comparison")
    if leaked:
        lines += [
            "",
            "## Label-leak control",
            "",
            "The same eval re-run with `Retriever(is_selected_boost=0.1)` — the production",
            "default. `is_selected` is the relevance label being scored, so boosting by it",
            "hands the ranker the answer. The gap below is the size of that leak, and is the",
            "reason the headline numbers above disable the boost.",
            "",
            "| metric | honest | with is_selected boost |",
            "| --- | --- | --- |",
            f"| MRR | {overall.get('mrr', 0):.3f} | {leaked.get('mrr', 0):.3f} |",
        ]
        for k in k_values:
            key = f"hit_rate@{k}"
            lines.append(
                f"| hit rate@{k} | {overall.get(key, 0):.1%} | {leaked.get(key, 0):.1%} |"
            )

    per_language = results.get("per_language") or {}
    if len(per_language) > 1:
        lines += [
            "",
            "## By query language",
            "",
            "| language | n | MRR | " + " | ".join(f"hit@{k}" for k in k_values) + " |",
            "| --- | --- | --- | " + " | ".join("---" for _ in k_values) + " |",
        ]
        for lang, metrics in per_language.items():
            hits = " | ".join(f"{metrics.get(f'hit_rate@{k}', 0):.1%}" for k in k_values)
            lines.append(
                f"| {lang} | {metrics.get('n', 0)} | {metrics.get('mrr', 0):.3f} | {hits} |"
            )

    report = "\n".join(lines) + "\n"
    if report_path is not None:
        Path(report_path).write_text(report, encoding="utf-8")
    print(report)
    return report
