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
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Literal

import requests
from pydantic import BaseModel, Field
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from src import stt
from src.chunking import Chunk
from src.generation import Generator
from src.guardrails import REFUSAL_RESPONSE, GroundingGuardrail, GuardResult, InputGuardrail, RelevanceGuardrail
from src.retrieval import Retriever, RetrievalResult
from src.vectorstore import VectorStore

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


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


class StageTiming(BaseModel):
    """Start/end timestamps and duration for one pipeline stage."""

    stage: str
    started_at: float
    ended_at: float
    duration_ms: float


class LatencyTrace(BaseModel):
    """Ordered timings for every stage that ran during one PipelineHarness.run() call."""

    stages: list[StageTiming] = Field(default_factory=list)

    def total_duration_ms(self) -> float:
        """Sum of all recorded stage durations, in milliseconds."""
        return sum(stage.duration_ms for stage in self.stages)


class PipelineResult(BaseModel):
    """The outcome of one PipelineHarness.run() call."""

    answer: str
    query_text: str = ""
    sources: list[Chunk] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    latency_trace: LatencyTrace
    guard_flags: dict[str, GuardResult] = Field(default_factory=dict)
    degraded: bool = False
    errors: list[StageError] = Field(default_factory=list)


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

    @staticmethod
    @contextmanager
    def _timed(latency_trace: LatencyTrace, stage: str):
        started_at = time.time()
        try:
            yield
        finally:
            ended_at = time.time()
            latency_trace.stages.append(
                StageTiming(stage=stage, started_at=started_at, ended_at=ended_at, duration_ms=(ended_at - started_at) * 1000)
            )

    def _run_stage(self, stage: str, latency_trace: LatencyTrace, errors: list[StageError], fn, *args):
        """Run a synchronous, non-retried stage; record its timing; capture failures as a StageError."""
        try:
            with self._timed(latency_trace, stage):
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
        latency_trace: LatencyTrace,
        errors: list[StageError],
    ) -> str | None:
        try:
            with self._timed(latency_trace, "stt"):
                return await asyncio.to_thread(self._transcribe_with_retry, audio_input, filename, content_type)
        except Exception as exc:
            errors.append(StageError(stage="stt", error_type=type(exc).__name__, message=str(exc)))
            return None

    def _retrieve_stage(self, query_text: str, latency_trace: LatencyTrace, errors: list[StageError]) -> RetrievalResult | None:
        try:
            if self.store.index is None or self.store.index.ntotal == 0:
                with self._timed(latency_trace, "chunking"):
                    if not self.chunks:
                        raise RuntimeError("VectorStore is empty and no chunks were provided to index")
                    self.store.build(self.chunks)
            with self._timed(latency_trace, "retrieval"):
                return self.retriever.retrieve(query_text)
        except Exception as exc:
            errors.append(StageError(stage="retrieval", error_type=type(exc).__name__, message=str(exc)))
            return None

    async def _generate_stage(
        self, query_text: str, retrieval_result: RetrievalResult, latency_trace: LatencyTrace, errors: list[StageError]
    ) -> str | None:
        try:
            with self._timed(latency_trace, "generation"):
                retrieved_chunks = list(zip(retrieval_result.chunks, retrieval_result.scores))
                return await self.generator.answer(query_text, retrieved_chunks)
        except Exception as exc:
            errors.append(StageError(stage="generation", error_type=type(exc).__name__, message=str(exc)))
            return None

    def _degraded(
        self,
        answer: str,
        latency_trace: LatencyTrace,
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
            latency_trace=latency_trace,
            guard_flags=guard_flags,
            degraded=True,
            errors=errors,
        )

    async def run(
        self, audio_or_text_input: str | bytes, filename: str = "audio.wav", content_type: str | None = None
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

        Returns:
            PipelineResult: the final answer, its source chunks, a timing
            trace, per-guardrail results, and whether the response is a
            degraded fallback (a stage technically failed) rather than a
            normal answer or guardrail refusal.
        """
        latency_trace = LatencyTrace()
        errors: list[StageError] = []
        guard_flags: dict[str, GuardResult] = {}

        # Stage 1: STT (only if the input looks like audio)
        is_audio = isinstance(audio_or_text_input, (bytes, bytearray)) or (
            isinstance(audio_or_text_input, str) and _is_audio_path(audio_or_text_input)
        )
        if is_audio:
            query_text = await self._transcribe_stage(audio_or_text_input, filename, content_type, latency_trace, errors)
            if query_text is None:
                return self._degraded(
                    "I couldn't understand the audio after a couple of tries. Could you type your question instead?",
                    latency_trace,
                    guard_flags,
                    errors,
                )
        else:
            query_text = audio_or_text_input

        # Stage 2: InputGuardrail
        input_result = self._run_stage("input_guardrail", latency_trace, errors, self.input_guardrail.check, query_text)
        if input_result is None:
            return self._degraded(REFUSAL_RESPONSE, latency_trace, guard_flags, errors, query_text=query_text)
        guard_flags["input"] = input_result
        if not input_result.allowed:
            return PipelineResult(
                answer=input_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=[],
                latency_trace=latency_trace,
                guard_flags=guard_flags,
                degraded=False,
                errors=errors,
            )

        # Stage 3: chunking/retrieval (chunking is skipped once the store is indexed)
        retrieval_result = self._retrieve_stage(query_text, latency_trace, errors)
        if retrieval_result is None:
            return self._degraded(REFUSAL_RESPONSE, latency_trace, guard_flags, errors, query_text=query_text)

        # Stage 4: RelevanceGuardrail
        relevance_result = self._run_stage(
            "relevance_guardrail", latency_trace, errors, self.relevance_guardrail.check, retrieval_result
        )
        if relevance_result is None:
            return self._degraded(
                REFUSAL_RESPONSE, latency_trace, guard_flags, errors,
                query_text=query_text, sources=retrieval_result.chunks, scores=retrieval_result.scores,
            )
        guard_flags["relevance"] = relevance_result
        if not relevance_result.allowed:
            return PipelineResult(
                answer=relevance_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=retrieval_result.chunks,
                scores=retrieval_result.scores,
                latency_trace=latency_trace,
                guard_flags=guard_flags,
                degraded=False,
                errors=errors,
            )

        # Stage 5: Generator.answer (retries happen inside Generator/LLMProvider)
        answer_text = await self._generate_stage(query_text, retrieval_result, latency_trace, errors)
        if answer_text is None:
            return self._degraded(
                "I'm having trouble generating an answer right now. Please try again shortly.",
                latency_trace,
                guard_flags,
                errors,
                query_text=query_text,
                sources=retrieval_result.chunks,
                scores=retrieval_result.scores,
            )

        # Stage 6: GroundingGuardrail
        grounding_result = self._run_stage(
            "grounding_guardrail", latency_trace, errors, self.grounding_guardrail.check, answer_text, retrieval_result.chunks
        )
        if grounding_result is None:
            return self._degraded(
                REFUSAL_RESPONSE, latency_trace, guard_flags, errors,
                query_text=query_text, sources=retrieval_result.chunks, scores=retrieval_result.scores,
            )
        guard_flags["grounding"] = grounding_result
        if not grounding_result.allowed:
            return PipelineResult(
                answer=grounding_result.response_override or REFUSAL_RESPONSE,
                query_text=query_text,
                sources=retrieval_result.chunks,
                scores=retrieval_result.scores,
                latency_trace=latency_trace,
                guard_flags=guard_flags,
                degraded=False,
                errors=errors,
            )

        # Final response
        return PipelineResult(
            answer=answer_text,
            query_text=query_text,
            sources=retrieval_result.chunks,
            scores=retrieval_result.scores,
            latency_trace=latency_trace,
            guard_flags=guard_flags,
            degraded=False,
            errors=errors,
        )

    async def run_streaming(
        self, audio_or_text_input: str | bytes, filename: str = "audio.wav", content_type: str | None = None
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
        latency_trace = LatencyTrace()
        errors: list[StageError] = []
        guard_flags: dict[str, GuardResult] = {}

        # Stage 1: STT
        is_audio = isinstance(audio_or_text_input, (bytes, bytearray)) or (
            isinstance(audio_or_text_input, str) and _is_audio_path(audio_or_text_input)
        )
        if is_audio:
            query_text = await self._transcribe_stage(audio_or_text_input, filename, content_type, latency_trace, errors)
            if query_text is None:
                yield StreamDoneEvent(
                    result=self._degraded(
                        "I couldn't understand the audio after a couple of tries. Could you type your question instead?",
                        latency_trace,
                        guard_flags,
                        errors,
                    )
                )
                return
        else:
            query_text = audio_or_text_input

        # Stage 2: InputGuardrail
        input_result = self._run_stage("input_guardrail", latency_trace, errors, self.input_guardrail.check, query_text)
        if input_result is None:
            yield StreamDoneEvent(
                result=self._degraded(REFUSAL_RESPONSE, latency_trace, guard_flags, errors, query_text=query_text)
            )
            return
        guard_flags["input"] = input_result
        if not input_result.allowed:
            yield StreamDoneEvent(
                result=PipelineResult(
                    answer=input_result.response_override or REFUSAL_RESPONSE,
                    query_text=query_text,
                    sources=[],
                    latency_trace=latency_trace,
                    guard_flags=guard_flags,
                    degraded=False,
                    errors=errors,
                )
            )
            return

        # Stage 3: chunking/retrieval
        retrieval_result = self._retrieve_stage(query_text, latency_trace, errors)
        if retrieval_result is None:
            yield StreamDoneEvent(
                result=self._degraded(REFUSAL_RESPONSE, latency_trace, guard_flags, errors, query_text=query_text)
            )
            return

        # Stage 4: RelevanceGuardrail
        relevance_result = self._run_stage(
            "relevance_guardrail", latency_trace, errors, self.relevance_guardrail.check, retrieval_result
        )
        if relevance_result is None:
            yield StreamDoneEvent(
                result=self._degraded(
                    REFUSAL_RESPONSE,
                    latency_trace,
                    guard_flags,
                    errors,
                    query_text=query_text,
                    sources=retrieval_result.chunks,
                    scores=retrieval_result.scores,
                )
            )
            return
        guard_flags["relevance"] = relevance_result
        if not relevance_result.allowed:
            yield StreamDoneEvent(
                result=PipelineResult(
                    answer=relevance_result.response_override or REFUSAL_RESPONSE,
                    query_text=query_text,
                    sources=retrieval_result.chunks,
                    scores=retrieval_result.scores,
                    latency_trace=latency_trace,
                    guard_flags=guard_flags,
                    degraded=False,
                    errors=errors,
                )
            )
            return

        # Stage 5: streaming generation, with per-sentence grounding as each one completes
        retrieved_chunks = list(zip(retrieval_result.chunks, retrieval_result.scores))
        accepted_sentences: list[str] = []
        dropped_count = 0
        buffer = ""
        started_at = time.time()
        try:
            async for delta in self.generator.stream_answer(query_text, retrieved_chunks):
                buffer += delta
                while True:
                    match = _SENTENCE_BOUNDARY_RE.search(buffer)
                    if not match:
                        break
                    sentence, buffer = buffer[: match.end()].strip(), buffer[match.end() :]
                    if not sentence:
                        continue
                    if self.grounding_guardrail.is_sentence_grounded(sentence, retrieval_result.chunks):
                        accepted_sentences.append(sentence)
                        yield StreamSentenceEvent(text=sentence)
                    else:
                        dropped_count += 1
            trailing = buffer.strip()
            if trailing:
                if self.grounding_guardrail.is_sentence_grounded(trailing, retrieval_result.chunks):
                    accepted_sentences.append(trailing)
                    yield StreamSentenceEvent(text=trailing)
                else:
                    dropped_count += 1
        except Exception as exc:
            errors.append(StageError(stage="generation", error_type=type(exc).__name__, message=str(exc)))
        finally:
            ended_at = time.time()
            latency_trace.stages.append(
                StageTiming(stage="generation", started_at=started_at, ended_at=ended_at, duration_ms=(ended_at - started_at) * 1000)
            )

        if not accepted_sentences:
            fallback = (
                "I'm having trouble generating an answer right now. Please try again shortly."
                if errors and errors[-1].stage == "generation"
                else REFUSAL_RESPONSE
            )
            yield StreamDoneEvent(
                result=self._degraded(
                    fallback,
                    latency_trace,
                    guard_flags,
                    errors,
                    query_text=query_text,
                    sources=retrieval_result.chunks,
                    scores=retrieval_result.scores,
                )
            )
            return

        guard_flags["grounding"] = GuardResult(
            allowed=True,
            reason="ok" if dropped_count == 0 else f"dropped_{dropped_count}_ungrounded_sentence(s)",
        )
        yield StreamDoneEvent(
            result=PipelineResult(
                answer=" ".join(accepted_sentences),
                query_text=query_text,
                sources=retrieval_result.chunks,
                scores=retrieval_result.scores,
                latency_trace=latency_trace,
                guard_flags=guard_flags,
                degraded=False,
                errors=errors,
            )
        )


def load_benchmark(benchmark_path: str) -> list[dict[str, Any]]:
    """
    Load a benchmark dataset of (query, expected_answer, expected_sources) records.

    Args:
        benchmark_path: filesystem path to the benchmark JSON file.

    Returns:
        list[dict]: benchmark case records.
    """
    raise NotImplementedError


def run_benchmark(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Run the pipeline over each benchmark case and compute aggregate metrics.

    Args:
        cases: benchmark case records as returned by load_benchmark.

    Returns:
        dict: aggregate metrics (e.g. accuracy, groundedness rate, avg latency)
        plus per-case results.
    """
    raise NotImplementedError


def report_results(results: dict[str, Any]) -> None:
    """
    Print/write a human-readable summary of benchmark results.

    Args:
        results: results dict as returned by run_benchmark.
    """
    raise NotImplementedError
