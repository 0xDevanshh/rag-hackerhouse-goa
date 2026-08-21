"""
FastAPI backend for the voice-rag demo.

Exposes the PipelineHarness over HTTP: POST /query accepts an uploaded audio
file and runs it through the full voice-RAG pipeline, POST /query/text takes an
already-typed question (the only path that can meet a sub-200ms budget, since
it skips the network-bound speech-to-text round trip), and GET /health lets
the frontend check backend availability before allowing recording.
"""

import json
import logging
import os
import sys
import uuid
import asyncio
import json as json_module
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if __name__ == "__main__" and str(ROOT_DIR) not in sys.path:
    # Allow `python src/api.py` to work directly: when run as a script (not
    # via `-m` or an installed package), Python puts src/ on sys.path
    # instead of the repo root, so `from src....` below would otherwise fail.
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import websockets

from src import data_loader
from src.chunking import ChunkerRegistry
from src.harness import PipelineHarness, PipelineResult
from src.latency import RequestTrace
from src.vectorstore import Embedder, VectorStore

load_dotenv(ROOT_DIR / ".env")

# Without this, src/harness.py's per-request latency breakdown (logger.info)
# is silently dropped: the root logger's default level is WARNING and it has
# no handler attached, so INFO records go nowhere. This makes them show up
# on stdout — i.e. the terminal running `python src/api.py` / uvicorn.
#
# Deliberately scoped to just the "src.harness" logger (not
# logging.basicConfig on the root logger) — that would also flip every
# third-party library's logger to INFO (huggingface_hub, httpx, urllib3,
# ...), flooding the output with unrelated request/download noise. The
# timestamp prefix (once per block, since it's one multi-line log record)
# helps tell consecutive requests' breakdowns apart in a live log stream.
#
# "src.api" gets the same treatment for the same reason: which corpus the
# index was actually built from (MSMARCO-XI or the demo fallback) is logged
# at INFO, and that line needs to reach the operator's terminal.
# "src.vectorstore" and "src.generation" are included so model-load device and
# provider prewarm results are visible at startup.
_latency_log_handler = logging.StreamHandler()
_latency_log_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
for _logger_name in ("src.harness", "src.api", "src.vectorstore", "src.generation"):
    _scoped_logger = logging.getLogger(_logger_name)
    _scoped_logger.addHandler(_latency_log_handler)
    _scoped_logger.setLevel(logging.INFO)
    _scoped_logger.propagate = False

# Named explicitly, not via __name__: running this module directly
# (`python src/api.py`) makes __name__ "__main__", which would miss the
# handler attached above.
logger = logging.getLogger("src.api")

DEFAULT_CORPUS_PATH = ROOT_DIR / "data" / "sample_corpus.json"
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "http://localhost:3000")
SARVAM_REALTIME_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

# Corpus selection. The served index is ai4bharat/MSMARCO-XI by default,
# indexed in both the target language and the original English (so a query in
# either language can retrieve), falling back to the bundled demo corpus if
# the dataset can't be loaded.
CORPUS = os.environ.get("CORPUS", "msmarco").strip().lower()
if CORPUS not in ("msmarco", "demo", "both"):
    logger.warning("unknown CORPUS=%r; falling back to 'msmarco'", CORPUS)
    CORPUS = "msmarco"

CORPUS_LANGUAGE = os.environ.get("CORPUS_LANGUAGE", "hi")
CORPUS_SPLIT = os.environ.get("CORPUS_SPLIT", "validation")
CORPUS_LIMIT = int(os.environ.get("CORPUS_LIMIT", "500"))
CORPUS_INCLUDE_ENGLISH = os.environ.get("CORPUS_INCLUDE_ENGLISH", "1") != "0"

_harness: PipelineHarness | None = None
_harness_init_error: str | None = None

# Completed request traces, newest last. Kept so a benchmark can read the
# *server-side* accounting including the slices that are measured after the
# handler returns — serialization and response flush cannot appear in a
# response body that is describing itself. Bounded, and off unless
# TRACE_BUFFER_SIZE is set, so it can't grow without limit in production.
TRACE_BUFFER_SIZE = int(os.environ.get("TRACE_BUFFER_SIZE", "0"))
_recent_traces: "deque[dict]" = deque(maxlen=TRACE_BUFFER_SIZE or 1)


def _load_sample_chunks(corpus_path: Path = DEFAULT_CORPUS_PATH) -> list:
    """Chunk the bundled sample corpus into indexable Chunk records."""
    with corpus_path.open("r", encoding="utf-8") as f:
        documents = json.load(f)
    chunker = ChunkerRegistry.build("recursive", max_chunk_size=300)
    return [chunk for doc in documents for chunk in chunker.chunk(doc)]


def _load_msmarco_chunks() -> list:
    """
    Chunk MSMARCO-XI into indexable Chunk records, one chunk per passage.

    Uses the metadata_aware strategy rather than a size-based one: MSMARCO-XI
    passages are already short, self-contained, and relevance-labeled, so
    re-splitting them would discard the is_selected signal that
    Retriever's rerank boost and the eval set both depend on.
    """
    documents = data_loader.load_chunker_docs(
        language=CORPUS_LANGUAGE,
        split=CORPUS_SPLIT,
        limit=CORPUS_LIMIT,
        include_english=CORPUS_INCLUDE_ENGLISH,
    )
    chunker = ChunkerRegistry.build("metadata_aware")
    return [chunk for doc in documents for chunk in chunker.chunk(doc)]


def _load_default_chunks() -> list:
    """
    Build the chunk set the API serves, selected by the CORPUS env var:

      "msmarco" (default) — ai4bharat/MSMARCO-XI only. Real search-engine
          queries about arbitrary topics. This is the corpus the retrieval
          evaluation scores against, so it is the right default for measuring
          quality — but it means an off-corpus question ("what is retrieval
          augmented generation") is *correctly* refused by RelevanceGuardrail,
          because the answer genuinely isn't in there.
      "demo" — the bundled 2-document corpus only. Covers RAG and voice-
          pipeline concepts, so questions about how this system works get real
          answers. Too small for meaningful retrieval metrics.
      "both" — demo documents indexed alongside MSMARCO-XI. Costs a couple of
          extra chunks and lets the demo answer questions about itself without
          giving up the MSMARCO corpus.

    MSMARCO loading falls back to the demo corpus on failure, which keeps the
    backend startable with no network and no dataset cache (a fresh clone, an
    offline demo, CI). Which corpus actually won is logged either way: the two
    produce very different answers, and silently serving 2 documents while
    believing MSMARCO-XI is loaded would be badly misleading.
    """
    if CORPUS == "demo":
        chunks = _load_sample_chunks()
        logger.info("serving the bundled demo corpus only (%d chunks, CORPUS=demo)", len(chunks))
        return chunks

    try:
        chunks = _load_msmarco_chunks()
    except Exception as exc:
        logger.warning(
            "could not load MSMARCO-XI (%s: %s); falling back to the bundled demo corpus at %s",
            type(exc).__name__,
            exc,
            DEFAULT_CORPUS_PATH,
        )
        return _load_sample_chunks()

    if CORPUS == "both":
        demo_chunks = _load_sample_chunks()
        chunks = chunks + demo_chunks
        logger.info("added %d demo chunks alongside MSMARCO-XI (CORPUS=both)", len(demo_chunks))

    languages = sorted({chunk.metadata.get("language") for chunk in chunks})
    logger.info(
        "indexed %d chunks from MSMARCO-XI (%s/%s, limit=%d, languages=%s)",
        len(chunks),
        CORPUS_LANGUAGE,
        CORPUS_SPLIT,
        CORPUS_LIMIT,
        ",".join(str(lang) for lang in languages),
    )
    return chunks


def get_harness() -> PipelineHarness:
    """
    Build (once) and return the PipelineHarness backing /query.

    Construction is lazy and cached: it happens on the first request that
    needs it, not at import time, so /health stays available even if the
    harness can't be built yet (e.g. no GROQ_API_KEY/ANTHROPIC_API_KEY set).
    A prior failure is remembered and re-raised as a 503 rather than
    retrying the (possibly slow) build on every request.
    """
    global _harness, _harness_init_error
    if _harness is not None:
        return _harness
    if _harness_init_error is not None:
        raise HTTPException(status_code=503, detail=f"Pipeline not configured: {_harness_init_error}")
    try:
        store = VectorStore()
        _harness = PipelineHarness(store=store, chunks=_load_default_chunks())
        return _harness
    except Exception as exc:
        _harness_init_error = str(exc)
        raise HTTPException(status_code=503, detail=f"Pipeline not configured: {exc}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Eagerly build the harness, index its corpus, and pay every cold-start
    cost at process startup rather than inside whichever request happens to
    arrive first.

    The index build was already done here. What's added is
    PipelineHarness.prewarm(), which covers the three costs that a live
    request was still absorbing:

      - the embedding model's lazy initialization: ~690ms on the very first
        encode of a process, plus a fresh per-shape cost on MPS
      - the LLM provider's DNS + TCP + TLS handshake: ~80ms to api.groq.com
      - the STT provider's handshake: ~100ms to api.sarvam.ai

    A one-time cost belongs at startup, not inside a live request's budget.
    """
    try:
        harness = get_harness()
    except HTTPException:
        # No LLM key configured yet: get_harness() already recorded
        # _harness_init_error, so /query will return a clear 503 on first
        # use. Nothing to warm up until that's fixed (and the process
        # restarted) — startup itself must still succeed either way.
        yield
        return
    # Via the harness, not store.build() directly, so the index version the
    # answer cache keys on actually reflects this build — see
    # PipelineHarness.build_index.
    await asyncio.to_thread(harness.build_index)
    await harness.prewarm()
    yield


class LatencyTraceMiddleware:
    """
    Pure-ASGI middleware that owns the request's wall clock.

    This is the piece that makes the trace's arithmetic honest. The previous
    breakdown reported total_ms as the sum of the stages it knew about, so
    every millisecond spent outside a named stage was invisible *by
    construction* — no amount of staring at that log could reveal the
    ~958ms it was omitting. Here total_ms is measured independently, from ASGI
    entry to final body flush, and anything the pipeline doesn't claim shows
    up as unaccounted_ms.

    Deliberately pure ASGI rather than a BaseHTTPMiddleware subclass:
    BaseHTTPMiddleware wraps the response in an anyio task group and buffers
    streaming bodies, which both adds latency of its own and would break the
    Server-Sent Events on /query/stream.

    The slices measured here — and why they need marks rather than a `with`
    block — are boundaries that fall in different callbacks:

      middleware_ms     ASGI entry -> route handler entered. The framework's
                        own routing plus the CORS middleware. (No auth_ms:
                        this service has no authentication, so there is no
                        such stage to measure. Reporting a fabricated 0.0 for
                        one would be worse than its absence.)
      serialization_ms  route handler returned -> response headers sent, i.e.
                        Pydantic model dump plus JSON encoding.
      response_write_ms response headers sent -> last body chunk flushed.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        trace = RequestTrace()
        trace.start()
        trace.label("request_id", uuid.uuid4().hex)
        trace.mark("arrival")
        # Starlette exposes scope["state"] as request.state, which is how the
        # route handler below gets hold of this trace.
        scope.setdefault("state", {})["trace"] = trace

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                trace.mark("response_start")
                trace.span_between("serialization", "route_end", "response_start")
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                trace.mark("response_flushed")
                trace.span_between("response_write", "response_start", "response_flushed")

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            trace.finish()
            if TRACE_BUFFER_SIZE:
                _recent_traces.append(
                    {
                        "path": scope.get("path"),
                        "spans": trace.as_dict(),
                        "labels": dict(trace.labels),
                    }
                )
            # Only log here for requests the pipeline itself didn't already
            # report on (health checks, 4xx/5xx, static routes). A pipeline
            # request logs its own complete breakdown; logging twice would
            # double every trace in the operator's terminal.
            if not trace.labels.get("pipeline"):
                logger.info(
                    "%s %s -> total_ms: %.1f unaccounted_ms: %.1f",
                    scope.get("method"),
                    scope.get("path"),
                    trace.total_ms,
                    trace.unaccounted_ms,
                )


app = FastAPI(title="voice-rag API", lifespan=lifespan)

# Order matters: added last means outermost, so LatencyTraceMiddleware wraps
# CORSMiddleware and its cost lands inside middleware_ms rather than escaping
# into unaccounted_ms.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LatencyTraceMiddleware)


def _request_trace(request: Request) -> RequestTrace:
    """
    The trace LatencyTraceMiddleware started for this request, marked as
    pipeline-owned so the middleware doesn't log a second summary line.

    Falls back to a fresh trace if the middleware isn't installed (e.g. a unit
    test calling the route function directly), so the routes never depend on it
    being there.
    """
    trace = getattr(request.state, "trace", None)
    if trace is None:
        trace = RequestTrace()
        trace.start()
    trace.label("pipeline", True)
    trace.mark("route_start")
    trace.span_between("middleware", "arrival", "route_start")
    return trace


@app.get("/health")
def health() -> dict:
    """Liveness check for the frontend to confirm the backend is reachable."""
    return {"status": "ok"}


@app.get("/stt/realtime/config")
def realtime_stt_config() -> JSONResponse:
    """Return a browser-safe realtime config when a short-lived token is supplied.

    Sarvam documents API keys, but does not document a token-mint endpoint. A
    deployment may inject a separately scoped, short-lived token as
    SARVAM_REALTIME_TOKEN. The long-lived SARVAM_API_KEY is never returned.
    Without that token, the browser must use the server proxy below.
    """
    token = os.environ.get("SARVAM_REALTIME_TOKEN")
    if not token or token.startswith("your_"):
        return JSONResponse(status_code=404, content={"direct": False})
    return {
        "direct": True,
        "url": SARVAM_REALTIME_URL,
        "protocol": f"api-subscription-key.{token}",
        "query": {
            "language_code": "auto",
            "model": "saaras:v3-realtime",
            "stream_type": "fast",
            "endpointing": "vad",
            "encoding": "linear16",
            "sample_rate": "16000",
            "mode": "transcribe",
            "prefix_padding_ms": "100",
            # Lower endpointing reduces final-result delay, at a small risk of
            # cutting off speech during natural pauses; partials are unaffected.
            "silence_duration_ms": "100",
            "min_speech_duration_ms": "100",
        },
    }


@app.websocket("/stt/realtime")
async def stt_realtime(websocket: WebSocket) -> None:
    """Proxy browser PCM frames to Sarvam realtime STT without exposing the key."""
    await websocket.accept()
    session_started = asyncio.get_running_loop().time()
    first_audio_received: float | None = None
    first_audio_forwarded: float | None = None
    first_partial_received: float | None = None
    last_audio_received: float | None = None
    last_partial_received: float | None = None
    audio_frames = 0
    audio_bytes = 0
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key or api_key.startswith("your_"):
        await websocket.close(code=1011, reason="SARVAM_API_KEY is not configured")
        return

    query = (
        "language_code=auto&model=saaras:v3-realtime&stream_type=fast&"
        "endpointing=vad&encoding=linear16&sample_rate=16000&mode=transcribe&"
        "prefix_padding_ms=100&silence_duration_ms=100&min_speech_duration_ms=100"
    )
    upstream_url = f"wss://api.sarvam.ai/speech-to-text-realtime/ws?{query}"
    try:
        async with websockets.connect(
            upstream_url,
            subprotocols=[f"api-subscription-key.{api_key}"],
            ping_interval=20,
            open_timeout=10,
        ) as upstream:
            async def client_to_sarvam() -> None:
                nonlocal first_audio_received, first_audio_forwarded, last_audio_received, audio_frames, audio_bytes
                while True:
                    message = await websocket.receive_text()
                    payload = json_module.loads(message)
                    if payload.get("event") == "audio_input":
                        received_at = asyncio.get_running_loop().time()
                        if first_audio_received is None:
                            first_audio_received = received_at
                        last_audio_received = received_at
                        audio_frames += 1
                        audio_bytes += len(payload.get("audio", "")) * 3 // 4
                        if first_audio_forwarded is None:
                            first_audio_forwarded = received_at
                    await upstream.send(message)

            async def sarvam_to_client() -> None:
                nonlocal first_partial_received, last_partial_received
                async for message in upstream:
                    if first_partial_received is None:
                        try:
                            if json_module.loads(message).get("event") == "transcript.partial":
                                first_partial_received = asyncio.get_running_loop().time()
                        except (TypeError, json_module.JSONDecodeError):
                            pass
                    if first_partial_received is not None:
                        last_partial_received = asyncio.get_running_loop().time()
                    await websocket.send_text(message)

            forward_task = asyncio.create_task(client_to_sarvam())
            receive_task = asyncio.create_task(sarvam_to_client())
            done, pending = await asyncio.wait(
                {forward_task, receive_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                exception = task.exception()
                if exception and not isinstance(exception, (WebSocketDisconnect, websockets.ConnectionClosed)):
                    raise exception
    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.warning("realtime STT proxy closed: %s", exc)
        if websocket.client_state.value == "CONNECTED":
            await websocket.close(code=1011, reason="realtime STT unavailable")
    finally:
        ended = asyncio.get_running_loop().time()
        logger.info(
            "realtime_stt frames=%d audio_ms=%.1f receive_ms=%s forward_ms=%s "
            "first_partial_ms=%s first_partial_after_audio_ms=%s "
            "upstream_partial_window_ms=%s session_ms=%.1f",
            audio_frames,
            audio_bytes / 32.0,
            f"{(first_audio_received - session_started) * 1000:.1f}" if first_audio_received else "none",
            f"{(first_audio_forwarded - session_started) * 1000:.1f}" if first_audio_forwarded else "none",
            f"{(first_partial_received - session_started) * 1000:.1f}" if first_partial_received else "none",
            f"{(first_partial_received - first_audio_received) * 1000:.1f}"
            if first_partial_received and first_audio_received
            else "none",
            f"{(last_partial_received - first_partial_received) * 1000:.1f}"
            if last_partial_received and first_partial_received
            else "none",
            (ended - session_started) * 1000,
        )


@app.get("/ready")
def ready() -> dict:
    """Report whether the indexed corpus and generation provider are usable."""
    if _harness is None:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "pipeline_not_initialized"})
    if _harness.store.index is None or _harness.store.index.ntotal == 0:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "index_unavailable"})
    provider = getattr(_harness.generator, "provider", None)
    provider_ready = provider is not None
    if not provider_ready:
        return JSONResponse(status_code=503, content={"ready": False, "reason": "llm_unavailable"})
    return {
        "ready": True,
        "index_chunks": int(_harness.store.index.ntotal),
        "embedding_model": _harness.store.embedder.model_name,
        "llm_provider": type(provider).__name__,
    }


@app.get("/metrics/cache")
def cache_metrics() -> dict:
    """
    Cache effectiveness counters, for checking that the fast paths are
    actually being hit rather than assuming they are.
    """
    harness = get_harness()
    return {
        "embedding_cache": Embedder.cache_stats(),
        "answer_cache": {
            "size": len(harness._result_cache),
            "index_version": harness._index_version,
        },
    }


@app.get("/metrics/traces")
def recent_traces() -> dict:
    """
    The most recent completed request traces, including the serialization and
    response-flush slices that a response body cannot report about itself.
    Empty unless TRACE_BUFFER_SIZE is set.
    """
    return {"enabled": bool(TRACE_BUFFER_SIZE), "traces": list(_recent_traces)}


class TextQuery(BaseModel):
    """A typed (already-transcribed) question."""

    query: str


@app.post("/query/text", response_model=PipelineResult)
async def query_text(body: TextQuery, request: Request) -> PipelineResult:
    """
    Fast path: run one already-typed question through the pipeline using the
    local extractive provider (zero network calls beyond embedding + FAISS).

    This is the default text endpoint and what the frontend uses after STT.
    Expected total latency: ~30-50ms (embedding + BM25 + FAISS + grounding
    short-circuit).  Grounding is verified structurally (every answer sentence
    is a verbatim word-subset of a retrieved passage) rather than by a second
    embedding pass, so grounding_guard_ms ≈ 0ms on this path.

    For LLM-quality answers use POST /query/text/llm (explicit opt-in).
    """
    from src.generation import ExtractiveProvider, Generator

    trace = _request_trace(request)
    harness = get_harness()
    result = await harness.run(
        body.query,
        trace=trace,
        generator=Generator(provider=ExtractiveProvider()),
    )
    trace.mark("route_end")
    return result


@app.post("/query/text/llm", response_model=PipelineResult)
async def query_text_llm(body: TextQuery, request: Request) -> PipelineResult:
    """
    Quality path: run one already-typed question through the pipeline using
    the configured LLM provider (Groq / Anthropic, selected by LLM_PROVIDER).

    This path runs two network round-trips: one to the LLM (~700-1300ms) and
    one embedding pass for grounding verification.  It cannot meet the <200ms
    latency target.  Use it when answer quality matters more than latency, e.g.
    complex multi-sentence questions or questions that need synthesis across
    multiple passages.

    For the fast default path use POST /query/text.
    """
    trace = _request_trace(request)
    harness = get_harness()
    result = await harness.run(body.query, trace=trace)  # uses harness.generator = Groq/Anthropic
    trace.mark("route_end")
    return result


@app.post("/query", response_model=PipelineResult)
async def query(audio: UploadFile = File(...), request: Request = None) -> PipelineResult:
    """
    Run one uploaded audio recording through the full voice-RAG pipeline.

    Args:
        audio: the recorded audio file (multipart/form-data field "audio").

    Returns:
        PipelineResult: the pipeline's answer, sources, latency trace,
        guard flags, and degraded status.

    Note the trace carried in the response body is complete up to the moment
    the handler returns; serialization_ms and response_write_ms are measured
    after that and so appear only in the logged breakdown, not in the body
    describing itself.
    """
    trace = _request_trace(request)
    harness = get_harness()
    with trace.span("body_parse"):
        audio_bytes = await audio.read()
        trace.label("audio_bytes", len(audio_bytes))
    result = await harness.run(
        audio_bytes,
        filename=audio.filename or "audio.wav",
        content_type=audio.content_type,
        trace=trace,
    )
    trace.mark("route_end")
    return result


@app.post("/query/realtime", response_model=PipelineResult)
async def query_realtime(audio: UploadFile = File(...), request: Request = None) -> PipelineResult:
    """
    Lowest-latency audio path: overlaps retrieval with STT by firing retrieval
    on the first *stable* partial transcript while Sarvam is still streaming
    the audio.  When the final transcript arrives and retrieval is already
    complete, the pipeline skips straight to guardrail + fast grounded answer,
    saving the retrieval round-trip (~80ms P50) from the critical path.

    Uses the fast grounded (local extractive) answer path by default so the
    remote LLM (~700-1300ms) doesn't undo the overlap savings.  Set
    LLM_PROVIDER=groq/anthropic in the environment to switch to a hosted model.

    The trace in the response body includes:
      - stt_to_first_partial_ms : time from stream open to first partial
      - stt_final_ms            : full STT wall clock (stream open → final)
      - retrieval_on_partial_ms : wall clock from stable partial to retrieval done
      - stt_overlap_savings_ms  : retrieval ms that ran concurrently with STT
      - total_ms                : ASGI entry → response flush (never adjusted)

    Requires SARVAM_API_KEY; falls back to a 503 if missing.
    """
    trace = _request_trace(request)
    harness = get_harness()
    with trace.span("body_parse"):
        audio_bytes = await audio.read()
        trace.label("audio_bytes", len(audio_bytes))
    result = await harness.run_overlapped(
        audio_bytes,
        filename=audio.filename or "audio.wav",
        content_type=audio.content_type,
        trace=trace,
    )
    trace.mark("route_end")
    return result


class RealtimeTextQuery(BaseModel):
    """A pre-transcribed question for benchmarking the overlapped path."""

    query: str


@app.post("/query/realtime/text", response_model=PipelineResult)
async def query_realtime_text(body: RealtimeTextQuery, request: Request) -> PipelineResult:
    """
    Benchmark-only endpoint: runs the overlapped pipeline path with a
    MockRealtimeSTT that emits canned partial + final events for `query`,
    so the overlap instrumentation is exercised without a live microphone
    or SARVAM_API_KEY.

    The trace includes the same overlap spans as /query/realtime, making
    it possible to measure the post-STT pipeline latency (embedding,
    FAISS, BM25, guardrails, fast answer) in an automated benchmark.
    """
    from src.stt import MockRealtimeSTT

    trace = _request_trace(request)
    harness = get_harness()

    mock = MockRealtimeSTT()
    mock._transcripts["unknown"] = body.query

    result = await harness.run_overlapped(
        b"mock-audio",          # ignored by MockRealtimeSTT
        filename="mock.wav",
        trace=trace,
        realtime_stt_client=mock,
    )
    trace.mark("route_end")
    return result


@app.post("/query/stream")
async def query_stream(audio: UploadFile = File(...), request: Request = None) -> StreamingResponse:
    """
    Streaming counterpart to /query: same input, but sends each accepted
    (grounding-checked) sentence to the client as Server-Sent Events as
    soon as it's ready, instead of waiting for the complete answer. Reduces
    perceived latency for the (network-bound, non-optimizable) generation
    stage without weakening GroundingGuardrail — see
    PipelineHarness.run_streaming.

    Emits `event: sentence` frames with `{"text": "..."}` as they're
    accepted, followed by exactly one `event: done` frame with the full
    PipelineResult JSON (mirroring POST /query's response body).
    """
    trace = _request_trace(request)
    harness = get_harness()
    with trace.span("body_parse"):
        audio_bytes = await audio.read()
        trace.label("audio_bytes", len(audio_bytes))
    filename = audio.filename or "audio.wav"
    content_type = audio.content_type

    async def event_stream():
        async for event in harness.run_streaming(
            audio_bytes, filename=filename, content_type=content_type, trace=trace
        ):
            yield f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"
        trace.mark("route_end")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
