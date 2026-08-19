"""
FastAPI backend for the voice-rag demo.

Exposes the PipelineHarness over HTTP: POST /query accepts an uploaded audio
file and runs it through the full voice-RAG pipeline, and GET /health lets
the frontend check backend availability before allowing recording.
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if __name__ == "__main__" and str(ROOT_DIR) not in sys.path:
    # Allow `python src/api.py` to work directly: when run as a script (not
    # via `-m` or an installed package), Python puts src/ on sys.path
    # instead of the repo root, so `from src....` below would otherwise fail.
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.chunking import ChunkerRegistry
from src.harness import PipelineHarness, PipelineResult
from src.vectorstore import VectorStore

load_dotenv(ROOT_DIR / ".env")

DEFAULT_CORPUS_PATH = ROOT_DIR / "data" / "sample_corpus.json"
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "http://localhost:3000")

app = FastAPI(title="voice-rag API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_harness: PipelineHarness | None = None
_harness_init_error: str | None = None


def _load_default_chunks(corpus_path: Path = DEFAULT_CORPUS_PATH) -> list:
    """Chunk the bundled sample corpus into indexable Chunk records."""
    with corpus_path.open("r", encoding="utf-8") as f:
        documents = json.load(f)
    chunker = ChunkerRegistry.build("recursive", max_chunk_size=300)
    return [chunk for doc in documents for chunk in chunker.chunk(doc)]


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


@app.on_event("startup")
def _warm_up_harness() -> None:
    """
    Eagerly build the harness and index its corpus at process startup,
    instead of lazily on the first request. Without this, whichever request
    happens to arrive first pays the one-time corpus-chunking/embedding-
    model-loading cost (roughly 1-1.5s) as part of its own latency — a
    one-time cost belongs at startup, not inside a live request's budget.
    """
    try:
        harness = get_harness()
    except HTTPException:
        # No LLM key configured yet: get_harness() already recorded
        # _harness_init_error, so /query will return a clear 503 on first
        # use. Nothing to warm up until that's fixed (and the process
        # restarted) — startup itself must still succeed either way.
        return
    harness.store.build(harness.chunks)


@app.get("/health")
def health() -> dict:
    """Liveness check for the frontend to confirm the backend is reachable."""
    return {"status": "ok"}


@app.post("/query", response_model=PipelineResult)
async def query(audio: UploadFile = File(...)) -> PipelineResult:
    """
    Run one uploaded audio recording through the full voice-RAG pipeline.

    Args:
        audio: the recorded audio file (multipart/form-data field "audio").

    Returns:
        PipelineResult: the pipeline's answer, sources, latency trace,
        guard flags, and degraded status.
    """
    harness = get_harness()
    audio_bytes = await audio.read()
    return await harness.run(audio_bytes, filename=audio.filename or "audio.wav", content_type=audio.content_type)


@app.post("/query/stream")
async def query_stream(audio: UploadFile = File(...)) -> StreamingResponse:
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
    harness = get_harness()
    audio_bytes = await audio.read()
    filename = audio.filename or "audio.wav"
    content_type = audio.content_type

    async def event_stream():
        async for event in harness.run_streaming(audio_bytes, filename=filename, content_type=content_type):
            yield f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
