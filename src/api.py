"""
FastAPI backend for the voice-rag demo.

Exposes the PipelineHarness over HTTP: POST /query accepts an uploaded audio
file and runs it through the full voice-RAG pipeline, and GET /health lets
the frontend check backend availability before allowing recording.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.chunking import ChunkerRegistry
from src.harness import PipelineHarness, PipelineResult
from src.vectorstore import VectorStore

ROOT_DIR = Path(__file__).resolve().parent.parent
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
    return await harness.run(audio_bytes)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
