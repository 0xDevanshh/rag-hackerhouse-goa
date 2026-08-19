# voice-rag

Speak a question → transcribe it → retrieve grounded context → answer it → verify the answer is actually
supported by that context. A voice-driven RAG pipeline with three guardrail layers and a latency budget
that separates what we control (in-process compute) from what we don't (third-party API round trips).

## Architecture

```
                    ┌──────────────────────────┐
                    │  Browser (mic input)      │
                    │  Next.js frontend  :3000  │
                    └────────────┬──────────────┘
                                 │  POST /query
                                 │  multipart/form-data (audio)
                                 ▼
                    ┌──────────────────────────┐
                    │  FastAPI backend   :8000  │
                    │      (src/api.py)         │
                    └────────────┬──────────────┘
                                 ▼
        ┌────────────────────────────────────────────────┐
        │            PipelineHarness.run()                │
        │              (src/harness.py)                   │
        │                                                  │
        │  1. STT                (Sarvam / MockSTT)  ──────┼──▶ network-bound
        │  2. InputGuardrail                              │
        │  3. Retrieval           (FAISS + Embedder)       │
        │  4. RelevanceGuardrail                           │
        │  5. Generator          (Groq / Anthropic)  ──────┼──▶ network-bound
        │  6. GroundingGuardrail                           │
        │                                                  │
        │  every stage: timed + wrapped in try/except      │
        └────────────────────────────────────────────────┘
                                 ▼
              PipelineResult { answer, query_text, sources,
                scores, latency_trace, guard_flags, degraded }

  Offline / lazy indexing (once per process):
  data/sample_corpus.json → Chunker (src/chunking.py) → Embedder → FAISS IndexFlatIP
                                                          (src/vectorstore.py)
```

A guardrail refusal or a failed stage never crashes the request — it short-circuits to a canned
response (guardrail) or a degraded fallback (technical failure), always returning a `PipelineResult`.

## Setup

**Backend**

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in the keys you have — see "Running locally" below
```

**Frontend**

```bash
cd frontend
pnpm install            # or npm install
cp .env.local.example .env.local
```

## Running locally

The backend always needs *some* LLM key to generate answers (`Generator` reads `LLM_PROVIDER`, default
`groq`, and constructs its provider eagerly). STT is the one stage that can be fully mocked with zero
keys. Two ways to run it:

### Real keys

```bash
# .env
SARVAM_API_KEY=...          # real speech-to-text
LLM_PROVIDER=groq           # or "anthropic"
GROQ_API_KEY=...            # free tier, default provider
# ANTHROPIC_API_KEY=...     # set instead if LLM_PROVIDER=anthropic
```

```bash
venv/bin/python src/api.py          # backend on :8000
cd frontend && pnpm dev             # frontend on :3000 (falls back to :3001 etc. if taken)
```

Open the frontend, click **Record**, ask a question, click **Stop** — the browser uploads the recording
to `POST /query` and renders the answer, sources, and latency breakdown.

### Mock mode (no keys)

- `GET /health` always works — the harness isn't built until the first `/query`, so a missing key never
  blocks the health check the frontend uses to gate recording.
- `POST /query` without a working LLM key returns a clear `503 Pipeline not configured: ...` instead of
  crashing.
- To exercise the pipeline with **zero** external calls (e.g. in a script or notebook), pass a plain text
  query — `run()` detects it isn't audio and skips the STT stage entirely:

  ```python
  from src.harness import PipelineHarness
  from src.vectorstore import VectorStore

  harness = PipelineHarness(store=VectorStore(), chunks=my_chunks)
  result = await harness.run("what is retrieval augmented generation")  # str input -> STT is skipped
  ```

  To instead test the audio code path without hitting the real Sarvam API, pass `stt_client=MockSTT()`
  and call `run()` with (any) audio bytes — this is exactly what `benchmarks/run_benchmark.py` does.
- `benchmarks/run_benchmark.py` always uses `MockSTT` for its in-process numbers, and falls back to a
  clearly-labeled simulated LLM provider if no `GROQ_API_KEY`/`ANTHROPIC_API_KEY` is set, so the
  benchmark (and CI) never needs real credentials to run end-to-end.

## Chunking strategies (`src/chunking.py`)

Four strategies behind one `Chunker` interface, selectable via `ChunkerRegistry.build(name, **kwargs)`:

| Strategy | How it splits | Why it's here |
|---|---|---|
| **FixedSizeChunker** | Sliding window over characters, fixed `overlap` | Cheap, deterministic baseline with no embedding cost at ingest — the control to compare smarter strategies against |
| **SentenceSemanticChunker** | Sentences merged while running-centroid cosine similarity to the next sentence stays above `similarity_threshold` | Topic shifts happen at sentence boundaries, not fixed character counts — keeps each chunk topically coherent for better embedding/retrieval quality |
| **MetadataAwareChunker** | One chunk per MSMARCO-XI passage, tagged with `query_id`/`passage_id`/`language`/`is_selected` | MSMARCO-XI passages are already human-relevance-labeled — re-chunking would throw away that signal; keeping them whole preserves metadata for later retrieval boosting |
| **RecursiveChunker** | Paragraph-first, recursively falls back to sentences for oversized paragraphs, then greedily repacks pieces up to `max_chunk_size` | **Production default** (used by `src/api.py` and the benchmark): respects real document structure, degrades gracefully for dense text, keeps chunk count low without truncating mid-sentence |

`scripts/compare_chunkers.py` runs all four on `data/sample_corpus.json` and prints chunk count, avg
length, and a sample chunk per strategy — useful for eyeballing the tradeoffs on real content.

## Guardrails (`src/guardrails.py`) — 3 layers

Each layer spends more compute than the last, and only runs once the cheaper layer upstream has passed.
Every refusal — from any layer — returns the same canned message, so the user experience doesn't reveal
*which* check failed.

1. **InputGuardrail** *(pre-retrieval)* — rejects empty/too-short input, a small unsafe-content
   regex/keyword screen, and gibberish (alphabetic-character-ratio heuristic). Cheapest check, runs
   before any embedding or LLM spend.
2. **RelevanceGuardrail** *(post-retrieval, pre-generation)* — refuses to call the LLM at all if
   `Retriever`'s `low_confidence` flag is set or the top retrieval score is below threshold. Stops the
   model from being asked to improvise over context that doesn't actually cover the question.
3. **GroundingGuardrail** *(post-generation)* — splits the generated answer into sentences, embeds each,
   and checks max cosine similarity against the retrieved chunks. If more than 30% of sentences aren't
   supported by any retrieved passage, the whole answer is discarded — catches hallucination that slips
   past relevance filtering.

## Latency methodology (`benchmarks/`)

`benchmarks/run_benchmark.py` runs 32 queries (24 in-domain, 8 adversarial/off-topic, the latter to
exercise guardrail short-circuits) through `PipelineHarness`, with **`MockSTT` substituted for real
speech-to-text** — this is the core methodological choice: it isolates in-process compute (embedding,
FAISS search, guardrail checks, LLM generation) from network-bound STT, so the numbers reflect only what
this codebase controls. A separate 5-call pass against the real Sarvam API reports STT's own P50/P100,
explicitly labeled as network-bound and **outside** the in-process target.

**Why split them:** in-process latency is a target we can engineer against (< 200ms). STT latency is
dominated by network RTT and a third party's queueing/inference time — folding it into the same budget
would make an infra-controlled number look like it's failing an engineering target it was never meant to
measure against.

Snapshot from a run in an environment with no `GROQ_API_KEY`/`SARVAM_API_KEY` configured (generation used
the benchmark's simulated ~50ms-delay provider — **not real LLM latency**; re-run with real keys for
meaningful numbers):

| Stage | P50 | P70 | P100 (max) | Samples |
|---|---|---|---|---|
| Retrieval only | 19.3 ms | 29.8 ms | 260.9 ms | 31 |
| Generation only (simulated) | 51.1 ms | 51.1 ms | 51.2 ms | 25 |
| Total in-process (excl. STT) | 128.9 ms | 135.7 ms | 260.9 ms | 32 |
| STT (Sarvam, real) | — | — | — | skipped: no `SARVAM_API_KEY` |

Sample counts below 32 are expected, not bugs: adversarial queries correctly blocked by a guardrail
before reaching retrieval/generation simply don't contribute a sample for that stage. Full report,
regenerated on each run: [`benchmarks/latency_report.md`](benchmarks/latency_report.md).

```bash
venv/bin/python benchmarks/run_benchmark.py
```

## API reference

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"status": "ok"}` — frontend checks this before allowing recording |
| `POST /query` | multipart field `audio` → runs `PipelineHarness.run`, returns `PipelineResult` JSON |

```bash
curl -X POST http://localhost:8000/query -F "audio=@question.wav"
```

## Testing

```bash
venv/bin/python -m pytest tests/ -q
```

## Project layout

```
voice-rag/
  src/
    api.py              # FastAPI app: GET /health, POST /query
    harness.py          # PipelineHarness — orchestrates every stage below
    stt.py               # SarvamSTT + MockSTT
    chunking.py          # 4 chunking strategies + ChunkerRegistry
    vectorstore.py       # Embedder (multilingual) + FAISS VectorStore
    retrieval.py         # Retriever — rerank + low-confidence query rewrite
    guardrails.py        # InputGuardrail / RelevanceGuardrail / GroundingGuardrail
    generation.py        # Generator — swappable GroqProvider / AnthropicProvider
    data_loader.py       # ai4bharat/MSMARCO-XI streaming loader (separate from the demo corpus below)
    pipeline.py          # skeleton, superseded by api.py + harness.py — not implemented
    config.py            # skeleton, not implemented
  frontend/              # Next.js (App Router, TS) — single-page recorder UI
  data/sample_corpus.json  # small demo corpus, indexed by the live API and benchmark
  scripts/               # inspect_dataset.py, compare_chunkers.py — dev tools
  benchmarks/            # run_benchmark.py + generated latency_report.md
  tests/
```
