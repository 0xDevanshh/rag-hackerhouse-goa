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
        │  3. Retrieval           (FAISS + BM25 + RRF)    │
        │  4. RelevanceGuardrail                           │
        │  5. Generator          (Groq / Anthropic)  ──────┼──▶ network-bound
        │  6. GroundingGuardrail                           │
        │                                                  │
        │  every stage: timed + wrapped in try/except      │
        └────────────────────────────────────────────────┘
                                 ▼
              PipelineResult { answer, query_text, sources,
                scores, latency_trace, guard_flags, degraded }

  Offline indexing (once, at startup):
  ai4bharat/MSMARCO-XI → data_loader.py → MetadataAwareChunker → Embedder + BM25 → FAISS/RRF
   (hi + en passages)    (src/data_loader.py)  (src/chunking.py)        (src/vectorstore.py)
     └─ falls back to data/sample_corpus.json + RecursiveChunker if the dataset can't be loaded
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

## Corpus: ai4bharat/MSMARCO-XI (`src/data_loader.py`)

MS MARCO translated into 13 Indic languages (~11.5M rows / 55.6GB). Each row carries a query, ~10
candidate passages, and a reference answer in **both** the target language and the original English,
plus per-passage `is_selected` relevance labels.

The API indexes Hindi **and** English by default (both come from the same row, so no extra download),
tagged `language: "hi"` / `"en"`. That makes the retriever's language-match boost and the vector store's
metadata filter meaningful, and lets a query in either language retrieve.

Configure via env vars: `CORPUS_LANGUAGE` (default `hi`), `CORPUS_SPLIT` (`validation`), `CORPUS_LIMIT`
(`500`), `CORPUS_INCLUDE_ENGLISH` (`1`). Slices are cached under `data/msmarco_xi_cache/`, so only the
first run touches the network; if the dataset can't be loaded at all, the API logs a warning and falls
back to `data/sample_corpus.json` so it still starts offline.

**Two loading traps this dataset sets.** It publishes exactly one builder config, `default`, and encodes
language in the *parquet filename* (`validation/hinval.parquet`) with a 3-letter code:

- Passing a language as the config — `load_dataset("ai4bharat/MSMARCO-XI", "hi", ...)` — raises
  `ValueError: BuilderConfig 'hi' not found. Available: ['default']`.
- Passing **no** config silently concatenates every language in file order, so the first rows come back
  Assamese, not Hindi — a wrong-language index with no error.

`_data_file()` addresses the parquet directly via `data_files=` to avoid both. Note Telugu ships a
validation parquet but no train parquet, and `datasets` ≥ 4 ignores the repo's `ms_marco_translations.py`
loading script entirely.

## Retrieval (`src/retrieval.py`)

Each request uses the locally warmed multilingual embedding model for dense FAISS search and an
in-process BM25 index built once during startup. The two ranked candidate lists are combined with
configurable reciprocal-rank fusion (`RRF_K`, default 60), then lightly reranked with language and
metadata signals. BM25 construction is never performed in a request handler. Retrieval traces expose
`vector_search`, `bm25`, `fusion`, and `reranking` separately, so the RAG SLA cannot hide lexical work.

## Retrieval evaluation (`benchmarks/run_eval.py`)

`is_selected` doubles as retrieval ground truth, so retrieval accuracy is measurable without an LLM:
hit rate@k, recall@k, precision@k, and MRR, split by query language. No API keys, no LLM spend,
deterministic. Writes `data/eval_set.json` and `benchmarks/eval_report.md`.

Two things the scoring has to get right:

- **A passage counts once.** Each passage is indexed in both languages, so results are deduped by
  `(query_id, passage_id)` before ranks are assigned — otherwise one passage retrieved in two languages
  would look like two distinct hits.
- **`is_selected` must not be a ranking feature.** `Retriever` boosts chunks tagged `is_selected` by
  `+0.1` — but that flag *is* the label being scored, so leaving it on leaks ground truth into the
  ranking. The eval passes `is_selected_boost=0.0` and reports the boosted run alongside as a control;
  on the Hindi validation slice the leak is worth **+40pp hit rate@1** (22.8% → 62.5%). The boost is
  also meaningless in production: `is_selected` is relevance *relative to the dataset's own query*, not
  to whatever a user asks. Consider setting `IS_SELECTED_BOOST = 0`.

About half the rows have no `is_selected` passage at all (252 of 500 in the Hindi validation slice);
those are dropped from the eval set — scoring them would count an unlabeled query as a miss — but their
passages stay in the index as distractors.

## Chunking strategies (`src/chunking.py`)

Four strategies behind one `Chunker` interface, selectable via `ChunkerRegistry.build(name, **kwargs)`:

| Strategy | How it splits | Why it's here |
|---|---|---|
| **FixedSizeChunker** | Sliding window over characters, fixed `overlap` | Cheap, deterministic baseline with no embedding cost at ingest — the control to compare smarter strategies against |
| **SentenceSemanticChunker** | Sentences merged while running-centroid cosine similarity to the next sentence stays above `similarity_threshold` | Topic shifts happen at sentence boundaries, not fixed character counts — keeps each chunk topically coherent for better embedding/retrieval quality |
| **MetadataAwareChunker** | One chunk per MSMARCO-XI passage, tagged with `query_id`/`passage_id`/`language`/`is_selected` | **Production default for MSMARCO-XI** (used by `src/api.py` and `benchmarks/run_eval.py`): passages are already short and self-contained, so re-splitting them would only discard the `is_selected` label the eval set scores against |
| **RecursiveChunker** | Paragraph-first, recursively falls back to sentences for oversized paragraphs, then greedily repacks pieces up to `max_chunk_size` | Default for free-form documents (the `data/sample_corpus.json` fallback): respects real document structure, degrades gracefully for dense text, keeps chunk count low without truncating mid-sentence |

Sentence-boundary detection for the sentence-based strategies lives in `src/text.py`, shared with the
grounding guardrail and answer streaming. It matches the Indic and Urdu terminators (`।`, `॥`, `۔`, `؟`)
alongside `.!?` — Hindi ends sentences with a danda, so a Latin-only `[.!?]` split would treat an entire
multi-sentence Hindi passage or answer as one sentence.

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
   and checks max cosine similarity against the retrieved chunks. Catches hallucination that slips past
   relevance filtering.

   Two details matter for whether correct answers survive it:

   - **Similarity is measured in two passes.** Pass 1 compares each sentence against whole-chunk vectors,
     read free out of the FAISS index. That is cheap but pessimistic — a one-sentence paraphrase of one
     sentence inside a 300-character passage is diluted by the rest of the passage. On a correct,
     fully-grounded answer, pass 1 alone scored 0.600 / 0.702 / 0.505 against a 0.5 threshold: passing,
     but so narrowly that any wording drift tipped a sentence under. Pass 2 re-scores **only** the failing
     sentences against the chunks' individual sentences, where the same three score 0.721 / 0.890 / 1.000.
     Hallucinations ("RAG was invented at Stanford in 1998") score 0.13–0.40 either way and are still
     refused, so this widens the grounded/ungrounded gap rather than lowering the bar. Citation markers
     (`[passage_id: X]`) are stripped before embedding — this pipeline's own prompt asks for them, and they
     dilute short sentences.
   - **Unsupported sentences are dropped, not fatal.** The old rule discarded the whole answer above 30%
     unsupported, which was degenerate for short answers: at three sentences it tolerated *zero*
     (1/3 = 0.33 > 0.3), so one connective sentence threw away two grounded ones and the user was told the
     dataset had no answer. Now the unsupported sentences are removed and the rest returned — a stricter
     guarantee on what actually reaches the user, and the same rule `run_streaming` already applied per
     sentence. Above `max_drift_ratio` (50%) the answer is still refused outright, since returning a small
     remainder would mislead by omission. *Tradeoff:* dropping a sentence can leave the remainder reading
     abruptly.

## Full-request latency tracing (`src/latency.py`, `benchmarks/run_latency_trace.py`)

Every request carries a `RequestTrace` whose `total_ms` is **measured independently** — wall clock from
ASGI entry to response flush, owned by `LatencyTraceMiddleware` — with `unaccounted_ms` as the residual
`total_ms - sum(spans)`.

This inverts the earlier breakdown, which defined `total_ms` as the sum of the stages it happened to
measure. That arithmetic cannot surface missing time even in principle: an untraced stage just shrinks the
reported total to match. In practice it was hiding ~958ms per voice request — mostly the Sarvam STT round
trip, which was timed all along and then omitted from the log line while still counting into the total.
Now unattributed time has nowhere to hide, and `unaccounted_ms` near zero is the invariant that says the
trace is honest.

Spans are flat and non-overlapping. Overlapping aggregates (`llm_ttft_ms`, which spans `llm_network` +
`llm_client_wait`) are recorded as *details* instead, so they can't double-count into the residual.

```
middleware · body_parse · stt_network · cache_lookup · query_preprocessing
embedding_cache · embedding_compute · vector_search · reranking · retrieval_overhead
relevance_guard · context_build · llm_network · llm_client_wait · llm_generation
llm_retry_wait · grounding_guard · serialization · response_write
unaccounted · total
```

There is no `auth_ms` or `db_ms`: this service has no authentication and no database. Reporting a
fabricated `0.0` for stages that don't exist would be worse than their absence.

```bash
venv/bin/python benchmarks/run_latency_trace.py            # text path: 1 cold, 20 warm, 20 cached
venv/bin/python benchmarks/run_latency_trace.py --voice    # audio path (measures the STT round trip)
venv/bin/python benchmarks/run_concurrency.py               # text route at concurrency 1/5/10/25
```

Writes [`benchmarks/latency_trace_text.md`](benchmarks/latency_trace_text.md) and
[`benchmarks/latency_trace_voice.md`](benchmarks/latency_trace_voice.md). Both drive the real ASGI app, so
middleware, multipart parsing, serialization, and flush are all inside the numbers.

**Two things the benchmark controls for, because both silently corrupt latency numbers:**

- *Provider rate limits.* Groq's free tier caps **tokens**, not requests (measured 8000 TPM against ~850
  tokens/request ≈ 9 requests/min). Unpaced, 20 back-to-back requests trip it and the 429 retry backoff is
  measured as pipeline latency: one such run reported a warm p95 of 3933ms, of which 3220ms was backoff.
  `--delay` (default 7s) paces requests; the backoff that does occur is now attributed to
  `llm_retry_wait` rather than left in the residual.
- *CPU frequency state.* That same pacing inflates every CPU-bound span. A ~5ms forward pass is too small
  a burst to make an idled CPU ramp back up, so the identical encode measures **4.8ms back-to-back and
  22ms after a 7s gap**. Neither is wrong; they bound different traffic shapes. The report measures
  embedding under both conditions rather than picking the flattering one.

### Meeting a sub-200ms end-to-end budget

A **remote LLM cannot meet a 200ms end-to-end budget**, and no amount of local optimization changes that.
Measured time-to-first-token on the fastest reachable hosted model is 450–680ms — the budget is gone
before a single token arrives. That is a property of the provider, not of this code.

What does meet it is `LLM_PROVIDER=local` (`ExtractiveProvider`): a zero-network answerer that selects the
best-supported sentences from the retrieved passages and cites them. Measured through the real ASGI stack,
20 warm uncached queries drawn from the served corpus:

| path | p50 | p95 | p99 | max | over 200ms |
|---|---|---|---|---|---|
| uncached, `LLM_PROVIDER=local` | **28.0 ms** | 41.7 ms | 50.2 ms | 52.3 ms | 0/20 |
| cached | 0.2 ms | 0.3 ms | — | — | 0/20 |
| uncached, `LLM_PROVIDER=groq` | 802 ms | 1238 ms | 1368 ms | — | 20/20 |

Over 120 queries taken across `hi_validation_500.jsonl`, the local path answered **119/120 (99%)** with
`total_ms` p50 14.8 / p95 29.6 / max 36.2, and **0 requests over 200ms**. The single failure is one record
whose Hindi query field is corrupt — the machine translation of "suit definition" degenerated into the same
clause repeated ~200 times, and `InputGuardrail` correctly rejects it as repetitive. Its English form
answers normally.

**The tradeoff is answer style, and it is real.** The local provider *extracts*; it does not *synthesize*.

- Best case it is indistinguishable from the gold answer: "what is a corporation?" returns *"A corporation
  is a company or group of people authorized to act as a single entity (legally a person) and recognized as
  such in law. [passage_id: 5]"* — verbatim the dataset's own answer.
- Worse case it leads with an off-target sentence from the right passage: the Hindi form of the same
  question opens on a sentence about McDonald's Corporation before giving the correct definition, because
  sentence choice is query-term overlap rather than comprehension.
- It reads as quoted passage text, not fluent prose, and inherits the passage's punctuation.

So: `local` for the latency target, `groq` for fluency. Retrieval accuracy bounds both — see the hit-rate
figures in `src/retrieval.py`; roughly one answerable query in four does not have its labelled passage in
context even at `top_n=10`.

`bm25_ms` (p50 16.9, p95 29.3) is now the single largest stage on the local path — about 60% of the budget.
It is the obvious next target if more headroom is wanted; nothing here needed it to hit 200ms.

### Latency-relevant configuration

| Variable | Default | Why |
|---|---|---|
| `EMBEDDING_DEVICE` | `cpu` | MPS is ~1.6x faster for the startup index build but far worse per query — measured p50 115ms / p95 226ms for a query arriving after idle, vs 26ms / 35ms on CPU, because each new padded input shape compiles a fresh Metal kernel. CPU and MPS vectors agree to cosine 1.000000, so switching does not invalidate a persisted index. |
| `EMBEDDING_TORCH_THREADS` | `1` | Required, not tuning: `faiss-cpu` bundles its own `libomp.dylib` and multi-threaded CPU torch alongside it **SIGSEGVs** on macOS/arm64. Also costs nothing — p50 is within noise across 1–8 threads at batch size 1. |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | The previous default `llama-3.1-8b-instant` is decommissioned (404). |
| `TRACE_BUFFER_SIZE` | `0` (off) | Ring buffer of completed traces at `GET /metrics/traces`, including the post-handler spans a response body cannot report about itself. |
| `LLM_MAX_TOKENS` | `2000` | The default provider is a *reasoning* model whose hidden reasoning is billed against this budget; too low a cap is spent thinking and emits no answer at all. Latency-neutral (the model stops when done). |
| `LLM_PROVIDER` | auto (`groq` if a key is set, else `local`) | `local` is the only setting that meets a sub-200ms end-to-end budget — see above. `groq`/`anthropic` synthesize more fluent answers at ~800ms+. |

### Which corpus is served (`CORPUS`)

| Value | Serves | Use when |
|---|---|---|
| `msmarco` (default) | ai4bharat/MSMARCO-XI only | Measuring retrieval quality — this is what `benchmarks/run_eval.py` scores against. |
| `demo` | the bundled 2-document corpus | Asking about RAG / this pipeline itself. Too small for meaningful metrics. |
| `both` | demo documents indexed alongside MSMARCO-XI | Demoing. Costs two extra chunks and answers both kinds of question. |

This matters more than it looks. MSMARCO-XI is real search-engine queries about arbitrary topics — hotels in
Smithville NJ, how caffeine is metabolised. Asking "what is retrieval augmented generation" against it is
**correctly** refused by `RelevanceGuardrail`, because the answer genuinely isn't in the corpus. That refusal
looks identical to a bug. Use `CORPUS=both` (or `demo`) to ask questions about the system itself.

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
# Quick labeled evaluation using 100 source rows (the minimum requested size)
EVAL_LIMIT=100 venv/bin/python benchmarks/run_eval.py
```

## API reference

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"status": "ok"}` — frontend checks this before allowing recording |
| `POST /query` | multipart field `audio` → runs `PipelineHarness.run`, returns `PipelineResult` JSON |
| `POST /query/text` | `{"query": "..."}` → same pipeline, skipping STT. The RAG SLA path |
| `POST /query/stream` | multipart field `audio` → Server-Sent Events, one grounding-checked sentence at a time |
| `GET /metrics/cache` | embedding- and answer-cache hit/miss counters |
| `GET /metrics/traces` | recent completed request traces (needs `TRACE_BUFFER_SIZE`) |
| `GET /ready` | readiness check for the indexed corpus and initialized LLM provider |

```bash
curl -X POST http://localhost:8000/query -F "audio=@question.wav"
curl -X POST http://localhost:8000/query/text -H 'content-type: application/json' \
     -d '{"query":"what is retrieval augmented generation"}'
```

`POST /query` and `/query/stream` cannot be brought under 200ms: transcription is a round trip to Sarvam
(measured p50 268ms / p95 453ms on a 1-second clip) and the query isn't known until it completes, so even
an answer-cache hit costs the full STT time — the measured cached *voice* path is p50 230ms. Only
`/query/text` can skip it.

## Testing

```bash
venv/bin/python -m pytest tests/ -q
```

## Backend Docker deployment

Only the FastAPI backend is containerized. The Next.js frontend remains a separate Render service.
Docker loads the prebuilt RAG index once at startup; S3/MinIO is never used in the query hot path.

Build the artifact set on an indexing machine, then upload all five files under one S3-compatible
prefix (`manifest.json`, `faiss.index`, `bm25.pkl`, `chunks.json`, and `metadata.json`):

```bash
python scripts/build_rag_artifacts.py --output-dir artifacts/hi-v1 --version v1 --limit 500
```

Configure the backend with `RAG_ARTIFACT_BUCKET`, `RAG_ARTIFACT_PREFIX`, and optional
`RAG_ARTIFACT_VERSION`, plus `S3_ENDPOINT_URL`, `S3_REGION`, `S3_ACCESS_KEY`, and `S3_SECRET_KEY`.
For AWS, leave `S3_ENDPOINT_URL` empty and use the provider's normal credentials. For MinIO, set the
endpoint to its reachable URL. The service reports `/health` when the process is alive and `/ready`
only after the artifact index and generation provider are usable.

```bash
docker build -t rag-backend:local .
docker run --rm -p 8000:8000 --env-file .env rag-backend:local
```

Render uses the root `Dockerfile`, binds to its supplied `PORT`, and health-checks `/health`. Docker
must be installed and its daemon running before the local build commands can be executed.

## Project layout

```
voice-rag/
  src/
    api.py              # FastAPI app + LatencyTraceMiddleware (owns total_ms)
    latency.py          # RequestTrace — flat spans, measured total, unaccounted residual
    harness.py          # PipelineHarness — orchestrates every stage below
    stt.py               # SarvamSTT + MockSTT
    chunking.py          # 4 chunking strategies + ChunkerRegistry
    vectorstore.py       # Embedder (multilingual) + FAISS VectorStore
    retrieval.py         # Retriever — rerank + low-confidence query rewrite
    guardrails.py        # InputGuardrail / RelevanceGuardrail / GroundingGuardrail
    generation.py        # Generator — swappable GroqProvider / AnthropicProvider
    data_loader.py       # ai4bharat/MSMARCO-XI loader — the corpus the API serves
    text.py              # shared sentence-boundary detection (Latin + Indic terminators)
    pipeline.py          # skeleton, superseded by api.py + harness.py — not implemented
    config.py            # skeleton, not implemented
  frontend/              # Next.js (App Router, TS) — single-page recorder UI
  data/
    msmarco_xi_cache/    # generated: cached dataset slices (gitignored)
    eval_set.json        # generated: labeled retrieval cases from is_selected
    sample_corpus.json   # small demo corpus — offline fallback only
  scripts/               # inspect_dataset.py, compare_chunkers.py — dev tools
  benchmarks/            # run_benchmark.py (latency) + run_eval.py (retrieval accuracy)
  tests/
```
