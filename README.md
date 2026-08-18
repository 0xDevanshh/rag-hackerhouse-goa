# voice-rag

A voice-driven Retrieval-Augmented Generation (RAG) pipeline: speech in, transcribed and
grounded in a document corpus, answered through an LLM with guardrails in the loop.

## Pipeline overview

1. **STT** (`src/stt.py`) — transcribe spoken audio to text (Sarvam API).
2. **Chunking** (`src/chunking.py`) — split source documents into retrievable chunks.
3. **Vector store** (`src/vectorstore.py`) — embed and index chunks (FAISS) for similarity search.
4. **Retrieval** (`src/retrieval.py`) — fetch the most relevant chunks for a query.
5. **Guardrails** (`src/guardrails.py`) — validate/filter input and output for safety and relevance.
6. **Generation** (`src/generation.py`) — call an LLM (Anthropic) to produce a grounded answer.
7. **Pipeline** (`src/pipeline.py`) — orchestrate the end-to-end flow above.
8. **Harness** (`src/harness.py`) — test/evaluation harness for running the pipeline over benchmarks.

Configuration lives in `src/config.py` and is loaded from environment variables (see `.env.example`).

## Project layout

```
voice-rag/
  src/            # application source code
  data/           # sample data (e.g. sample_corpus.json)
  frontend/       # frontend client (TBD)
  benchmarks/     # evaluation benchmarks/datasets
  tests/          # unit/integration tests
  requirements.txt
  .env.example
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in SARVAM_API_KEY and ANTHROPIC_API_KEY
```

## Status

Skeleton only — module interfaces are defined but not yet implemented.
