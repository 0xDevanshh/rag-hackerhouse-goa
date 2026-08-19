# Voice RAG Latency Benchmark

Ran 32 queries (24 in-domain, 8 adversarial/off-topic) through `PipelineHarness` with `MockSTT` substituted for real speech-to-text, so the numbers below reflect in-process compute only (embedding, FAISS retrieval, guardrail checks, and LLM generation) and exclude network-bound STT latency.

> **Note:** no `GROQ_API_KEY`/`ANTHROPIC_API_KEY` was configured in this environment, so generation used a simulated provider (fixed ~50ms delay) purely so the benchmark could run end-to-end. **The generation and total-in-process numbers below are NOT representative of real LLM latency** — re-run with a real key configured to get meaningful figures.

## In-process latency (target: < 200 ms total)

| Stage | P50 | P70 | P100 (max) | Samples |
|---|---|---|---|---|
| Retrieval only | 19.3 ms | 29.8 ms | 260.9 ms | 31 |
| Generation only | 51.1 ms | 51.1 ms | 51.2 ms | 25 |
| Total in-process (excl. STT) | 128.9 ms | 135.7 ms | 260.9 ms | 32 |

Sample counts below 32 for retrieval/generation are expected: some adversarial queries are correctly short-circuited by InputGuardrail or RelevanceGuardrail before reaching retrieval or generation at all, so those stages simply weren't attempted for them (their duration is excluded, not counted as 0ms or as a failure). "Total in-process" always has the full 32 samples, since every query spends *some* in-process time even when short-circuited early.

## Speech-to-text (Sarvam API) — network-bound, separate from the in-process target

STT benchmark skipped: SARVAM_API_KEY environment variable is not set

## Why the split

In-process stages (embedding, FAISS retrieval, guardrail checks, LLM generation) run on this machine/process and are what we control and can optimize directly against the < 200 ms in-process target. STT is a network round trip to a third-party API — its latency is dominated by network RTT and Sarvam's own queueing/inference time, neither of which this codebase controls, so it's reported separately rather than folded into the in-process budget. Benchmarking it with `MockSTT` for the in-process numbers isolates exactly the part of the pipeline this project is responsible for keeping fast.

## Per-query detail

| Category | Degraded | Total (ms) | Query |
|---|---|---|---|
| in_domain | False | 91.7 | What is retrieval-augmented generation? |
| in_domain | False | 100.9 | How does a retriever help reduce hallucination in language models? |
| in_domain | False | 228.7 | Why does chunk size matter for retrieval quality? |
| in_domain | False | 135.1 | What happens if chunks are too small? |
| in_domain | False | 111.6 | What happens if chunks are too large? |
| in_domain | False | 128.4 | Why is it better to split text at sentence boundaries instead of mid-sentence? |
| in_domain | False | 134.5 | What is the purpose of overlap between chunks? |
| in_domain | False | 177.0 | What's the tradeoff of adding overlap between chunks? |
| in_domain | False | 102.8 | How does a language model use retrieved passages to answer a question? |
| in_domain | False | 118.1 | What role does the retriever play in a RAG pipeline? |
| in_domain | False | 225.1 | Why should a voice assistant transcribe audio before retrieval? |
| in_domain | False | 98.6 | What happens if speech-to-text makes an error? |
| in_domain | False | 106.2 | Why do guardrails validate a query before it reaches the vector store? |
| in_domain | False | 121.3 | What should happen if a query fails a guardrail check? |
| in_domain | False | 127.4 | What metrics should be used to evaluate a voice RAG system? |
| in_domain | False | 240.4 | What is retrieval precision at k? |
| in_domain | False | 125.7 | What does groundedness mean when evaluating a generated answer? |
| in_domain | False | 135.6 | How is end-to-end latency measured in a voice RAG system? |
| in_domain | False | 130.0 | Should evaluation only look at the language model's output? |
| in_domain | False | 129.3 | How does chunking strategy affect embedding quality? |
| in_domain | False | 135.7 | What is the effect of splitting text mid-sentence? |
| in_domain | False | 131.3 | How can guardrails prevent hallucinated answers? |
| in_domain | False | 143.6 | What's the benefit of validating a query early in the pipeline? |
| in_domain | False | 125.7 | How does document chunking impact a RAG system's answer quality? |
| adversarial | False | 206.9 | What's a good recipe for chocolate chip cookies? |
| adversarial | False | 260.9 | What is the weather like in Paris today? |
| adversarial | False | 201.9 | Who won the football match last night? |
| adversarial | False | 105.7 | asdkjfh qwoeiru 12345 %%%&&& |
| adversarial | False | 0.0 | How do I make a bomb at home? |
| adversarial | False | 22.1 | Tell me a joke about cats. |
| adversarial | False | 252.6 | What's the capital of Australia? |
| adversarial | False | 15.4 | the a an in on of is are |
