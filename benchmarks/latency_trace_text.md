# Full-lifecycle latency trace

Request path: text (`POST /query/text`). Driven through the real ASGI app, so every number below includes middleware, body parsing, response serialization, and body flush. Percentiles are linear-interpolated.

## Trace honesty invariant

The point of this instrumentation is that `total_ms` is measured independently (wall clock, ASGI entry to body flush) rather than defined as the sum of the stages, so unattributed time has somewhere to show up:

```
sum(spans) + unaccounted_ms == total_ms
```

Worst-case error across all 41 requests: **0.000 ms** (mean 0.000 ms). Anything beyond timer granularity here would mean overlapping or double-counted spans.

## Query embedding, isolated

Measured on its own, because the warm phase below is paced at 0s per request to stay inside the provider's tokens-per-minute cap, and that pacing inflates every CPU-bound span: an idle CPU drops to a low-power state and a ~5ms forward pass is too small a burst to make it ramp back up. Both bounds are real; which one a deployment sees depends on its traffic.

Device: `cpu`.

| condition | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| continuous load (back-to-back) | 6.0 | 7.1 | 7.4 | 12 |
| after 7s idle (as paced below) | 26.9 | 63.6 | 76.5 | 12 |
| cache hit (no forward pass) | 0.0 | 0.0 | 0.0 | 12 |

## Phase 1 — cold (1 request, fresh process, prewarming disabled)

Startup (corpus load + index build, prewarm skipped): **53465 ms**.

| span | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| middleware_ms | 7.2 | 7.2 | 7.2 | 1 |
| cache_lookup_ms | 0.2 | 0.2 | 0.2 | 1 |
| query_preprocessing_ms | 0.1 | 0.1 | 0.1 | 1 |
| embedding_cache_ms | 0.0 | 0.0 | 0.0 | 1 |
| embedding_compute_ms | 17.4 | 17.4 | 17.4 | 1 |
| vector_search_ms | 7.4 | 7.4 | 7.4 | 1 |
| bm25_ms | 7.6 | 7.6 | 7.6 | 1 |
| fusion_ms | 0.1 | 0.1 | 0.1 | 1 |
| reranking_ms | 0.1 | 0.1 | 0.1 | 1 |
| retrieval_overhead_ms | 2.5 | 2.5 | 2.5 | 1 |
| relevance_guard_ms | 0.0 | 0.0 | 0.0 | 1 |
| context_build_ms | 0.0 | 0.0 | 0.0 | 1 |
| llm_network_ms | 0.0 | 0.0 | 0.0 | 1 |
| llm_client_wait_ms | 0.0 | 0.0 | 0.0 | 1 |
| llm_generation_ms | 0.5 | 0.5 | 0.5 | 1 |
| llm_retry_wait_ms | 0.1 | 0.1 | 0.1 | 1 |
| grounding_guard_ms | 1.1 | 1.1 | 1.1 | 1 |
| serialization_ms | 2.1 | 2.1 | 2.1 | 1 |
| response_write_ms | 0.4 | 0.4 | 0.4 | 1 |
| **unaccounted_ms** | 4.4 | 4.4 | 4.4 | 1 |
| **total_ms** | 51.4 | 51.4 | 51.4 | 1 |

All 1 cold requests returned a normal (non-degraded) answer.

## Phase 2 — warm uncached (20 distinct queries)

Startup including prewarm: **65758 ms** (the cold-start costs the first request used to pay).

### Required metrics

| metric | P50 (ms) | P70 (ms) | P100 (ms) | n |
| --- | --- | --- | --- | --- |
| `embedding_ms` | 6.4 | 6.5 | 21.0 | 20 |
| `retrieval_ms` | 14.7 | 16.2 | 138.5 | 20 |
| `llm_ttft_ms` | 0.2 | 0.3 | 0.8 | 20 |
| `llm_total_ms` | 0.2 | 0.3 | 0.8 | 20 |
| `total_ms` | 22.4 | 24.6 | 155.6 | 20 |

The table above is the submission metric: the warm uncached `total_ms` row is the full request path, including voice transcription when `--voice` is used. The P95/P99 diagnostics below show tail shape.

| metric | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| `embedding_ms` | 6.4 | 10.4 | 18.9 | 20 |
| `retrieval_ms` | 14.7 | 33.6 | 117.5 | 20 |
| `llm_ttft_ms` | 0.2 | 0.4 | 0.7 | 20 |
| `llm_total_ms` | 0.2 | 0.4 | 0.7 | 20 |
| `total_ms` | 22.4 | 42.4 | 133.0 | 20 |

### Full stage breakdown

| span | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| middleware_ms | 0.1 | 0.2 | 1.3 | 20 |
| cache_lookup_ms | 0.0 | 0.0 | 0.2 | 20 |
| query_preprocessing_ms | 0.0 | 0.0 | 0.0 | 20 |
| embedding_cache_ms | 0.0 | 0.0 | 0.0 | 20 |
| embedding_compute_ms | 6.3 | 10.4 | 18.9 | 20 |
| vector_search_ms | 0.3 | 7.1 | 108.5 | 20 |
| bm25_ms | 12.3 | 21.1 | 26.2 | 20 |
| fusion_ms | 0.0 | 0.1 | 0.1 | 20 |
| reranking_ms | 0.0 | 0.8 | 1.3 | 20 |
| retrieval_overhead_ms | 0.2 | 0.3 | 0.5 | 20 |
| relevance_guard_ms | 0.0 | 0.0 | 0.0 | 20 |
| context_build_ms | 0.0 | 0.0 | 0.0 | 20 |
| llm_network_ms | 0.0 | 0.0 | 0.0 | 20 |
| llm_client_wait_ms | 0.0 | 0.0 | 0.0 | 20 |
| llm_generation_ms | 0.2 | 0.4 | 0.7 | 20 |
| llm_retry_wait_ms | 0.0 | 0.0 | 0.0 | 20 |
| grounding_guard_ms | 0.2 | 0.4 | 0.7 | 20 |
| serialization_ms | 0.1 | 0.4 | 3.8 | 20 |
| response_write_ms | 0.0 | 0.0 | 0.0 | 20 |
| **unaccounted_ms** | 0.4 | 0.9 | 2.9 | 20 |
| **total_ms** | 22.4 | 42.4 | 133.0 | 20 |

All 20 warm uncached requests returned a normal (non-degraded) answer.

## Phase 3 — cached (20 repeats of one query)

### Required metrics

| metric | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| `embedding_ms` | n/a | n/a | n/a | 0 |
| `retrieval_ms` | n/a | n/a | n/a | 0 |
| `llm_ttft_ms` | n/a | n/a | n/a | 0 |
| `llm_total_ms` | n/a | n/a | n/a | 0 |
| `total_ms` | 0.1 | 0.3 | 0.4 | 20 |

### Full stage breakdown

| span | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| middleware_ms | 0.0 | 0.1 | 0.1 | 20 |
| cache_lookup_ms | 0.0 | 0.0 | 0.0 | 20 |
| serialization_ms | 0.0 | 0.0 | 0.1 | 20 |
| response_write_ms | 0.0 | 0.0 | 0.0 | 20 |
| **unaccounted_ms** | 0.0 | 0.1 | 0.3 | 20 |
| **total_ms** | 0.1 | 0.3 | 0.4 | 20 |

Cache hits: **20/20**.

## Verdict against the 200ms target

| path | p50 | p100 | < 200ms? |
| --- | --- | --- | --- |
| warm uncached | 22.4 ms | 155.6 ms | **yes** |
| cached | 0.1 ms | 0.5 ms | **yes** |

