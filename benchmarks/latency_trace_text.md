# Full-lifecycle latency trace

Request path: text (`POST /query/text`). Driven through the real ASGI app, so every number below includes middleware, body parsing, response serialization, and body flush. Percentiles are linear-interpolated.

## Trace honesty invariant

The point of this instrumentation is that `total_ms` is measured independently (wall clock, ASGI entry to body flush) rather than defined as the sum of the stages, so unattributed time has somewhere to show up:

```
sum(spans) + unaccounted_ms == total_ms
```

Worst-case error across all 41 requests: **0.000 ms** (mean 0.000 ms). Anything beyond timer granularity here would mean overlapping or double-counted spans.

## Query embedding, isolated

Measured on its own, because the warm phase below is paced at 8s per request to stay inside the provider's tokens-per-minute cap, and that pacing inflates every CPU-bound span: an idle CPU drops to a low-power state and a ~5ms forward pass is too small a burst to make it ramp back up. Both bounds are real; which one a deployment sees depends on its traffic.

Device: `cpu`.

| condition | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| continuous load (back-to-back) | 4.9 | 5.4 | 5.4 | 12 |
| after 8s idle (as paced below) | 24.7 | 31.7 | 31.9 | 12 |
| cache hit (no forward pass) | 0.0 | 0.0 | 0.0 | 12 |

## Phase 1 — cold (1 request, fresh process, prewarming disabled)

Startup (corpus load + index build, prewarm skipped): **100707 ms**.

| span | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| middleware_ms | 1.0 | 1.0 | 1.0 | 1 |
| cache_lookup_ms | 0.1 | 0.1 | 0.1 | 1 |
| query_preprocessing_ms | 0.0 | 0.0 | 0.0 | 1 |
| embedding_cache_ms | 0.0 | 0.0 | 0.0 | 1 |
| embedding_compute_ms | 5.9 | 5.9 | 5.9 | 1 |
| vector_search_ms | 7.1 | 7.1 | 7.1 | 1 |
| reranking_ms | 0.0 | 0.0 | 0.0 | 1 |
| retrieval_overhead_ms | 0.8 | 0.8 | 0.8 | 1 |
| relevance_guard_ms | 0.0 | 0.0 | 0.0 | 1 |
| context_build_ms | 0.0 | 0.0 | 0.0 | 1 |
| llm_network_ms | 502.8 | 502.8 | 502.8 | 1 |
| llm_client_wait_ms | 527.4 | 527.4 | 527.4 | 1 |
| llm_generation_ms | 4.2 | 4.2 | 4.2 | 1 |
| llm_retry_wait_ms | 0.1 | 0.1 | 0.1 | 1 |
| grounding_guard_ms | 19.3 | 19.3 | 19.3 | 1 |
| serialization_ms | 0.7 | 0.7 | 0.7 | 1 |
| response_write_ms | 0.0 | 0.0 | 0.0 | 1 |
| **unaccounted_ms** | 0.9 | 0.9 | 0.9 | 1 |
| **total_ms** | 1070.4 | 1070.4 | 1070.4 | 1 |

All 1 cold requests returned a normal (non-degraded) answer.

## Phase 2 — warm uncached (20 distinct queries)

Startup including prewarm: **99182 ms** (the cold-start costs the first request used to pay).

### Required metrics

| metric | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| `embedding_ms` | 19.7 | 25.6 | 26.4 | 20 |
| `retrieval_ms` | 2.2 | 6.9 | 7.5 | 20 |
| `llm_ttft_ms` | 677.9 | 1061.3 | 1087.9 | 19 |
| `llm_total_ms` | 763.6 | 1215.2 | 1300.1 | 20 |
| `total_ms` | 802.1 | 1237.7 | 1368.3 | 20 |

### Full stage breakdown

| span | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| middleware_ms | 0.6 | 2.2 | 2.2 | 20 |
| cache_lookup_ms | 0.1 | 0.3 | 0.6 | 20 |
| query_preprocessing_ms | 0.1 | 0.2 | 0.3 | 20 |
| embedding_cache_ms | 0.0 | 0.1 | 0.1 | 20 |
| embedding_compute_ms | 19.7 | 25.6 | 26.4 | 20 |
| vector_search_ms | 1.5 | 5.9 | 6.1 | 20 |
| reranking_ms | 0.0 | 0.1 | 0.1 | 20 |
| retrieval_overhead_ms | 0.5 | 1.2 | 4.7 | 20 |
| relevance_guard_ms | 0.0 | 0.0 | 0.0 | 20 |
| context_build_ms | 0.0 | 0.1 | 0.1 | 20 |
| llm_network_ms | 384.8 | 662.9 | 746.0 | 20 |
| llm_client_wait_ms | 287.4 | 703.6 | 786.4 | 20 |
| llm_generation_ms | 52.1 | 288.9 | 298.7 | 19 |
| llm_retry_wait_ms | 0.0 | 0.0 | 0.0 | 20 |
| grounding_guard_ms | 19.5 | 37.9 | 51.4 | 20 |
| serialization_ms | 0.2 | 0.3 | 0.3 | 20 |
| response_write_ms | 0.0 | 0.0 | 0.0 | 20 |
| **unaccounted_ms** | 1.1 | 2.1 | 2.4 | 20 |
| **total_ms** | 802.1 | 1237.7 | 1368.3 | 20 |

All 20 warm uncached requests returned a normal (non-degraded) answer.

## Phase 3 — cached (20 repeats of one query)

### Required metrics

| metric | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| `embedding_ms` | n/a | n/a | n/a | 0 |
| `retrieval_ms` | n/a | n/a | n/a | 0 |
| `llm_ttft_ms` | n/a | n/a | n/a | 0 |
| `llm_total_ms` | n/a | n/a | n/a | 0 |
| `total_ms` | 0.2 | 0.4 | 0.6 | 20 |

### Full stage breakdown

| span | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| middleware_ms | 0.1 | 0.2 | 0.5 | 20 |
| cache_lookup_ms | 0.0 | 0.0 | 0.0 | 20 |
| serialization_ms | 0.0 | 0.0 | 0.1 | 20 |
| response_write_ms | 0.0 | 0.0 | 0.0 | 20 |
| **unaccounted_ms** | 0.1 | 0.1 | 0.1 | 20 |
| **total_ms** | 0.2 | 0.4 | 0.6 | 20 |

Cache hits: **20/20**.

## Verdict against the 200ms target

| path | p50 | p95 | < 200ms? |
| --- | --- | --- | --- |
| warm uncached | 802.1 ms | 1237.7 ms | **no** |
| cached | 0.2 ms | 0.4 ms | **yes** |

