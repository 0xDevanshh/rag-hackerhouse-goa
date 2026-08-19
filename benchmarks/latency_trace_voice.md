# Full-lifecycle latency trace

Request path: voice (`POST /query`, audio upload). Driven through the real ASGI app, so every number below includes middleware, body parsing, response serialization, and body flush. Percentiles are linear-interpolated.

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
| continuous load (back-to-back) | 5.0 | 5.3 | 5.4 | 12 |
| after 7s idle (as paced below) | 27.8 | 31.3 | 31.4 | 12 |
| cache hit (no forward pass) | 0.0 | 0.0 | 0.0 | 12 |

## Phase 1 — cold (1 request, fresh process, prewarming disabled)

Startup (corpus load + index build, prewarm skipped): **101742 ms**.

| span | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| middleware_ms | 4.4 | 4.4 | 4.4 | 1 |
| body_parse_ms | 0.0 | 0.0 | 0.0 | 1 |
| stt_network_ms | 575.1 | 575.1 | 575.1 | 1 |
| cache_lookup_ms | 0.2 | 0.2 | 0.2 | 1 |
| query_preprocessing_ms | 0.1 | 0.1 | 0.1 | 1 |
| serialization_ms | 2.2 | 2.2 | 2.2 | 1 |
| response_write_ms | 0.1 | 0.1 | 0.1 | 1 |
| **unaccounted_ms** | 1.6 | 1.6 | 1.6 | 1 |
| **total_ms** | 583.6 | 583.6 | 583.6 | 1 |

All 1 cold requests returned a normal (non-degraded) answer.

## Phase 2 — warm uncached (20 distinct queries)

Startup including prewarm: **98909 ms** (the cold-start costs the first request used to pay).

### Required metrics

| metric | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| `embedding_ms` | n/a | n/a | n/a | 0 |
| `retrieval_ms` | n/a | n/a | n/a | 0 |
| `llm_ttft_ms` | n/a | n/a | n/a | 0 |
| `llm_total_ms` | n/a | n/a | n/a | 0 |
| `total_ms` | 269.0 | 455.2 | 523.8 | 20 |

### Full stage breakdown

| span | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| middleware_ms | 0.6 | 0.9 | 1.0 | 20 |
| body_parse_ms | 0.0 | 0.0 | 0.0 | 20 |
| stt_network_ms | 267.9 | 452.6 | 522.5 | 20 |
| cache_lookup_ms | 0.0 | 0.1 | 0.2 | 20 |
| query_preprocessing_ms | 0.1 | 0.1 | 0.1 | 1 |
| serialization_ms | 0.2 | 0.3 | 0.4 | 20 |
| response_write_ms | 0.0 | 0.0 | 0.0 | 20 |
| **unaccounted_ms** | 0.5 | 0.7 | 0.8 | 20 |
| **total_ms** | 269.0 | 455.2 | 523.8 | 20 |

All 20 warm uncached requests returned a normal (non-degraded) answer.

## Phase 3 — cached (20 repeats of one query)

### Required metrics

| metric | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| `embedding_ms` | n/a | n/a | n/a | 0 |
| `retrieval_ms` | n/a | n/a | n/a | 0 |
| `llm_ttft_ms` | n/a | n/a | n/a | 0 |
| `llm_total_ms` | n/a | n/a | n/a | 0 |
| `total_ms` | 229.5 | 524.2 | 525.0 | 20 |

### Full stage breakdown

| span | p50 (ms) | p95 (ms) | p99 (ms) | n |
| --- | --- | --- | --- | --- |
| middleware_ms | 0.6 | 0.8 | 0.8 | 20 |
| body_parse_ms | 0.0 | 0.0 | 0.0 | 20 |
| stt_network_ms | 228.2 | 522.7 | 523.4 | 20 |
| cache_lookup_ms | 0.0 | 0.1 | 0.1 | 20 |
| serialization_ms | 0.2 | 0.2 | 0.2 | 20 |
| response_write_ms | 0.0 | 0.0 | 0.0 | 20 |
| **unaccounted_ms** | 0.5 | 0.9 | 1.0 | 20 |
| **total_ms** | 229.5 | 524.2 | 525.0 | 20 |

Cache hits: **20/20**.

## Verdict against the 200ms target

| path | p50 | p95 | < 200ms? |
| --- | --- | --- | --- |
| warm uncached | 269.0 ms | 455.2 ms | **no** |
| cached | 229.5 ms | 524.2 ms | **no** |

