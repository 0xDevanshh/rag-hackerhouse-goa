// Mirrors the pydantic models returned by the FastAPI backend (src/harness.py).

export interface Chunk {
  text: string;
  metadata: Record<string, unknown>;
  strategy_name: string;
}

export interface GuardResult {
  allowed: boolean;
  reason: string;
  response_override: string | null;
}

/** One measured, non-overlapping slice of a request's wall clock. */
export interface Span {
  name: string;
  started_at: number;
  ended_at: number;
  duration_ms: number;
}

/**
 * Mirrors src/latency.py RequestTrace. Replaces the former LatencyTrace, whose
 * `stages` array this used to sum to get a total.
 *
 * Do NOT sum `spans` to compute a total. `total_ms` is measured independently
 * on the server (wall clock from ASGI entry to response flush) and is sent
 * here for exactly that reason: the span sum omits any time that wasn't
 * instrumented, which is how ~958ms per request went unnoticed. `unaccounted_ms`
 * is the difference, and should be close to zero.
 */
export interface RequestTrace {
  spans: Span[];
  /** Overlapping aggregates (e.g. `llm_ttft`, `llm_total`) — excluded from the span sum on purpose. */
  details: Record<string, number>;
  /** Non-timing facts: cache hit/miss, retry count, provider finish reason. */
  labels: Record<string, string | number | boolean>;
  total_ms: number;
  unaccounted_ms: number;
}

export interface StageError {
  stage: string;
  error_type: string;
  message: string;
}

export interface PipelineResult {
  answer: string;
  query_text: string;
  sources: Chunk[];
  scores: number[];
  trace: RequestTrace;
  guard_flags: Record<string, GuardResult>;
  degraded: boolean;
  /** True when the answer came from the answer cache instead of a fresh retrieval + generation. */
  cached: boolean;
  errors: StageError[];
}
