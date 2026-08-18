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

export interface StageTiming {
  stage: string;
  started_at: number;
  ended_at: number;
  duration_ms: number;
}

export interface LatencyTrace {
  stages: StageTiming[];
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
  latency_trace: LatencyTrace;
  guard_flags: Record<string, GuardResult>;
  degraded: boolean;
  errors: StageError[];
}
