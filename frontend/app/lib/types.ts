// ─── Backend contract types ────────────────────────────────────────────────
// Mirrors src/harness.py PipelineResult and src/latency.py RequestTrace exactly.
// Never spread raw backend payloads into React components — use the normalized
// FrontendResult model instead (see normalize.ts).

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

export interface Span {
  name: string;
  started_at: number;
  ended_at: number;
  duration_ms: number;
}

export interface RequestTrace {
  spans: Span[];
  details: Record<string, number>;
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
  cached: boolean;
  errors: StageError[];
}

// ─── UI state machine ───────────────────────────────────────────────────────

export type UiPhase =
  | "IDLE"
  | "RECORDING"
  | "UPLOADING"
  | "TRANSCRIBING"
  | "RETRIEVING"
  | "GROUNDING"
  | "GENERATING"
  | "COMPLETE"
  | "DEGRADED"
  | "ERROR"
  | "OFFLINE_DEMO";

export type BackendStatus = "CHECKING" | "ONLINE" | "CONNECTING" | "OFFLINE" | "PROCESSING";

export type InputMode = "VOICE" | "TEXT";

// ─── Normalized frontend model ──────────────────────────────────────────────

export interface LatencyBreakdown {
  totalMs: number | null;
  ragCoreMs: number | null;
  sttMs: number | null;
  embeddingMs: number | null;
  retrievalMs: number | null;
  bm25Ms: number | null;
  fusionMs: number | null;
  rerankMs: number | null;
  generationMs: number | null;
  groundingMs: number | null;
  guardrailsMs: number | null;
  unaccountedMs: number | null;
  overheadMs: number | null;
  llmTtftMs: number | null;
  // Overlap-specific
  sttToFirstPartialMs: number | null;
  sttFinalMs: number | null;
  retrievalOnPartialMs: number | null;
  sttOverlapSavingsMs: number | null;
  // Path info
  isOverlappedPath: boolean;
  isCached: boolean;
}

export interface EvidenceCard {
  index: number;
  text: string;
  score: number;
  language: string | null;
  passageId: string | null;
  queryId: string | null;
  chunkId: string | null;
  documentId: string | null;
  strategyName: string;
  isSelected: boolean;
}

export interface GuardrailStatus {
  name: string;
  label: string;
  description: string;
  status: "PASSED" | "BLOCKED" | "DEGRADED" | "UNKNOWN";
  reason: string | null;
}

export interface GroundingInfo {
  status: "GROUNDED" | "DEGRADED" | "REFUSED" | "UNKNOWN";
  score: number | null;
  reason: string | null;
}

export interface FrontendResult {
  query: string;
  answer: string;
  inputMode: InputMode;
  latency: LatencyBreakdown;
  evidence: EvidenceCard[];
  guardrails: GuardrailStatus[];
  grounding: GroundingInfo;
  isDegraded: boolean;
  isCached: boolean;
  errors: StageError[];
  ragTargetMs: number;
}

// ─── History item ───────────────────────────────────────────────────────────

export interface HistoryItem {
  id: string;
  timestamp: number;
  query: string;
  mode: InputMode;
  status: "COMPLETE" | "DEGRADED" | "ERROR";
  totalMs: number | null;
  result: FrontendResult | null;
}

// ─── Pipeline stage visualization ───────────────────────────────────────────

export type PipelineStageStatus = "PENDING" | "ACTIVE" | "COMPLETE" | "FAILED" | "SKIPPED";

export interface PipelineStage {
  id: string;
  label: string;
  shortLabel: string;
  status: PipelineStageStatus;
  durationMs: number | null;
}
