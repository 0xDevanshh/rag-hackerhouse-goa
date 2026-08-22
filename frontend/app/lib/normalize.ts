// ─── Response normalization ────────────────────────────────────────────────
// One place where raw backend payloads are parsed into the frontend's typed model.
// UI components never touch PipelineResult directly.

import type {
  PipelineResult,
  RequestTrace,
  FrontendResult,
  LatencyBreakdown,
  EvidenceCard,
  GuardrailStatus,
  GroundingInfo,
  InputMode,
} from "./types";

// ── Trace helpers ─────────────────────────────────────────────────────────

function spanTotal(trace: RequestTrace | undefined, ...names: string[]): number | null {
  const spans = trace?.spans;
  if (!Array.isArray(spans)) return null;
  const matched = spans.filter((s) => names.includes(s.name));
  return matched.length ? matched.reduce((sum, s) => sum + s.duration_ms, 0) : null;
}

function traceDetail(trace: RequestTrace | undefined, name: string): number | null {
  const value = trace?.details?.[name];
  return typeof value === "number" ? value : null;
}

function buildLatency(trace: RequestTrace | undefined, cached: boolean): LatencyBreakdown {
  if (!trace) {
    return {
      totalMs: null, ragCoreMs: null, sttMs: null, embeddingMs: null,
      retrievalMs: null, bm25Ms: null, fusionMs: null, rerankMs: null,
      generationMs: null, groundingMs: null, guardrailsMs: null,
      unaccountedMs: null, overheadMs: null, llmTtftMs: null,
      sttToFirstPartialMs: null, sttFinalMs: null,
      retrievalOnPartialMs: null, sttOverlapSavingsMs: null,
      isOverlappedPath: false, isCached: cached,
    };
  }

  const isOverlapped = spanTotal(trace, "stt_final") !== null;

  const embeddingMs = spanTotal(trace, "embedding_cache", "embedding_compute");
  const retrievalCoreMs = spanTotal(trace, "vector_search", "bm25", "fusion", "reranking", "retrieval_overhead");
  const generationMs = spanTotal(trace, "llm_network", "llm_client_wait", "llm_generation", "llm_retry_wait");
  const groundingMs = spanTotal(trace, "grounding_guard");
  const guardrailsMs = spanTotal(trace, "query_preprocessing", "relevance_guard", "grounding_guard");
  const overheadMs = spanTotal(trace, "middleware", "body_parse", "serialization", "response_write");

  // RAG core = everything except STT and server overhead
  const ragCoreMs = spanTotal(
    trace,
    "query_preprocessing",
    "embedding_cache",
    "embedding_compute",
    "vector_search",
    "bm25",
    "fusion",
    "reranking",
    "retrieval_overhead",
    "relevance_guard",
    "context_build",
    "llm_network",
    "llm_client_wait",
    "llm_generation",
    "llm_retry_wait",
    "grounding_guard",
  );

  return {
    totalMs: typeof trace.total_ms === "number" ? trace.total_ms : null,
    ragCoreMs,
    sttMs: isOverlapped
      ? spanTotal(trace, "stt_final")
      : spanTotal(trace, "stt_network"),
    embeddingMs,
    retrievalMs: retrievalCoreMs,
    bm25Ms: spanTotal(trace, "bm25"),
    fusionMs: spanTotal(trace, "fusion"),
    rerankMs: spanTotal(trace, "reranking"),
    generationMs,
    groundingMs,
    guardrailsMs,
    unaccountedMs: typeof trace.unaccounted_ms === "number" ? trace.unaccounted_ms : null,
    overheadMs,
    llmTtftMs: traceDetail(trace, "llm_ttft"),
    sttToFirstPartialMs: spanTotal(trace, "stt_to_first_partial"),
    sttFinalMs: isOverlapped ? spanTotal(trace, "stt_final") : null,
    retrievalOnPartialMs: spanTotal(trace, "retrieval_on_partial"),
    sttOverlapSavingsMs: spanTotal(trace, "stt_overlap_savings"),
    isOverlappedPath: isOverlapped,
    isCached: cached,
  };
}

function buildEvidence(result: PipelineResult): EvidenceCard[] {
  return result.sources.map((chunk, i) => ({
    index: i,
    text: chunk.text,
    score: result.scores[i] ?? 0,
    language: (chunk.metadata.language as string) ?? null,
    passageId: chunk.metadata.passage_id != null ? String(chunk.metadata.passage_id) : null,
    queryId: chunk.metadata.query_id != null ? String(chunk.metadata.query_id) : null,
    chunkId: chunk.metadata.chunk_id != null ? String(chunk.metadata.chunk_id) : null,
    documentId:
      (chunk.metadata.document_id ?? chunk.metadata.doc_id) != null
        ? String(chunk.metadata.document_id ?? chunk.metadata.doc_id)
        : null,
    strategyName: chunk.strategy_name,
    isSelected: Boolean(chunk.metadata.is_selected),
  }));
}

function buildGuardrails(result: PipelineResult): GuardrailStatus[] {
  const GUARD_META: Record<string, { label: string; description: string }> = {
    input: {
      label: "INPUT GUARD",
      description: "Blocks malformed, repetitive, unsafe, or invalid inputs.",
    },
    relevance: {
      label: "RELEVANCE GUARD",
      description: "Ensures retrieved evidence is sufficiently relevant.",
    },
    grounding: {
      label: "GROUNDING GUARD",
      description: "Checks that the final answer is supported by retrieved evidence.",
    },
  };

  const flags = result.guard_flags ?? {};

  return Object.entries(GUARD_META).map(([key, meta]) => {
    const flag = flags[key];
    let status: GuardrailStatus["status"] = "UNKNOWN";
    if (flag) {
      if (flag.allowed) {
        status = "PASSED";
      } else {
        const reason = flag.reason ?? "";
        status = reason.includes("degraded") ? "DEGRADED" : "BLOCKED";
      }
    } else if (result.degraded) {
      status = "DEGRADED";
    } else {
      // If we got a valid answer with no explicit flag, the guard implicitly passed
      status = result.answer && !result.degraded ? "PASSED" : "UNKNOWN";
    }

    return {
      name: key,
      label: meta.label,
      description: meta.description,
      status,
      reason: flag?.reason ?? null,
    };
  });
}

function buildGrounding(result: PipelineResult): GroundingInfo {
  const groundingFlag = result.guard_flags?.grounding;
  const relevanceFlag = result.guard_flags?.relevance;

  if (result.degraded) {
    return { status: "DEGRADED", score: null, reason: "Pipeline encountered errors" };
  }

  if (relevanceFlag && !relevanceFlag.allowed) {
    return { status: "REFUSED", score: null, reason: relevanceFlag.reason };
  }

  if (groundingFlag) {
    if (!groundingFlag.allowed) {
      return { status: "REFUSED", score: null, reason: groundingFlag.reason };
    }
    const reason = groundingFlag.reason ?? "";
    // Compute a rough grounding score from available evidence scores
    const topScore = result.scores[0] ?? null;
    const score = topScore !== null ? Math.min(1, topScore) : null;
    return { status: "GROUNDED", score, reason };
  }

  if (result.sources.length > 0 && result.answer) {
    const topScore = result.scores[0] ?? null;
    return {
      status: "GROUNDED",
      score: topScore !== null ? Math.min(1, topScore) : null,
      reason: "ok",
    };
  }

  return { status: "UNKNOWN", score: null, reason: null };
}

export function normalizeResult(
  result: PipelineResult,
  inputMode: InputMode,
  ragTargetMs: number = 200,
): FrontendResult {
  return {
    query: result.query_text,
    answer: result.answer,
    inputMode,
    latency: buildLatency(result.trace, result.cached),
    evidence: buildEvidence(result),
    guardrails: buildGuardrails(result),
    grounding: buildGrounding(result),
    isDegraded: result.degraded,
    isCached: result.cached,
    errors: result.errors ?? [],
    ragTargetMs,
  };
}

export function formatMs(ms: number | null, decimals = 0): string {
  if (ms === null) return "—";
  return `${ms.toFixed(decimals)} ms`;
}
