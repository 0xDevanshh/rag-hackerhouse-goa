"use client";

import { motion } from "framer-motion";
import type { LatencyBreakdown } from "../lib/types";
import { formatMs } from "../lib/normalize";

interface LatencyPanelProps {
  latency: LatencyBreakdown;
  ragTargetMs?: number;
}

function LatencyRow({
  label,
  value,
  note,
  highlight,
  isTotal,
  isSavings,
}: {
  label: string;
  value: string;
  note?: string;
  highlight?: boolean;
  isTotal?: boolean;
  isSavings?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        padding: "0.4375rem 0.625rem",
        borderBottom: "1px solid var(--border-soft)",
        gap: "1rem",
        background: isTotal ? "var(--bg-secondary)" : "transparent",
      }}
    >
      <div>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.6875rem",
            color: isTotal ? "var(--fg)" : "var(--fg-secondary)",
            fontWeight: isTotal ? 600 : 400,
          }}
        >
          {label}
        </span>
        {note && (
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.5625rem",
              color: "var(--fg-muted)",
              marginTop: "0.125rem",
            }}
          >
            {note}
          </div>
        )}
      </div>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.6875rem",
          fontWeight: isTotal ? 700 : 500,
          color: isSavings
            ? "var(--success)"
            : highlight
              ? "var(--accent2)"
              : isTotal
                ? "var(--fg)"
                : "var(--fg-muted)",
          whiteSpace: "nowrap",
        }}
      >
        {isSavings && value !== "—" ? `−${value}` : value}
      </span>
    </div>
  );
}

function SLAMeter({
  actual,
  target,
  label,
}: {
  actual: number | null;
  target: number;
  label: string;
}) {
  if (actual === null) return null;

  const pct = Math.min((actual / target) * 100, 130);
  const pass = actual <= target;
  const barColor = pass ? "var(--success)" : "var(--danger)";
  const overflow = !pass;

  return (
    <div
      style={{
        padding: "0.875rem",
        border: `1px solid ${pass ? "var(--success)" : "var(--danger)"}`,
        borderRadius: "2px",
        background: pass ? "var(--success-bg)" : "var(--danger-bg)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "0.5rem",
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.625rem",
            fontWeight: 500,
            letterSpacing: "0.1em",
            color: pass ? "var(--success)" : "var(--danger)",
          }}
        >
          {label}
        </span>
        <span
          className={pass ? "chip chip-success" : "chip chip-danger"}
        >
          {pass ? "PASS" : "MISS"}
        </span>
      </div>

      {/* Bar */}
      <div
        style={{
          height: "4px",
          background: "var(--border)",
          borderRadius: "2px",
          position: "relative",
          marginBottom: "0.375rem",
          overflow: "hidden",
        }}
      >
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(pct, 100)}%` }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            height: "100%",
            background: barColor,
            borderRadius: "2px",
          }}
        />
        {overflow && (
          <div
            style={{
              position: "absolute",
              right: 0,
              top: 0,
              height: "100%",
              width: "4px",
              background: "var(--danger)",
              animation: "overflow-blink 0.8s ease infinite",
            }}
          />
        )}
      </div>

      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.5625rem",
          color: pass ? "var(--success)" : "var(--danger)",
        }}
      >
        {actual.toFixed(1)} ms / {target} ms target
      </div>

      <style>{`
        @keyframes overflow-blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
      `}</style>
    </div>
  );
}

export default function LatencyPanel({
  latency,
  ragTargetMs = 200,
}: LatencyPanelProps) {
  const {
    totalMs, ragCoreMs, sttMs, embeddingMs, retrievalMs, bm25Ms, fusionMs,
    rerankMs, generationMs, groundingMs, guardrailsMs, unaccountedMs, overheadMs,
    llmTtftMs, sttToFirstPartialMs, sttFinalMs, retrievalOnPartialMs,
    sttOverlapSavingsMs, isOverlappedPath, isCached,
  } = latency;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Header */}
      <div style={{ marginBottom: "1rem" }}>
        <div className="label-mono" style={{ marginBottom: "0.25rem" }}>
          LATENCY TRACE
        </div>
        {isCached && (
          <span className="chip chip-muted">CACHE HIT — trace shows lookup cost only</span>
        )}
      </div>

      {/* SLA meters */}
      <div className="sla-grid" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "1rem" }}>
        <SLAMeter
          actual={ragCoreMs}
          target={ragTargetMs}
          label="RAG CORE SLA"
        />
        {totalMs !== null && sttMs !== null && (
          <div
            style={{
              padding: "0.875rem",
              border: "1px solid var(--border)",
              borderRadius: "2px",
              background: "var(--bg-card)",
            }}
          >
            <div className="label-mono" style={{ marginBottom: "0.5rem" }}>FULL VOICE E2E</div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "1.25rem",
                fontWeight: 500,
                color: "var(--fg)",
              }}
            >
              {formatMs(totalMs)}
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.5625rem",
                color: "var(--fg-muted)",
                marginTop: "0.25rem",
              }}
            >
              Includes STT network round-trip
            </div>
          </div>
        )}
      </div>

      {/* Trace table */}
      <div
        style={{
          border: "1px solid var(--border)",
          borderRadius: "2px",
          overflow: "hidden",
        }}
      >
        {/* Table header */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            padding: "0.375rem 0.625rem",
            background: "var(--bg-secondary)",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.5625rem", letterSpacing: "0.1em", color: "var(--fg-muted)" }}>STAGE</span>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.5625rem", letterSpacing: "0.1em", color: "var(--fg-muted)" }}>DURATION</span>
        </div>

        {/* STT rows */}
        {isOverlappedPath ? (
          <>
            <LatencyRow
              label="STT → FIRST PARTIAL"
              value={formatMs(sttToFirstPartialMs)}
              note="time until first transcript.partial"
            />
            <LatencyRow
              label="STT TOTAL"
              value={formatMs(sttFinalMs)}
              note="stream open → transcript.final"
            />
            <LatencyRow
              label="RETRIEVAL ON PARTIAL"
              value={formatMs(retrievalOnPartialMs)}
              note="ran concurrently with STT"
            />
            {sttOverlapSavingsMs !== null && (
              <LatencyRow
                label="⚡ OVERLAP SAVINGS"
                value={formatMs(sttOverlapSavingsMs)}
                note="retrieval ms hidden inside STT"
                isSavings
              />
            )}
          </>
        ) : sttMs !== null ? (
          <LatencyRow
            label="SPEECH-TO-TEXT"
            value={formatMs(sttMs)}
            note="batch STT network round-trip"
          />
        ) : null}

        {/* Core RAG stages */}
        <LatencyRow label="EMBEDDING" value={formatMs(embeddingMs)} />
        <LatencyRow
          label="VECTOR SEARCH (FAISS)"
          value={formatMs(retrievalMs)}
          note="dense semantic retrieval"
        />
        {bm25Ms !== null && (
          <LatencyRow label="BM25 LEXICAL" value={formatMs(bm25Ms)} />
        )}
        {fusionMs !== null && (
          <LatencyRow label="RRF FUSION" value={formatMs(fusionMs)} note="reciprocal rank fusion" />
        )}
        {rerankMs !== null && (
          <LatencyRow label="RERANKING" value={formatMs(rerankMs)} />
        )}
        {guardrailsMs !== null && (
          <LatencyRow
            label="GUARDRAILS"
            value={formatMs(guardrailsMs)}
            note="input + relevance + grounding"
          />
        )}
        {generationMs !== null && (
          <LatencyRow
            label="GENERATION"
            value={formatMs(generationMs)}
            note={llmTtftMs !== null ? `first token at ${formatMs(llmTtftMs)}` : undefined}
          />
        )}
        {overheadMs !== null && (
          <LatencyRow
            label="SERVER OVERHEAD"
            value={formatMs(overheadMs)}
            note="middleware, body parse, serialization, flush"
          />
        )}
        {unaccountedMs !== null && (
          <LatencyRow
            label="UNACCOUNTED"
            value={formatMs(unaccountedMs)}
            note="wall clock not claimed by any stage"
          />
        )}

        {/* Totals */}
        <LatencyRow
          label="RAG CORE (excl. STT + overhead)"
          value={formatMs(ragCoreMs)}
          isTotal
        />
        <LatencyRow
          label="FULL END-TO-END"
          value={formatMs(totalMs)}
          isTotal
        />
      </div>

      {/* Important note */}
      <div
        style={{
          marginTop: "0.625rem",
          padding: "0.5rem 0.75rem",
          background: "var(--bg-card)",
          border: "1px solid var(--border-soft)",
          borderRadius: "2px",
          fontFamily: "var(--font-mono)",
          fontSize: "0.5625rem",
          color: "var(--fg-muted)",
          lineHeight: 1.6,
        }}
      >
        RAG CORE ≠ FULL E2E. The &lt;{ragTargetMs}ms SLA applies to retrieval + grounding only.
        Full voice pipeline includes STT network latency (~250–525ms) which cannot be eliminated.
      </div>
    </motion.div>
  );
}
