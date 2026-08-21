"use client";

import { motion } from "framer-motion";

const PIPELINE_STEPS = [
  { label: "VOICE",    note: "Browser microphone → MediaRecorder → PCM stream" },
  { label: "STT",      note: "Sarvam saaras:v3-realtime — partial + final transcript" },
  { label: "NORMALIZE", note: "Unicode NFKC normalization, query validation" },
  { label: "EMBED",    note: "paraphrase-multilingual-MiniLM-L12-v2 → 384-dim vector" },
  { label: "FAISS + BM25", note: "Dense cosine search + lexical BM25, RRF fusion" },
  { label: "RERANK",   note: "Language match + metadata boosts, top-5 selected" },
  { label: "GROUND",   note: "Answer sentences verified against retrieved evidence" },
  { label: "ANSWER",   note: "Extractive fast path or LLM-generated response" },
];

const FEATURE_CARDS = [
  {
    label: "HYBRID RETRIEVAL",
    description: "Dense semantic retrieval via FAISS + lexical BM25 fusion using Reciprocal Rank Fusion (k=60).",
  },
  {
    label: "MULTILINGUAL",
    description: "English + Hindi query handling. MSMARCO-XI corpus indexed in both languages.",
  },
  {
    label: "GROUNDING",
    description: "Answer verification against retrieved evidence via cosine similarity. Two-pass sentence-level check.",
  },
  {
    label: "OVERLAPPED STT",
    description: "Retrieval fires on the first stable partial transcript while STT continues — saving retrieval latency from the critical path.",
  },
  {
    label: "GUARDRAIL STACK",
    description: "InputGuardrail → RelevanceGuardrail → GroundingGuardrail. Refusal is a successful safety outcome.",
  },
  {
    label: "LATENCY HONEST",
    description: "RAG core (<200ms) reported separately from full E2E (includes STT ~250–525ms). No numbers are manufactured.",
  },
];

export default function ArchitectureSection() {
  return (
    <div>
      {/* Section label */}
      <div className="label-mono" style={{ marginBottom: "0.25rem" }}>
        SYSTEM ARCHITECTURE
      </div>
      <h2
        style={{
          fontFamily: "var(--font-head)",
          fontSize: "clamp(1.75rem, 4vw, 2.75rem)",
          fontWeight: 800,
          letterSpacing: "-0.04em",
          lineHeight: 1.05,
          color: "var(--fg)",
          marginBottom: "3rem",
        }}
      >
        FROM VOICE TO<br />
        <span style={{ color: "var(--accent2)" }}>VERIFIED EVIDENCE.</span>
      </h2>

      {/* Pipeline visual */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 0,
          marginBottom: "3rem",
          position: "relative",
        }}
      >
        {PIPELINE_STEPS.map((step, i) => (
          <motion.div
            key={step.label}
            initial={{ opacity: 0, x: -8 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.3, delay: i * 0.06 }}
            style={{
              display: "flex",
              alignItems: "stretch",
              gap: 0,
            }}
          >
            {/* Left track */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                width: "2rem",
                flexShrink: 0,
              }}
            >
              <div
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  border: "2px solid var(--fg)",
                  background: i === 0 || i === PIPELINE_STEPS.length - 1 ? "var(--fg)" : "var(--bg)",
                  flexShrink: 0,
                  marginTop: "1.125rem",
                  position: "relative",
                  zIndex: 1,
                }}
              />
              {i < PIPELINE_STEPS.length - 1 && (
                <div
                  style={{
                    width: "1px",
                    flex: 1,
                    background: "var(--border)",
                    marginTop: "2px",
                    minHeight: "1.5rem",
                  }}
                />
              )}
            </div>

            {/* Content */}
            <div
              style={{
                flex: 1,
                padding: "0.75rem 0 0.75rem 1rem",
                borderBottom: i < PIPELINE_STEPS.length - 1 ? "none" : "none",
              }}
            >
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.8125rem",
                  fontWeight: 500,
                  letterSpacing: "0.06em",
                  color: "var(--fg)",
                  marginBottom: "0.2rem",
                }}
              >
                {step.label}
              </div>
              <div
                style={{
                  fontFamily: "var(--font-body)",
                  fontSize: "0.8125rem",
                  color: "var(--fg-muted)",
                  lineHeight: 1.4,
                }}
              >
                {step.note}
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Feature cards grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "1px",
          background: "var(--border)",
          border: "1px solid var(--border)",
          borderRadius: "2px",
          overflow: "hidden",
        }}
      >
        {FEATURE_CARDS.map((card, i) => (
          <motion.div
            key={card.label}
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.3, delay: i * 0.05 }}
            style={{
              padding: "1.25rem",
              background: "var(--bg-card)",
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.6875rem",
                fontWeight: 500,
                letterSpacing: "0.1em",
                color: "var(--accent2)",
                marginBottom: "0.5rem",
              }}
            >
              {card.label}
            </div>
            <p
              style={{
                fontFamily: "var(--font-body)",
                fontSize: "0.875rem",
                color: "var(--fg-secondary)",
                lineHeight: 1.5,
              }}
            >
              {card.description}
            </p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
