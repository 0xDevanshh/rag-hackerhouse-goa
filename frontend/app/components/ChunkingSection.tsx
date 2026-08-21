"use client";

import { motion } from "framer-motion";

// Strategies from src/chunking.py ChunkerRegistry._strategies
// Production default for MSMARCO-XI is "metadata_aware" (see src/api.py _load_msmarco_chunks)
// Production default for demo corpus is "recursive" (see src/api.py _load_sample_chunks)

const STRATEGIES = [
  {
    id: "fixed_size",
    label: "FIXED SIZE",
    description: "Sliding window over characters. Fixed chunk size with configurable overlap.",
    isProduction: false,
  },
  {
    id: "sentence",
    label: "SENTENCE",
    description: "One chunk per sentence, preserving document order.",
    isProduction: false,
  },
  {
    id: "paragraph",
    label: "PARAGRAPH",
    description: "One chunk per non-empty paragraph, split on double newlines.",
    isProduction: false,
  },
  {
    id: "sentence_semantic",
    label: "SEMANTIC",
    description: "Sentence embeddings with greedy merge on cosine similarity. Thresholded splits.",
    isProduction: false,
  },
  {
    id: "metadata_aware",
    label: "METADATA AWARE",
    description: "One chunk per MSMARCO-XI passage. Preserves is_selected, query_id, passage_id metadata.",
    isProduction: true,
    productionLabel: "PRODUCTION / MSMARCO-XI",
  },
  {
    id: "recursive",
    label: "RECURSIVE",
    description: "Split by paragraph → sentence → pack within max_chunk_size. Used for demo corpus.",
    isProduction: true,
    productionLabel: "PRODUCTION / DEMO CORPUS",
  },
  {
    id: "sliding_window",
    label: "SLIDING WINDOW",
    description: "Alias for Fixed Size. Character-level windows with overlap for long documents.",
    isProduction: false,
  },
];

export default function ChunkingSection() {
  return (
    <div>
      <div className="label-mono" style={{ marginBottom: "0.25rem" }}>
        CHUNKING ENGINE
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          gap: "1rem",
          marginBottom: "1.5rem",
          flexWrap: "wrap",
        }}
      >
        <h2
          style={{
            fontFamily: "var(--font-head)",
            fontSize: "clamp(1.5rem, 3vw, 2rem)",
            fontWeight: 800,
            letterSpacing: "-0.03em",
            color: "var(--fg)",
            lineHeight: 1.1,
          }}
        >
          7 STRATEGIES
        </h2>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.75rem",
            color: "var(--fg-muted)",
          }}
        >
          2 IN PRODUCTION
        </span>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: "0.5rem",
        }}
      >
        {STRATEGIES.map((s, i) => (
          <motion.div
            key={s.id}
            initial={{ opacity: 0, y: 4 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.2, delay: i * 0.04 }}
            style={{
              padding: "1rem",
              border: `1px solid ${s.isProduction ? "var(--accent)" : "var(--border)"}`,
              borderRadius: "2px",
              background: s.isProduction ? "var(--bg-secondary)" : "var(--bg-card)",
              position: "relative",
            }}
          >
            {s.isProduction && s.productionLabel && (
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.5rem",
                  fontWeight: 500,
                  letterSpacing: "0.1em",
                  color: "var(--accent)",
                  marginBottom: "0.375rem",
                }}
              >
                {s.productionLabel}
              </div>
            )}
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.6875rem",
                fontWeight: 500,
                letterSpacing: "0.08em",
                color: s.isProduction ? "var(--fg)" : "var(--fg-secondary)",
                marginBottom: "0.375rem",
              }}
            >
              {s.label}
            </div>
            <p
              style={{
                fontFamily: "var(--font-body)",
                fontSize: "0.75rem",
                color: "var(--fg-muted)",
                lineHeight: 1.5,
              }}
            >
              {s.description}
            </p>
          </motion.div>
        ))}
      </div>

      <div
        style={{
          marginTop: "0.75rem",
          padding: "0.625rem 0.875rem",
          border: "1px solid var(--border-soft)",
          borderRadius: "2px",
          fontFamily: "var(--font-mono)",
          fontSize: "0.5625rem",
          color: "var(--fg-muted)",
          lineHeight: 1.6,
        }}
      >
        Production strategy is determined by CORPUS env var. MSMARCO-XI uses metadata_aware
        (preserves dataset passage structure). Demo corpus uses recursive (max_chunk_size=300).
        No strategy is marked active unless the backend actually uses it.
      </div>
    </div>
  );
}
