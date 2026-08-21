"use client";

import { motion } from "framer-motion";

const STEPS = [
  {
    number: "01",
    label: "ASK",
    description: "Speak or type naturally in English or Hindi.",
    detail: "Browser microphone → PCM stream → Sarvam STT",
  },
  {
    number: "02",
    label: "RETRIEVE",
    description: "Relevant multilingual passages are ranked.",
    detail: "FAISS dense search + BM25 lexical, RRF fusion, top-5",
  },
  {
    number: "03",
    label: "VERIFY",
    description: "Answers are checked against evidence.",
    detail: "Sentence-level cosine similarity ≥ 0.5 threshold",
  },
];

export default function EvidenceTrail() {
  return (
    <div>
      <div className="label-mono" style={{ marginBottom: "0.25rem" }}>
        EVIDENCE TRAIL
      </div>
      <h2
        style={{
          fontFamily: "var(--font-head)",
          fontSize: "clamp(1.5rem, 3vw, 2rem)",
          fontWeight: 800,
          letterSpacing: "-0.03em",
          color: "var(--fg)",
          marginBottom: "2rem",
          lineHeight: 1.1,
        }}
      >
        HOW IT WORKS
      </h2>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "1px",
          background: "var(--border)",
          border: "1px solid var(--border)",
          borderRadius: "2px",
          overflow: "hidden",
        }}
      >
        {STEPS.map((step, i) => (
          <motion.div
            key={step.number}
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.3, delay: i * 0.1 }}
            style={{
              padding: "1.5rem",
              background: "var(--bg-card)",
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "2rem",
                fontWeight: 400,
                color: "var(--border-soft)",
                lineHeight: 1,
                marginBottom: "0.75rem",
              }}
            >
              {step.number}
            </div>
            <div
              style={{
                fontFamily: "var(--font-head)",
                fontSize: "1.25rem",
                fontWeight: 800,
                letterSpacing: "-0.02em",
                color: "var(--fg)",
                marginBottom: "0.5rem",
              }}
            >
              {step.label}
            </div>
            <p
              style={{
                fontFamily: "var(--font-body)",
                fontSize: "0.9375rem",
                color: "var(--fg-secondary)",
                lineHeight: 1.5,
                marginBottom: "0.625rem",
              }}
            >
              {step.description}
            </p>
            <p
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.625rem",
                color: "var(--fg-muted)",
                lineHeight: 1.5,
              }}
            >
              {step.detail}
            </p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
