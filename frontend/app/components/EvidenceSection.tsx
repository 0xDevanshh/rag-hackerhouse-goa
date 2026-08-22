"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { EvidenceCard, GroundingInfo } from "../lib/types";

interface EvidenceSectionProps {
  evidence: EvidenceCard[];
  grounding: GroundingInfo;
}

function EvidenceCardItem({ card, index }: { card: EvidenceCard; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showWhy, setShowWhy] = useState(false);

  const scorePercent = Math.round(card.score * 100);
  const scoreColor = card.score >= 0.7 ? "var(--success)" : card.score >= 0.4 ? "var(--warning)" : "var(--danger)";

  const copy = () => {
    navigator.clipboard.writeText(card.text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: index * 0.06, ease: [0.16, 1, 0.3, 1] }}
      style={{
        border: "1px solid var(--border)",
        borderRadius: "2px",
        overflow: "hidden",
      }}
    >
      {/* Card header */}
      <button
        onClick={() => setExpanded(!expanded)}
        style={{
          width: "100%",
          background: "var(--bg-card)",
          borderBottom: expanded ? "1px solid var(--border-soft)" : "none",
          padding: "0.75rem 1rem",
          display: "grid",
          gridTemplateColumns: "auto 1fr auto auto",
          alignItems: "center",
          gap: "0.75rem",
          cursor: "pointer",
          textAlign: "left",
        }}
        aria-expanded={expanded}
        aria-label={`Evidence ${index + 1}: relevance ${scorePercent}%. Click to ${expanded ? "collapse" : "expand"}`}
      >
        {/* Index */}
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.625rem",
            fontWeight: 500,
            color: "var(--fg-muted)",
            minWidth: "1.5rem",
          }}
        >
          {String(index + 1).padStart(2, "0")}
        </span>

        {/* Preview */}
        <span
          style={{
            fontFamily: "var(--font-body)",
            fontSize: "0.8125rem",
            color: "var(--fg-secondary)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {card.text.slice(0, 80)}…
        </span>

        {/* Score */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.625rem",
              fontWeight: 500,
              color: scoreColor,
              whiteSpace: "nowrap",
            }}
          >
            RELEVANCE {scorePercent}%
          </span>
          {card.language && (
            <span className="chip chip-muted" style={{ fontSize: "0.5rem" }}>
              {card.language.toUpperCase()}
            </span>
          )}
        </div>

        {/* Expand chevron */}
        <svg
          width="10"
          height="10"
          viewBox="0 0 10 10"
          fill="none"
          aria-hidden="true"
          style={{
            transform: expanded ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform var(--duration-fast)",
            color: "var(--fg-muted)",
            flexShrink: 0,
          }}
        >
          <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.5"/>
        </svg>
      </button>

      {/* Expanded body */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{ overflow: "hidden" }}
          >
            <div style={{ padding: "1rem" }}>
              <p
                style={{
                  fontFamily: "var(--font-body)",
                  fontSize: "0.9rem",
                  lineHeight: 1.65,
                  color: "var(--fg)",
                  marginBottom: "0.75rem",
                }}
              >
                {card.text}
              </p>

              {/* Metadata grid */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
                  gap: "0.375rem",
                  marginBottom: "0.75rem",
                }}
              >
                {card.passageId !== null && (
                  <MetaItem label="PASSAGE ID" value={card.passageId} />
                )}
                {card.documentId !== null && (
                  <MetaItem label="DOCUMENT" value={card.documentId} />
                )}
                {card.chunkId !== null && (
                  <MetaItem label="CHUNK ID" value={card.chunkId} />
                )}
                <MetaItem label="STRATEGY" value={card.strategyName.replace("_", " ").toUpperCase()} />
                <MetaItem label="SCORE" value={card.score.toFixed(4)} />
              </div>

              {/* Why this evidence? */}
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <button
                  className="btn btn-ghost"
                  onClick={() => setShowWhy(!showWhy)}
                  style={{ fontSize: "0.625rem" }}
                  aria-expanded={showWhy}
                >
                  {showWhy ? "HIDE" : "WHY THIS EVIDENCE?"}
                </button>
                <button
                  className="btn btn-ghost"
                  onClick={copy}
                  style={{ fontSize: "0.625rem" }}
                  aria-label="Copy passage text"
                >
                  {copied ? "COPIED" : "COPY PASSAGE"}
                </button>
              </div>

              <AnimatePresence>
                {showWhy && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    style={{
                      marginTop: "0.75rem",
                      padding: "0.75rem",
                      background: "var(--bg-secondary)",
                      border: "1px solid var(--border-soft)",
                      borderRadius: "2px",
                      overflow: "hidden",
                    }}
                  >
                    <WhyItem label="RELEVANCE SCORE" value={card.score.toFixed(4)} />
                    <WhyItem label="RETRIEVAL METHOD" value="HYBRID (FAISS + BM25 + RRF FUSION)" />
                    <WhyItem label="RANK" value={`#${index + 1} of top-5`} />
                    {card.language && <WhyItem label="LANGUAGE" value={card.language.toUpperCase()} />}
                    {card.isSelected && <WhyItem label="MSMARCO IS_SELECTED" value="TRUE" />}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.5rem", letterSpacing: "0.1em", color: "var(--fg-muted)", marginBottom: "0.125rem" }}>
        {label}
      </div>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.6875rem", color: "var(--fg)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {value}
      </div>
    </div>
  );
}

function WhyItem({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", marginBottom: "0.25rem" }}>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.5625rem", letterSpacing: "0.08em", color: "var(--fg-muted)" }}>{label}</span>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.5625rem", color: "var(--fg)" }}>{value}</span>
    </div>
  );
}

export default function EvidenceSection({ evidence, grounding }: EvidenceSectionProps) {
  const [copiedAll, setCopiedAll] = useState(false);

  const copyAll = () => {
    const text = evidence.map((e, i) => `[${i + 1}] ${e.text}`).join("\n\n");
    navigator.clipboard.writeText(text).then(() => {
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 2000);
    }).catch(() => {});
  };

  const groundingScoreDisplay =
    grounding.score !== null
      ? `${Math.round(grounding.score * 100)}%`
      : grounding.status === "GROUNDED"
        ? "VERIFIED"
        : grounding.status === "REFUSED"
          ? "REFUSED"
          : "—";

  const groundingColor =
    grounding.status === "GROUNDED" ? "var(--success)" :
    grounding.status === "REFUSED"  ? "var(--danger)" :
    "var(--fg-muted)";

  return (
    <div>
      {/* Section header */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          marginBottom: "1rem",
          flexWrap: "wrap",
          gap: "0.75rem",
        }}
      >
        <div>
          <div className="label-mono" style={{ marginBottom: "0.25rem" }}>
            RETRIEVED EVIDENCE
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "1.125rem",
                fontWeight: 500,
                color: groundingColor,
              }}
            >
              GROUNDING {groundingScoreDisplay}
            </span>
            {grounding.status === "GROUNDED" && (
              <span className="chip chip-success">PASSED</span>
            )}
            {grounding.status === "REFUSED" && (
              <span className="chip chip-danger">INSUFFICIENT</span>
            )}
          </div>
        </div>

        <button
          className="btn btn-ghost"
          onClick={copyAll}
          style={{ fontSize: "0.625rem" }}
          aria-label="Copy all evidence passages"
        >
          {copiedAll ? "COPIED" : "COPY EVIDENCE"}
        </button>
      </div>

      {/* Evidence cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {evidence.length > 0 ? (
          evidence.map((card, i) => (
            <EvidenceCardItem key={i} card={card} index={i} />
          ))
        ) : (
          <div
            style={{
              padding: "2rem",
              textAlign: "center",
              border: "1px solid var(--border-soft)",
              borderRadius: "2px",
              fontFamily: "var(--font-mono)",
              fontSize: "0.75rem",
              color: "var(--fg-muted)",
            }}
          >
            NO PASSAGES RETRIEVED
          </div>
        )}
      </div>
    </div>
  );
}
