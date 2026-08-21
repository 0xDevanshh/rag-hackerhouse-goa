"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { FrontendResult } from "../lib/types";

interface AnswerSectionProps {
  result: FrontendResult;
}

function copyToClipboard(text: string, onDone: () => void) {
  navigator.clipboard.writeText(text).then(onDone).catch(() => {});
}

export default function AnswerSection({ result }: AnswerSectionProps) {
  const [copied, setCopied] = useState(false);

  const { grounding, isDegraded, answer, query, inputMode, isCached } = result;

  const handleCopy = () => {
    copyToClipboard(answer, () => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({ title: "Voice RAG Answer", text: `Q: ${query}\nA: ${answer}` }).catch(() => {});
    } else {
      copyToClipboard(`Q: ${query}\nA: ${answer}`, () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      });
    }
  };

  const statusBadge =
    grounding.status === "GROUNDED" ? (
      <span className="chip chip-success">✓ GROUNDED</span>
    ) : grounding.status === "REFUSED" ? (
      <span className="chip chip-danger">× REFUSED</span>
    ) : isDegraded ? (
      <span className="chip chip-warning">! DEGRADED</span>
    ) : (
      <span className="chip chip-muted">UNVERIFIED</span>
    );

  const isRefused =
    grounding.status === "REFUSED" ||
    answer.includes("I don't have enough information");

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
    >
      <div
        style={{
          border: "1px solid var(--border)",
          borderRadius: "2px",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            background: "var(--bg-card)",
            borderBottom: "1px solid var(--border)",
            padding: "0.75rem 1.25rem",
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            flexWrap: "wrap",
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.6875rem",
              fontWeight: 500,
              letterSpacing: "0.12em",
              color: "var(--fg-muted)",
            }}
          >
            VERIFIED RESPONSE
          </span>
          {statusBadge}
          {isCached && <span className="chip chip-muted">CACHED</span>}
          {inputMode === "VOICE" && (
            <span className="chip chip-muted">
              <svg width="8" height="8" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                <rect x="4" y="1" width="4" height="7" rx="2" fill="currentColor"/>
                <path d="M2 6a4 4 0 008 0" stroke="currentColor" strokeWidth="1.5"/>
                <line x1="6" y1="10" x2="6" y2="12" stroke="currentColor" strokeWidth="1.5"/>
              </svg>
              VOICE INPUT
            </span>
          )}
        </div>

        {/* Query */}
        <div
          style={{
            padding: "1rem 1.25rem",
            borderBottom: "1px solid var(--border-soft)",
          }}
        >
          <div className="label-mono" style={{ marginBottom: "0.375rem" }}>
            YOUR QUERY
          </div>
          <p
            style={{
              fontFamily: "var(--font-body)",
              fontSize: "0.9375rem",
              color: "var(--fg-secondary)",
              fontStyle: "italic",
              lineHeight: 1.5,
            }}
          >
            &ldquo;{query}&rdquo;
          </p>
        </div>

        {/* Answer */}
        <div style={{ padding: "1.25rem" }}>
          <div className="label-mono" style={{ marginBottom: "0.625rem" }}>
            ANSWER
          </div>

          {isRefused ? (
            <div
              style={{
                padding: "1rem",
                border: "1px solid var(--warning)",
                borderRadius: "2px",
                background: "var(--warning-bg)",
                marginBottom: "1rem",
              }}
            >
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.75rem",
                  fontWeight: 500,
                  letterSpacing: "0.08em",
                  color: "var(--warning)",
                  marginBottom: "0.5rem",
                }}
              >
                INSUFFICIENT EVIDENCE
              </div>
              <p
                style={{
                  fontSize: "0.9375rem",
                  color: "var(--fg)",
                  lineHeight: 1.6,
                }}
              >
                {answer}
              </p>
            </div>
          ) : (
            <p
              style={{
                fontSize: "1.0625rem",
                lineHeight: 1.65,
                color: "var(--fg)",
                whiteSpace: "pre-wrap",
                fontWeight: 400,
              }}
            >
              {answer}
            </p>
          )}

          {/* Actions */}
          <div
            style={{
              display: "flex",
              gap: "0.5rem",
              marginTop: "1rem",
              flexWrap: "wrap",
            }}
          >
            <AnimatePresence mode="wait">
              <motion.button
                key={copied ? "copied" : "copy"}
                initial={{ opacity: 0.8 }}
                animate={{ opacity: 1 }}
                className="btn btn-ghost"
                onClick={handleCopy}
                aria-label="Copy answer to clipboard"
                style={{ fontSize: "0.625rem" }}
              >
                {copied ? "COPIED" : "COPY ANSWER"}
              </motion.button>
            </AnimatePresence>
            <button
              className="btn btn-ghost"
              onClick={handleShare}
              aria-label="Share result"
              style={{ fontSize: "0.625rem" }}
            >
              SHARE RESULT
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
