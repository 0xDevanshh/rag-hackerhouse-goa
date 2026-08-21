"use client";

import { motion } from "framer-motion";
import type { GuardrailStatus } from "../lib/types";

interface GuardrailsSectionProps {
  guardrails: GuardrailStatus[];
}

const STATUS_STYLES: Record<GuardrailStatus["status"], { chip: string; icon: string; color: string }> = {
  PASSED:   { chip: "chip-success", icon: "✓", color: "var(--success)" },
  BLOCKED:  { chip: "chip-danger",  icon: "×", color: "var(--danger)"  },
  DEGRADED: { chip: "chip-warning", icon: "!", color: "var(--warning)" },
  UNKNOWN:  { chip: "chip-muted",   icon: "?", color: "var(--fg-muted)" },
};

export default function GuardrailsSection({ guardrails }: GuardrailsSectionProps) {
  return (
    <div>
      <div className="label-mono" style={{ marginBottom: "0.25rem" }}>
        TRUST LAYER
      </div>
      <h2
        style={{
          fontFamily: "var(--font-head)",
          fontSize: "clamp(1.5rem, 3vw, 2rem)",
          fontWeight: 800,
          letterSpacing: "-0.03em",
          color: "var(--fg)",
          marginBottom: "1.5rem",
          lineHeight: 1.1,
        }}
      >
        GUARDRAILS
      </h2>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "0.75rem",
        }}
      >
        {guardrails.map((guard, i) => {
          const styles = STATUS_STYLES[guard.status];
          return (
            <motion.div
              key={guard.name}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, delay: i * 0.08 }}
              style={{
                padding: "1.25rem",
                border: "1px solid var(--border)",
                borderRadius: "2px",
                background: "var(--bg-card)",
                position: "relative",
                overflow: "hidden",
              }}
              role="article"
              aria-label={`${guard.label}: ${guard.status}`}
            >
              {/* Index */}
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.5rem",
                  letterSpacing: "0.12em",
                  color: "var(--fg-muted)",
                  marginBottom: "0.5rem",
                }}
              >
                {String(i + 1).padStart(2, "0")}
              </div>

              {/* Label + status */}
              <div
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  justifyContent: "space-between",
                  gap: "0.5rem",
                  marginBottom: "0.625rem",
                }}
              >
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.75rem",
                    fontWeight: 500,
                    letterSpacing: "0.08em",
                    color: "var(--fg)",
                    lineHeight: 1.3,
                  }}
                >
                  {guard.label}
                </div>
                <span className={`chip ${styles.chip}`} style={{ flexShrink: 0 }}>
                  {styles.icon} {guard.status}
                </span>
              </div>

              {/* Description */}
              <p
                style={{
                  fontFamily: "var(--font-body)",
                  fontSize: "0.8125rem",
                  color: "var(--fg-muted)",
                  lineHeight: 1.5,
                  marginBottom: guard.reason && guard.reason !== "ok" ? "0.5rem" : 0,
                }}
              >
                {guard.description}
              </p>

              {/* Reason (only when non-trivial) */}
              {guard.reason && guard.reason !== "ok" && (
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.5625rem",
                    color: styles.color,
                    marginTop: "0.375rem",
                    paddingTop: "0.375rem",
                    borderTop: "1px solid var(--border-soft)",
                    wordBreak: "break-word",
                  }}
                >
                  REASON: {guard.reason}
                </div>
              )}

              {/* Color bar */}
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  bottom: 0,
                  width: "3px",
                  background: styles.color,
                }}
                aria-hidden="true"
              />
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
