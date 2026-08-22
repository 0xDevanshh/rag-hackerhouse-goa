"use client";

import { motion } from "framer-motion";
import type { PipelineStage } from "../lib/types";

interface PipelineVizProps {
  stages: PipelineStage[];
}

const STATUS_COLORS = {
  PENDING:  { bg: "var(--bg-card)", border: "var(--border-soft)", fg: "var(--fg-muted)", dot: "var(--border-soft)" },
  ACTIVE:   { bg: "var(--bg-card)", border: "var(--accent)",      fg: "var(--fg)",       dot: "var(--accent)" },
  COMPLETE: { bg: "var(--bg-secondary)", border: "var(--border)", fg: "var(--fg)",       dot: "var(--success)" },
  FAILED:   { bg: "var(--danger-bg)",    border: "var(--danger)", fg: "var(--danger)",   dot: "var(--danger)" },
  SKIPPED:  { bg: "var(--bg-card)",      border: "var(--border-soft)", fg: "var(--fg-muted)", dot: "var(--fg-muted)" },
};

export default function PipelineViz({ stages }: PipelineVizProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "stretch",
        gap: 0,
        overflowX: "auto",
        paddingBottom: "0.25rem",
      }}
      role="list"
      aria-label="Pipeline stages"
    >
      {stages.map((stage, i) => {
        const colors = STATUS_COLORS[stage.status];
        const isActive = stage.status === "ACTIVE";

        return (
          <div
            key={stage.id}
            role="listitem"
            style={{ display: "flex", alignItems: "center", minWidth: 0 }}
          >
            {/* Stage node */}
            <motion.div
              initial={false}
              animate={{
                borderColor: colors.border,
                backgroundColor: colors.bg,
              }}
              transition={{ duration: 0.2 }}
              style={{
                position: "relative",
                padding: "0.625rem 0.75rem",
                borderWidth: "1px",
                borderStyle: "solid",
                borderColor: colors.border,
                borderRadius: "2px",
                minWidth: "80px",
                display: "flex",
                flexDirection: "column",
                gap: "0.25rem",
                overflow: "hidden",
              }}
              aria-label={`${stage.label}: ${stage.status}`}
            >
              {/* Active pulse overlay */}
              {isActive && (
                <motion.div
                  animate={{ opacity: [0.15, 0.3, 0.15] }}
                  transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: "var(--accent)",
                    pointerEvents: "none",
                  }}
                />
              )}

              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.5625rem",
                  fontWeight: 500,
                  letterSpacing: "0.12em",
                  color: "var(--fg-muted)",
                  lineHeight: 1,
                }}
              >
                {stage.shortLabel}
              </div>
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.625rem",
                  fontWeight: 500,
                  letterSpacing: "0.08em",
                  color: colors.fg,
                  lineHeight: 1,
                }}
              >
                {stage.label}
              </div>
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.5625rem",
                  color: stage.durationMs !== null ? "var(--accent2)" : "var(--fg-muted)",
                  lineHeight: 1,
                }}
              >
                {stage.durationMs !== null
                  ? `${stage.durationMs.toFixed(0)}ms`
                  : stage.status === "COMPLETE" ? "< 1ms" : "—"}
              </div>

              {/* Status indicator dot */}
              <div
                style={{
                  position: "absolute",
                  top: "0.375rem",
                  right: "0.375rem",
                  width: "5px",
                  height: "5px",
                  borderRadius: "50%",
                  background: colors.dot,
                }}
              />
            </motion.div>

            {/* Connector arrow */}
            {i < stages.length - 1 && (
              <div
                aria-hidden="true"
                style={{
                  display: "flex",
                  alignItems: "center",
                  padding: "0 2px",
                  flexShrink: 0,
                }}
              >
                <svg width="12" height="8" viewBox="0 0 12 8" fill="none">
                  <line x1="0" y1="4" x2="8" y2="4" stroke="var(--border-soft)" strokeWidth="1"/>
                  <path d="M6 1L10 4L6 7" stroke="var(--border-soft)" strokeWidth="1" fill="none"/>
                </svg>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
