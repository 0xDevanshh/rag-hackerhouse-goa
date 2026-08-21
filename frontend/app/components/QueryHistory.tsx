"use client";

import type { HistoryItem, FrontendResult } from "../lib/types";

interface QueryHistoryProps {
  history: HistoryItem[];
  onRestore: (result: FrontendResult) => void;
  onClear: () => void;
}

const STATUS_STYLES = {
  COMPLETE:  { color: "var(--success)", label: "OK" },
  DEGRADED:  { color: "var(--warning)", label: "DEGRADED" },
  ERROR:     { color: "var(--danger)",  label: "ERR" },
};

export default function QueryHistory({ history, onRestore, onClear }: QueryHistoryProps) {
  if (history.length === 0) return null;

  return (
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
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "0.5rem 0.75rem",
          background: "var(--bg-secondary)",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.5625rem",
            letterSpacing: "0.12em",
            color: "var(--fg-muted)",
          }}
        >
          RECENT QUERIES
        </span>
        <button
          className="btn btn-ghost"
          onClick={onClear}
          style={{ fontSize: "0.5rem", padding: "0.125rem 0.5rem" }}
          aria-label="Clear query history"
        >
          CLEAR
        </button>
      </div>

      {/* Items */}
      <div style={{ maxHeight: "14rem", overflowY: "auto" }}>
        {history.map((item) => {
          const statusStyle = STATUS_STYLES[item.status];
          const time = new Date(item.timestamp);
          const timeStr = time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

          return (
            <button
              key={item.id}
              onClick={() => item.result && onRestore(item.result)}
              disabled={!item.result}
              style={{
                width: "100%",
                display: "grid",
                gridTemplateColumns: "auto 1fr auto auto",
                alignItems: "center",
                gap: "0.625rem",
                padding: "0.5rem 0.75rem",
                borderBottom: "1px solid var(--border-soft)",
                background: "transparent",
                border: "none",
                borderBottomWidth: "1px",
                borderBottomStyle: "solid",
                borderBottomColor: "var(--border-soft)",
                cursor: item.result ? "pointer" : "default",
                textAlign: "left",
              }}
              aria-label={`Restore query: ${item.query}`}
            >
              {/* Mode icon */}
              <span style={{ color: "var(--fg-muted)" }}>
                {item.mode === "VOICE" ? (
                  <svg width="10" height="10" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                    <rect x="4" y="1" width="4" height="7" rx="2" fill="currentColor"/>
                    <path d="M2 6a4 4 0 008 0" stroke="currentColor" strokeWidth="1.5"/>
                  </svg>
                ) : (
                  <svg width="10" height="10" viewBox="0 0 12 12" fill="none" aria-hidden="true">
                    <path d="M1 3h10M1 6h8M1 9h5" stroke="currentColor" strokeWidth="1.5"/>
                  </svg>
                )}
              </span>

              {/* Query text */}
              <span
                style={{
                  fontFamily: "var(--font-body)",
                  fontSize: "0.75rem",
                  color: "var(--fg-secondary)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {item.query}
              </span>

              {/* Status */}
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.5rem",
                  color: statusStyle.color,
                  letterSpacing: "0.06em",
                  flexShrink: 0,
                }}
              >
                {statusStyle.label}
              </span>

              {/* Time + latency */}
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.5rem",
                  color: "var(--fg-muted)",
                  flexShrink: 0,
                  textAlign: "right",
                }}
              >
                {item.totalMs !== null ? `${item.totalMs.toFixed(0)}ms` : timeStr}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
