"use client";

const ITEMS = [
  "VOICE RAG",
  "MULTILINGUAL RETRIEVAL",
  "ENGLISH + HINDI",
  "HYBRID SEARCH",
  "GROUNDED ANSWERS",
  "VOICE-FIRST RESEARCH",
  "ASK. RETRIEVE. VERIFY.",
  "FAISS + BM25 FUSION",
  "GUARDRAIL VERIFIED",
  "<200ms RAG CORE",
];

export default function Ticker() {
  const content = ITEMS.join("  /  ");
  const doubled = `${content}  /  ${content}`;

  return (
    <div
      style={{
        background: "var(--ticker-bg)",
        color: "var(--ticker-fg)",
        borderBottom: "1px solid var(--border)",
        overflow: "hidden",
        height: "2.25rem",
        display: "flex",
        alignItems: "center",
      }}
      aria-hidden="true"
    >
      <div
        style={{
          display: "flex",
          gap: 0,
          whiteSpace: "nowrap",
          animation: "ticker-scroll 40s linear infinite",
          fontFamily: "var(--font-mono)",
          fontSize: "0.625rem",
          fontWeight: 500,
          letterSpacing: "0.12em",
          opacity: 0.85,
        }}
      >
        <span>{doubled}</span>
      </div>
      <style>{`
        @keyframes ticker-scroll {
          from { transform: translateX(0); }
          to   { transform: translateX(-50%); }
        }
        @media (prefers-reduced-motion: reduce) {
          [style*="ticker-scroll"] { animation: none; }
        }
      `}</style>
    </div>
  );
}
