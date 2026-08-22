"use client";

export default function Footer() {
  return (
    <footer
      style={{
        borderTop: "1px solid var(--border)",
        background: "var(--bg-card)",
        padding: "2.5rem 0",
        marginTop: "auto",
      }}
    >
      <div
        className="container footer-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto",
          alignItems: "end",
          gap: "2rem",
        }}
      >
        {/* Left */}
        <div>
          <div
            style={{
              fontFamily: "var(--font-head)",
              fontWeight: 800,
              fontSize: "1.25rem",
              letterSpacing: "-0.03em",
              color: "var(--fg)",
              marginBottom: "0.375rem",
            }}
          >
            VOICE RAG™
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.625rem",
              letterSpacing: "0.1em",
              color: "var(--fg-muted)",
              marginBottom: "0.875rem",
            }}
          >
            HACKER HOUSE GOA  ·  BUILT FOR #RAGGING  ·  ASK. RETRIEVE. VERIFY.
          </div>
          <div
            style={{
              fontFamily: "var(--font-body)",
              fontSize: "0.8125rem",
              color: "var(--fg-muted)",
              lineHeight: 1.6,
              maxWidth: "400px",
            }}
          >
            A multilingual voice-first RAG system that retrieves evidence, verifies grounding,
            and returns answers you can inspect.
          </div>
        </div>

        {/* Right — links */}
        <div
          className="footer-links"
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            gap: "0.375rem",
          }}
        >
          {[
            { label: "GITHUB", href: "https://github.com" },
            { label: "ARCHITECTURE", href: "#architecture" },
            { label: "LATENCY", href: "#telemetry" },
            { label: "GUARDRAILS", href: "#guardrails" },
          ].map((link) => (
            <a
              key={link.label}
              href={link.href}
              target={link.href.startsWith("http") ? "_blank" : undefined}
              rel={link.href.startsWith("http") ? "noopener noreferrer" : undefined}
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.625rem",
                letterSpacing: "0.1em",
                color: "var(--fg-muted)",
                transition: "color var(--duration-fast)",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "var(--fg)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "var(--fg-muted)")}
            >
              {link.label}
            </a>
          ))}
        </div>
      </div>
    </footer>
  );
}
