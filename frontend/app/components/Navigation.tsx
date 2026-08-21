"use client";

import { useEffect, useState } from "react";
import type { BackendStatus } from "../lib/types";

interface NavigationProps {
  backendStatus: BackendStatus;
  theme: "light" | "dark";
  onToggleTheme: () => void;
}

const STATUS_LABELS: Record<BackendStatus, string> = {
  CHECKING:   "API CONNECTING",
  ONLINE:     "SYSTEM ONLINE",
  CONNECTING: "API CONNECTING",
  OFFLINE:    "API OFFLINE",
  PROCESSING: "PROCESSING",
};

const STATUS_DOT_CLASS: Record<BackendStatus, string> = {
  CHECKING:   "checking",
  ONLINE:     "online",
  CONNECTING: "checking",
  OFFLINE:    "offline",
  PROCESSING: "processing",
};

export default function Navigation({
  backendStatus,
  theme,
  onToggleTheme,
}: NavigationProps) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav
      aria-label="Primary navigation"
      style={{
        position: "sticky",
        top: 0,
        zIndex: 100,
        background: "var(--nav-bg)",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        borderBottom: scrolled ? "1px solid var(--border)" : "1px solid transparent",
        transition: "border-color var(--duration-med) var(--ease-out)",
      }}
    >
      <div
        className="container"
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto 1fr",
          alignItems: "center",
          height: "3.5rem",
          gap: "1rem",
        }}
      >
        {/* LEFT — Brand */}
        <a
          href="#hero"
          style={{
            fontFamily: "var(--font-head)",
            fontWeight: 800,
            fontSize: "1rem",
            letterSpacing: "-0.02em",
            color: "var(--fg)",
          }}
        >
          VOICE RAG™
        </a>

        {/* CENTER — System status */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            fontFamily: "var(--font-mono)",
            fontSize: "0.6875rem",
            fontWeight: 500,
            letterSpacing: "0.1em",
            color: backendStatus === "ONLINE" ? "var(--success)" : "var(--fg-muted)",
          }}
          aria-live="polite"
          aria-label={`System status: ${STATUS_LABELS[backendStatus]}`}
        >
          <span className={`status-dot ${STATUS_DOT_CLASS[backendStatus]}`} />
          <span>{STATUS_LABELS[backendStatus]}</span>
        </div>

        {/* RIGHT — Controls */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
            justifyContent: "flex-end",
          }}
        >
          <button
            onClick={onToggleTheme}
            className="btn btn-ghost"
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
            style={{ fontSize: "0.625rem" }}
          >
            {theme === "light" ? "DARK" : "LIGHT"}
          </button>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-ghost"
            aria-label="View source on GitHub"
            style={{ fontSize: "0.625rem" }}
          >
            GITHUB
            <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
              <path d="M2 8L8 2M8 2H4M8 2V6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="square"/>
            </svg>
          </a>
        </div>
      </div>
    </nav>
  );
}
