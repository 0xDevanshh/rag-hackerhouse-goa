"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import Ticker from "./components/Ticker";
import Navigation from "./components/Navigation";
import VoiceTerminal from "./components/VoiceTerminal";
import AnswerSection from "./components/AnswerSection";
import EvidenceSection from "./components/EvidenceSection";
import LatencyPanel from "./components/LatencyPanel";
import GuardrailsSection from "./components/GuardrailsSection";
import ArchitectureSection from "./components/ArchitectureSection";
import ChunkingSection from "./components/ChunkingSection";
import EvidenceTrail from "./components/EvidenceTrail";
import QueryHistory from "./components/QueryHistory";
import Footer from "./components/Footer";

import type { BackendStatus, FrontendResult, HistoryItem, UiPhase } from "./lib/types";
import { checkHealth } from "./lib/api";

const RAG_TARGET_MS = 200;
const HISTORY_KEY = "vrag_history_v1";
const MAX_HISTORY = 12;

function generateId(): string {
  return Math.random().toString(36).slice(2, 10);
}

function loadHistory(): HistoryItem[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as HistoryItem[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(items: HistoryItem[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_HISTORY)));
  } catch {}
}

type Theme = "light" | "dark";

function getInitialTheme(): Theme {
  try {
    const stored = localStorage.getItem("vrag_theme");
    if (stored === "light" || stored === "dark") return stored;
  } catch {}
  return typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export default function Home() {
  const [theme, setTheme] = useState<Theme>("light");
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("CHECKING");
  const [currentResult, setCurrentResult] = useState<FrontendResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [requestCount, setRequestCount] = useState(0);
  const [isOffline, setIsOffline] = useState(false);
  const [phase, setPhase] = useState<UiPhase>("IDLE");

  const resultRef = useRef<HTMLDivElement>(null);

  // ── Theme init ────────────────────────────────────────────────────────
  useEffect(() => {
    const t = getInitialTheme();
    setTheme(t);
    document.documentElement.setAttribute("data-theme", t);
    setHistory(loadHistory());
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next: Theme = prev === "light" ? "dark" : "light";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("vrag_theme", next); } catch {}
      return next;
    });
  }, []);

  // ── Backend health polling ────────────────────────────────────────────
  useEffect(() => {
    let ignore = false;

    const check = async () => {
      const ok = await checkHealth();
      if (!ignore) {
        setBackendStatus(ok ? "ONLINE" : "OFFLINE");
        setIsOffline(!ok);
      }
    };

    check();
    const interval = setInterval(check, 30_000);
    return () => {
      ignore = true;
      clearInterval(interval);
    };
  }, []);

  // Update nav status when processing
  useEffect(() => {
    if (["RETRIEVING", "GROUNDING", "GENERATING", "TRANSCRIBING", "UPLOADING"].includes(phase)) {
      setBackendStatus("PROCESSING");
    } else if (isOffline) {
      setBackendStatus("OFFLINE");
    } else {
      setBackendStatus("ONLINE");
    }
  }, [phase, isOffline]);

  // ── Handle query result ───────────────────────────────────────────────
  const handleResult = useCallback((result: FrontendResult) => {
    setCurrentResult(result);

    // Add to history
    const item: HistoryItem = {
      id: generateId(),
      timestamp: Date.now(),
      query: result.query,
      mode: result.inputMode,
      status: result.isDegraded ? "DEGRADED" : "COMPLETE",
      totalMs: result.latency.totalMs,
      result,
    };
    setHistory((prev) => {
      const next = [item, ...prev].slice(0, MAX_HISTORY);
      saveHistory(next);
      return next;
    });

    // Scroll to results
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 200);
  }, []);

  const handleRestore = useCallback((result: FrontendResult) => {
    setCurrentResult(result);
    setTimeout(() => {
      resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
    try { localStorage.removeItem(HISTORY_KEY); } catch {}
  }, []);

  const resetResult = useCallback(() => {
    setCurrentResult(null);
  }, []);

  const isDegrade = currentResult?.isDegraded ?? false;
  const isRefused =
    currentResult?.grounding.status === "REFUSED" ||
    (currentResult?.answer ?? "").includes("I don't have enough information");

  return (
    <div className="page-wrapper">
      {/* 01 — Ticker */}
      <Ticker />

      {/* 02 — Navigation */}
      <Navigation
        backendStatus={backendStatus}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {/* Offline banner */}
      <AnimatePresence>
        {isOffline && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            style={{ overflow: "hidden" }}
          >
            <div
              style={{
                background: "var(--danger-bg)",
                borderBottom: "1px solid var(--danger)",
                padding: "0.5rem 2rem",
                display: "flex",
                alignItems: "center",
                gap: "0.75rem",
              }}
              role="alert"
            >
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.625rem",
                  fontWeight: 600,
                  letterSpacing: "0.1em",
                  color: "var(--danger)",
                }}
              >
                DEMO MODE / API OFFLINE
              </span>
              <span
                style={{
                  fontFamily: "var(--font-body)",
                  fontSize: "0.8125rem",
                  color: "var(--fg-secondary)",
                }}
              >
                Backend unreachable. Start the FastAPI server to run live queries.
              </span>
              <button
                className="btn btn-ghost"
                onClick={async () => {
                  setBackendStatus("CHECKING");
                  const ok = await checkHealth();
                  setBackendStatus(ok ? "ONLINE" : "OFFLINE");
                  setIsOffline(!ok);
                }}
                style={{ marginLeft: "auto", fontSize: "0.625rem" }}
              >
                RETRY
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <main>
        {/* 03 — Hero + Live voice terminal */}
        <section
          id="hero"
          className="section"
          style={{ paddingBottom: "3rem" }}
        >
          <div className="container">
            <div
              className="hero-grid"
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "4rem",
                alignItems: "start",
              }}
            >
              {/* LEFT — Editorial copy */}
              <div>
                <div className="label-accent" style={{ marginBottom: "1.25rem" }}>
                  01 / VOICE → EVIDENCE → ANSWER
                </div>
                <h1
                  style={{
                    fontFamily: "var(--font-head)",
                    fontWeight: 800,
                    fontSize: "clamp(3rem, 6vw, 5.5rem)",
                    letterSpacing: "-0.04em",
                    lineHeight: 0.95,
                    color: "var(--fg)",
                    marginBottom: "1.5rem",
                  }}
                >
                  VOICE-FIRST
                  <br />
                  <span
                    style={{
                      color: "var(--accent2)",
                      fontStyle: "italic",
                    }}
                  >
                    RESEARCH
                  </span>
                </h1>

                <h2
                  style={{
                    fontFamily: "var(--font-head)",
                    fontWeight: 800,
                    fontSize: "clamp(1.5rem, 3vw, 2.25rem)",
                    letterSpacing: "-0.03em",
                    lineHeight: 1,
                    color: "var(--fg)",
                    marginBottom: "1.5rem",
                  }}
                >
                  ASK.
                  <br />
                  RETRIEVE.
                  <br />
                  VERIFY.
                </h2>

                <p
                  style={{
                    fontFamily: "var(--font-body)",
                    fontSize: "1rem",
                    lineHeight: 1.65,
                    color: "var(--fg-secondary)",
                    marginBottom: "2rem",
                    maxWidth: "440px",
                  }}
                >
                  A multilingual voice-first RAG system that retrieves evidence,
                  checks relevance, verifies grounding, and returns answers
                  you can inspect.
                </p>

                {/* Metadata chips */}
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "2rem" }}>
                  {["EN + HI", "VOICE ENABLED", "HYBRID RETRIEVAL", "GROUNDED OUTPUT"].map((tag) => (
                    <span key={tag} className="chip chip-muted">{tag}</span>
                  ))}
                </div>

                {/* Pipeline arrow diagram */}
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.625rem",
                    letterSpacing: "0.06em",
                    color: "var(--fg-muted)",
                    lineHeight: 2.2,
                    borderLeft: "2px solid var(--border-soft)",
                    paddingLeft: "0.875rem",
                  }}
                >
                  {[
                    { label: "AUDIO", active: false },
                    { label: "SPEECH RECOGNITION", active: false },
                    { label: "QUERY UNDERSTANDING", active: false },
                    { label: "HYBRID RETRIEVAL", active: true },
                    { label: "EVIDENCE RANKING", active: false },
                    { label: "GROUNDING", active: true },
                    { label: "VERIFIED ANSWER", active: false },
                  ].map((step) => (
                    <div
                      key={step.label}
                      style={{
                        color: step.active ? "var(--accent2)" : "var(--fg-muted)",
                        fontWeight: step.active ? 500 : 400,
                      }}
                    >
                      ↓ {step.label}
                    </div>
                  ))}
                </div>
              </div>

              {/* RIGHT — Voice terminal */}
              <div>
                <VoiceTerminal
                  isOffline={isOffline}
                  onResult={handleResult}
                  onPhaseChange={setPhase}
                  requestCount={requestCount}
                  onRequestCountChange={setRequestCount}
                />
              </div>
            </div>
          </div>
        </section>

        {/* Results area */}
        <AnimatePresence>
          {currentResult && (
            <motion.div
              ref={resultRef}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3 }}
            >
              {/* Degraded / Refused alert */}
              {(isDegrade || isRefused) && (
                <section style={{ paddingBottom: "1.5rem" }}>
                  <div className="container-narrow">
                    <div
                      style={{
                        padding: "1rem 1.25rem",
                        border: "1px solid var(--warning)",
                        borderRadius: "2px",
                        background: "var(--warning-bg)",
                        display: "flex",
                        gap: "1rem",
                        alignItems: "flex-start",
                        flexWrap: "wrap",
                      }}
                      role="alert"
                    >
                      <div style={{ flex: 1 }}>
                        <div
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: "0.625rem",
                            fontWeight: 600,
                            letterSpacing: "0.1em",
                            color: "var(--warning)",
                            marginBottom: "0.25rem",
                          }}
                        >
                          {isRefused ? "INSUFFICIENT EVIDENCE" : "DEGRADED RESPONSE"}
                        </div>
                        <p
                          style={{
                            fontFamily: "var(--font-body)",
                            fontSize: "0.875rem",
                            color: "var(--fg-secondary)",
                          }}
                        >
                          {isRefused
                            ? "The system could not verify an answer from the available evidence. This is a successful guardrail outcome — the system correctly declined to hallucinate."
                            : "The pipeline encountered errors and returned a degraded fallback response."}
                        </p>
                      </div>
                      <div style={{ display: "flex", gap: "0.5rem", flexShrink: 0, flexWrap: "wrap" }}>
                        <button className="btn btn-ghost" onClick={resetResult} style={{ fontSize: "0.625rem" }}>
                          TRY ANOTHER QUERY
                        </button>
                      </div>
                    </div>
                  </div>
                </section>
              )}

              {/* 05 — Verified answer */}
              <section className="section-sm" id="answer">
                <div className="container-narrow">
                  <AnswerSection result={currentResult} />
                </div>
              </section>

              {/* 06 — Retrieved evidence */}
              <section className="section-sm" id="evidence">
                <div className="container-narrow">
                  <EvidenceSection
                    evidence={currentResult.evidence}
                    grounding={currentResult.grounding}
                  />
                </div>
              </section>

              {/* 07 — Latency telemetry */}
              <section className="section-sm" id="telemetry">
                <div className="container-narrow">
                  <LatencyPanel
                    latency={currentResult.latency}
                    ragTargetMs={RAG_TARGET_MS}
                  />
                </div>
              </section>

              {/* 08 — Guardrails */}
              <section className="section-sm" id="guardrails">
                <div className="container-narrow">
                  <GuardrailsSection guardrails={currentResult.guardrails} />
                </div>
              </section>

              {/* Ask another */}
              <section style={{ paddingBottom: "3rem" }}>
                <div className="container-narrow">
                  <hr className="divider" style={{ marginBottom: "1.5rem" }} />
                  <button
                    className="btn"
                    onClick={resetResult}
                    style={{ fontSize: "0.75rem" }}
                    aria-label="Clear results and ask another question"
                  >
                    ← ASK ANOTHER QUESTION
                  </button>
                </div>
              </section>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Query history */}
        {history.length > 0 && (
          <section style={{ paddingBottom: "2rem" }}>
            <div className="container-narrow">
              <QueryHistory
                history={history}
                onRestore={handleRestore}
                onClear={clearHistory}
              />
            </div>
          </section>
        )}

        {/* Divider */}
        <div className="container">
          <hr className="divider" style={{ margin: "1rem 0" }} />
        </div>

        {/* 29 — Evidence trail */}
        <section className="section" id="how">
          <div className="container">
            <EvidenceTrail />
          </div>
        </section>

        {/* 09 — Architecture */}
        <section className="section" id="architecture">
          <div className="container">
            <ArchitectureSection />
          </div>
        </section>

        {/* 28 — Chunking */}
        <section className="section-sm" id="chunking">
          <div className="container">
            <ChunkingSection />
          </div>
        </section>
      </main>

      {/* 10 — Footer */}
      <Footer />
    </div>
  );
}
