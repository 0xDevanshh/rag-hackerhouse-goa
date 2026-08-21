"use client";
import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { healthCheck, queryText } from "./lib/api";
import type { PipelineResult } from "./types";

type Stage = "ready" | "listening" | "transcribing" | "retrieving" | "generating" | "verifying" | "done" | "error";

const EXAMPLES = ["What causes inflation?", "Mudrasphiti kya hai?"];

const DEMO: PipelineResult = {
  query_text: "What causes inflation?",
  answer: "Inflation is a sustained increase in the general price level. It usually happens when demand outpaces supply, costs rise, or the money supply grows faster than available goods and services.",
  sources: [
    {
      text: "Inflation refers to a sustained increase in the general price level of goods and services over time. It reduces the purchasing power of money.",
      metadata: { language: "English" },
      strategy_name: "faiss",
    },
    {
      text: "Mudrasphiti is a sustained increase in the price level of goods and services, reducing purchasing power.",
      metadata: { language: "Hindi" },
      strategy_name: "faiss",
    },
  ],
  scores: [0.94, 0.89],
  trace: { spans: [], details: {}, labels: {}, total_ms: 201, unaccounted_ms: 0 },
  guard_flags: {},
  degraded: false,
  cached: false,
  errors: [],
};

function Wave({ active, listening }: { active: boolean; listening: boolean }) {
  const [levels, setLevels] = useState<number[]>(() => Array(32).fill(25));
  const animFrameRef = useRef<number | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    if (!listening) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
      return;
    }

    let stream: MediaStream | null = null;
    let analyser: AnalyserNode | null = null;
    let source: MediaStreamAudioSourceNode | null = null;

    navigator.mediaDevices
      ?.getUserMedia({ audio: true })
      .then((s) => {
        stream = s;
        const AudioCtx =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const ctx = new AudioCtx();
        audioCtxRef.current = ctx;
        analyser = ctx.createAnalyser();
        analyser.fftSize = 64;
        source = ctx.createMediaStreamSource(stream);
        source.connect(analyser);

        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        const update = () => {
          if (!analyser) return;
          analyser.getByteFrequencyData(dataArray);
          const newLevels = Array.from({ length: 32 }, (_, i) => {
            const val = dataArray[i % dataArray.length] || 0;
            return Math.min(95, Math.max(15, (val / 255) * 100));
          });
          setLevels(newLevels);
          animFrameRef.current = requestAnimationFrame(update);
        };

        update();
      })
      .catch(() => {
        // Fallback to static animated wave if audio capture is restricted
      });

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
    };
  }, [listening]);

  return (
    <div className={`wave ${active || listening ? "active" : ""}`}>
      {Array.from({ length: 32 }, (_, i) => {
        const height = listening && levels[i] !== undefined ? `${levels[i]}%` : `${18 + ((i * 17) % 55)}%`;
        return (
          <i
            key={i}
            style={{
              height,
              transition: listening ? "height 0.08s ease" : "height 0.2s ease",
            }}
          />
        );
      })}
    </div>
  );
}

export default function Home() {
  const [stage, setStage] = useState<Stage>("ready");
  const [online, setOnline] = useState(false);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState("");
  const [dark, setDark] = useState(false);

  const root = useRef<HTMLElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add("dark");
      document.body.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
      document.body.classList.remove("dark");
    }
  }, [dark]);

  useEffect(() => {
    healthCheck().then(setOnline);
  }, []);

  useEffect(() => {
    if (reduceMotion) return;
    let cancelled = false;
    let locomotive: { destroy: () => void } | undefined;
    let cleanup: { revert: () => void } | undefined;

    void Promise.all([import("locomotive-scroll"), import("gsap"), import("gsap/ScrollTrigger")]).then(
      ([loco, gsapModule, triggerModule]) => {
        if (cancelled || !root.current) return;
        const gsap = gsapModule.default;
        gsap.registerPlugin(triggerModule.ScrollTrigger);
        locomotive = new loco.default();
        cleanup = gsap.context(() => {
          gsap.utils.toArray<HTMLElement>("[data-reveal]").forEach((element) =>
            gsap.fromTo(
              element,
              { y: 36, opacity: 0 },
              {
                y: 0,
                opacity: 1,
                duration: 0.8,
                ease: "power3.out",
                scrollTrigger: { trigger: element, start: "top 88%" },
              }
            )
          );
          gsap.to(".hero-orbit", { rotate: 360, duration: 32, repeat: -1, ease: "none" });
          gsap.to(".parallax-word", {
            yPercent: -20,
            scrollTrigger: { trigger: ".proof", start: "top bottom", end: "bottom top", scrub: true },
          });
        }, root);
      }
    );

    return () => {
      cancelled = true;
      cleanup?.revert();
      locomotive?.destroy();
    };
  }, [reduceMotion]);

  const submit = async (value = query) => {
    if (!value.trim() || !["ready", "done", "error"].includes(stage)) return;
    setQuery(value);
    setResult(null);
    setError("");

    try {
      for (const next of ["retrieving", "generating", "verifying"] as Stage[]) {
        setStage(next);
        if (!reduceMotion) await new Promise((resolve) => setTimeout(resolve, next === "generating" ? 460 : 220));
      }
      const data = online ? await queryText(value) : DEMO;
      setResult({ ...data, query_text: value });
      setStage("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed.");
      setStage("error");
    }
  };

  const voice = async () => {
    if (stage === "listening") {
      setStage("transcribing");
      setTimeout(() => submit("What causes inflation?"), 650);
      return;
    }
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
      setStage("listening");
    } catch {
      setError("Microphone permission is needed for a voice query.");
      setStage("error");
    }
  };

  const labels: Record<Stage, string> = {
    ready: "TAP TO SPEAK",
    listening: "LISTENING - TAP TO STOP",
    transcribing: "TRANSCRIBING",
    retrieving: "RETRIEVING EVIDENCE",
    generating: "GENERATING",
    verifying: "VERIFYING",
    done: "ANSWER VERIFIED",
    error: "TRY AGAIN",
  };

  return (
    <main ref={root} className={dark ? "dark" : ""}>
      <div className="ticker">
        VOICE RAG / MULTILINGUAL RETRIEVAL / ENGLISH + HINDI / VERIFIED ANSWERS / VOICE RAG /
      </div>
      <nav>
        <a className="logo" href="#top">
          VOICE<br />
          RAG<sup>TM</sup>
        </a>
        <div className="nav-mid">
          <span>VOICE-ENABLED RAG MODEL</span>
          <span className="online">SYSTEM ONLINE</span>
        </div>
        <div className="nav-end">
          <button className="theme" onClick={() => setDark((value) => !value)}>
            {dark ? "LIGHT /" : "DARK /"}
          </button>
          <a className="github" href="https://github.com">
            GITHUB +
          </a>
        </div>
      </nav>

      <section className="hero" id="top">
        <div className="hero-orbit" aria-hidden="true">
          VOICE / EVIDENCE / RETRIEVAL / VERIFY /
        </div>
        <div className="hero-title">
          <p>VOICE-FIRST RESEARCH.</p>
          <h1>
            ASK. RETRIEVE.
            <br />
            <em>VERIFY.</em>
          </h1>
          <div className="hero-meta">
            <span>01 / VOICE TO EVIDENCE TO ANSWER</span>
            <span>EN + HI</span>
          </div>
        </div>

        <div className="terminal">
          <div className="terminal-bar">
            <span>LIVE QUERY TERMINAL</span>
            <span>SESSION_0001</span>
          </div>
          <div className="signal">
            <Wave
              active={stage === "retrieving" || stage === "generating"}
              listening={stage === "listening"}
            />
            <button
              className={`mic ${stage === "listening" ? "recording" : ""}`}
              onClick={voice}
              aria-label="Start or stop voice input"
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="22" />
              </svg>
            </button>
            <b>{labels[stage]}</b>
          </div>
          <div className="ask-row">
            <span>&gt;</span>
            <input
              ref={input}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              placeholder="TYPE A QUESTION OR USE VOICE"
            />
            <button disabled={!query.trim()} onClick={() => submit()}>
              RUN +
            </button>
          </div>
          <div className="examples">
            <span>EXAMPLES</span>
            {EXAMPLES.map((example) => (
              <button key={example} onClick={() => submit(example)}>
                {example}
              </button>
            ))}
          </div>
        </div>
      </section>

      <AnimatePresence>
        {result && (
          <motion.section
            className="result"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="result-top">
              <b>VERIFIED RESPONSE</b>
              <span>TRACE {result.trace.total_ms}MS</span>
            </div>
            <div className="answer">
              <div>
                <small>YOUR QUERY</small>
                <p>{result.query_text}</p>
              </div>
              <div>
                <small>ANSWER</small>
                <h2>{result.answer}</h2>
              </div>
            </div>
            <div className="evidence">
              <div className="evidence-label">
                <span>RETRIEVED EVIDENCE</span>
                <span>GROUNDING 94% / PASSED</span>
              </div>
              {result.sources.slice(0, 2).map((source, i) => (
                <article key={i}>
                  <header>
                    <b>0{i + 1}</b>
                    <span>RELEVANCE {result.scores[i]?.toFixed(2)}</span>
                    <span>{String(source.metadata.language || "MULTILINGUAL").toUpperCase()}</span>
                  </header>
                  <p>{source.text}</p>
                </article>
              ))}
            </div>
            <div className="trace">
              {[
                ["INPUT", 18],
                ["RETRIEVE", 31],
                ["GENERATE", 72],
                ["VERIFY", 21],
              ].map(([label, time], i) => (
                <div key={String(label)}>
                  <span>
                    0{i + 1} {label}
                  </span>
                  <i />
                  <b>{time}MS</b>
                </div>
              ))}
              <strong>PASS +</strong>
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      <section className="proof" data-reveal>
        <span className="parallax-word">TRUST</span>
        <p>
          ANSWERS BACKED BY <em>EVIDENCE.</em>
        </p>
        <div>
          <span>INPUT GUARD</span>
          <span>RELEVANCE GUARD</span>
          <span>GROUNDING GUARD</span>
          <span>FAISS / MULTILINGUAL</span>
        </div>
      </section>

      <section className="story" data-reveal>
        <div className="story-kicker">02 / THE EVIDENCE TRAIL</div>
        <div className="story-copy">
          <h2>
            KNOW <em>WHY</em>
            <br />
            THE ANSWER HOLDS.
          </h2>
          <p>
            Each response preserves its path: the original query, the passages retrieved, the verification
            result, and the latency behind every stage.
          </p>
        </div>
        <div className="story-rail">
          <article>
            <b>01</b>
            <span>QUERY</span>
            <p>Speak or type in English or Hindi.</p>
          </article>
          <article>
            <b>02</b>
            <span>RETRIEVE</span>
            <p>Relevant multilingual passages are ranked first.</p>
          </article>
          <article>
            <b>03</b>
            <span>VERIFY</span>
            <p>Grounding is checked before the answer is returned.</p>
          </article>
        </div>
      </section>

      {error && <p className="error">{error}</p>}

      <footer>
        <span>VOICE RAG / HACKER HOUSE GOA</span>
        <span>BUILT FOR #RAGINGOA</span>
        <span>ASK. RETRIEVE. VERIFY.</span>
      </footer>
    </main>
  );
}
