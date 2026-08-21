"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import MicButton from "./MicButton";
import Waveform from "./Waveform";
import PipelineViz from "./PipelineViz";
import type { UiPhase, FrontendResult, PipelineStage } from "../lib/types";
import { buildStages } from "../lib/pipeline-stages";
import { queryText, fetchRealtimeSttConfig, API_URL, isAbortError } from "../lib/api";
import { normalizeResult } from "../lib/normalize";
import type { PipelineResult } from "../lib/types";

interface VoiceTerminalProps {
  isOffline: boolean;
  onResult: (result: FrontendResult) => void;
  onPhaseChange?: (phase: UiPhase) => void;
  requestCount: number;
  onRequestCountChange: (n: number) => void;
}

const PHASE_LABELS: Record<UiPhase, string> = {
  IDLE:         "READY",
  RECORDING:    "LISTENING",
  UPLOADING:    "UPLOADING",
  TRANSCRIBING: "TRANSCRIBING",
  RETRIEVING:   "RETRIEVING",
  GROUNDING:    "VERIFYING",
  GENERATING:   "GENERATING",
  COMPLETE:     "ANSWER READY",
  DEGRADED:     "DEGRADED",
  ERROR:        "ERROR",
  OFFLINE_DEMO: "DEMO MODE",
};

const EXAMPLE_QUERIES = [
  "What causes inflation?",
  "मुद्रास्फीति क्या है?",
  "What is the full form of AC?",
  "How does hybrid retrieval work?",
];

export default function VoiceTerminal({
  isOffline,
  onResult,
  onPhaseChange,
  requestCount,
  onRequestCountChange,
}: VoiceTerminalProps) {
  const [phase, setPhase] = useState<UiPhase>("IDLE");
  const [textQuery, setTextQuery] = useState("");
  const [liveTranscript, setLiveTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [stages, setStages] = useState<PipelineStage[]>(() => buildStages("IDLE"));
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);

  // Refs
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sttTimingRef = useRef({
    audioAvailableMs: null as number | null,
    firstAudioSentMs: null as number | null,
    firstPartialReceivedMs: null as number | null,
  });
  // Stable SSR placeholder; replaced with a random value on the client only,
  // so the server-rendered HTML and the initial client render always match.
  const [sessionId, setSessionId] = useState("------");
  useEffect(() => {
    setSessionId(Math.random().toString(36).slice(2, 8).toUpperCase());
  }, []);

  const updatePhase = useCallback((p: UiPhase) => {
    setPhase(p);
    setStages(buildStages(p));
    onPhaseChange?.(p);
  }, [onPhaseChange]);

  // Cleanup audio resources
  const cleanupAudio = useCallback(() => {
    workletRef.current?.disconnect();
    audioSourceRef.current?.disconnect();
    analyserRef.current?.disconnect();
    audioCtxRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    workletRef.current = null;
    audioSourceRef.current = null;
    analyserRef.current = null;
    audioCtxRef.current = null;
    streamRef.current = null;
    setAnalyser(null);

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setRecordingSeconds(0);
  }, []);

  useEffect(() => () => {
    cleanupAudio();
    abortRef.current?.abort();
    wsRef.current?.close();
  }, [cleanupAudio]);

  // ── Submit transcript / text query to backend ─────────────────────────
  const submitQuery = useCallback(async (q: string) => {
    if (!q.trim()) return;

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    onRequestCountChange(requestCount + 1);
    updatePhase("RETRIEVING");

    try {
      const raw: PipelineResult = await queryText(q, ctrl.signal);
      updatePhase("GROUNDING");
      // Brief pause so the grounding stage is visible
      await new Promise((r) => setTimeout(r, 80));
      updatePhase("COMPLETE");
      const normalized = normalizeResult(raw, "VOICE");
      onResult(normalized);
    } catch (err) {
      if (isAbortError(err)) return;
      updatePhase("ERROR");
      setError(
        err instanceof Error
          ? err.message
          : "Query failed. Backend may be unavailable."
      );
    }
  }, [requestCount, onRequestCountChange, updatePhase, onResult]);

  // ── Send PCM frame to WS ──────────────────────────────────────────────
  const sendPcmFrame = useCallback((samples: Float32Array, inputRate: number) => {
    const now = performance.now();
    if (sttTimingRef.current.audioAvailableMs === null) {
      sttTimingRef.current.audioAvailableMs = now;
    }
    const ratio = inputRate / 16000;
    const outputLen = Math.floor(samples.length / ratio);
    const pcm = new Int16Array(outputLen);
    for (let i = 0; i < outputLen; i++) {
      const s = Math.max(-1, Math.min(1, samples[Math.floor(i * ratio)]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    let binary = "";
    const bytes = new Uint8Array(pcm.buffer);
    for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ event: "audio_input", audio: btoa(binary) }));
      if (sttTimingRef.current.firstAudioSentMs === null) {
        sttTimingRef.current.firstAudioSentMs = performance.now();
      }
    }
  }, []);

  // ── Stop recording / finalize audio ──────────────────────────────────
  const stopRecording = useCallback(() => {
    cleanupAudio();
    wsRef.current?.send(JSON.stringify({ event: "end" }));
    updatePhase("TRANSCRIBING");
  }, [cleanupAudio, updatePhase]);

  // ── Start recording ───────────────────────────────────────────────────
  const startRecording = useCallback(async () => {
    if (typeof AudioContext === "undefined" || typeof WebSocket === "undefined") {
      setError("This browser doesn't support realtime audio capture.");
      updatePhase("ERROR");
      return;
    }

    setError(null);
    setLiveTranscript("");
    sttTimingRef.current = { audioAvailableMs: null, firstAudioSentMs: null, firstPartialReceivedMs: null };

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Fetch STT config (may return null if SARVAM_API_KEY not configured)
      const sttConfig = await fetchRealtimeSttConfig();

      const wsUrl = sttConfig?.direct && sttConfig.url && sttConfig.query
        ? `${sttConfig.url}?${new URLSearchParams(sttConfig.query)}`
        : `${API_URL.replace(/^http/, "ws")}/stt/realtime`;

      const ws = sttConfig?.direct && sttConfig.protocol
        ? new WebSocket(wsUrl, sttConfig.protocol)
        : new WebSocket(wsUrl);

      wsRef.current = ws;

      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data) as { event?: string; text?: string };
        if (msg.event === "transcript.partial" && msg.text) {
          if (sttTimingRef.current.firstPartialReceivedMs === null) {
            sttTimingRef.current.firstPartialReceivedMs = performance.now();
          }
          setLiveTranscript(msg.text);
        }
        if (msg.event === "transcript.final" && msg.text) {
          setLiveTranscript(msg.text);
          cleanupAudio();
          ws.close();
          submitQuery(msg.text);
        }
      };

      ws.onerror = () => {
        cleanupAudio();
        setError("Realtime speech recognition is unavailable. Use text input.");
        updatePhase("ERROR");
      };

      // Wait for WS open
      await new Promise<void>((resolve, reject) => {
        ws.addEventListener("open", () => resolve(), { once: true });
        ws.addEventListener("error", () => reject(new Error("STT connection failed")), { once: true });
      });

      // Set up AudioContext + AnalyserNode + Worklet
      const audioCtx = new AudioContext({ sampleRate: 16000 });
      const analyserNode = audioCtx.createAnalyser();
      analyserNode.fftSize = 256;
      const source = audioCtx.createMediaStreamSource(stream);

      if (!audioCtx.audioWorklet) throw new Error("AudioWorklet required");
      await audioCtx.audioWorklet.addModule("/audio-capture-worklet.js");
      const worklet = new AudioWorkletNode(audioCtx, "audio-capture-processor");

      worklet.port.onmessage = (e: MessageEvent<Float32Array>) => {
        sendPcmFrame(e.data, audioCtx.sampleRate);
      };

      source.connect(analyserNode);
      source.connect(worklet);
      worklet.connect(audioCtx.destination);

      audioCtxRef.current = audioCtx;
      audioSourceRef.current = source;
      analyserRef.current = analyserNode;
      workletRef.current = worklet;

      setAnalyser(analyserNode);

      // Recording timer
      timerRef.current = setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);

      updatePhase("RECORDING");
    } catch {
      cleanupAudio();
      setError("Microphone access was denied or is unavailable. Use text input below.");
      updatePhase("ERROR");
    }
  }, [cleanupAudio, sendPcmFrame, submitQuery, updatePhase]);

  // ── Handle mic button ─────────────────────────────────────────────────
  const handleMicClick = useCallback(() => {
    if (phase === "RECORDING") {
      stopRecording();
    } else if (!isOffline) {
      setStages(buildStages("IDLE"));
      updatePhase("IDLE");
      startRecording();
    }
  }, [phase, isOffline, stopRecording, startRecording, updatePhase]);

  // ── Handle text submit ────────────────────────────────────────────────
  const handleTextSubmit = useCallback(async (q: string) => {
    if (!q.trim() || isOffline) return;
    setTextQuery("");
    setLiveTranscript("");
    updatePhase("RETRIEVING");

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    onRequestCountChange(requestCount + 1);

    try {
      const raw = await queryText(q, ctrl.signal);
      updatePhase("GROUNDING");
      await new Promise((r) => setTimeout(r, 80));
      updatePhase("COMPLETE");
      const normalized = normalizeResult(raw, "TEXT");
      onResult(normalized);
    } catch (err) {
      if (isAbortError(err)) return;
      updatePhase("ERROR");
      setError(err instanceof Error ? err.message : "Query failed.");
    }
  }, [isOffline, updatePhase, onRequestCountChange, requestCount, onResult]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleTextSubmit(textQuery);
    }
  }, [textQuery, handleTextSubmit]);

  const micState =
    phase === "RECORDING"   ? "RECORDING" :
    phase === "TRANSCRIBING" || phase === "RETRIEVING" || phase === "GROUNDING" || phase === "GENERATING" || phase === "UPLOADING"
      ? "PROCESSING" :
    phase === "COMPLETE"    ? "SUCCESS" :
    phase === "ERROR"       ? "ERROR" :
    "IDLE";

  const isProcessing = ["UPLOADING", "TRANSCRIBING", "RETRIEVING", "GROUNDING", "GENERATING"].includes(phase);

  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: "2px",
        overflow: "hidden",
      }}
    >
      {/* Terminal header */}
      <div
        style={{
          background: "var(--bg-secondary)",
          borderBottom: "1px solid var(--border)",
          padding: "0.5rem 1rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.75rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.625rem",
              fontWeight: 500,
              letterSpacing: "0.12em",
              color: "var(--fg-muted)",
            }}
          >
            LIVE QUERY TERMINAL
          </span>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.5rem",
              color: "var(--border-soft)",
            }}
          >
            SESSION_{sessionId}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.5rem",
              letterSpacing: "0.1em",
              color: "var(--fg-muted)",
            }}
          >
            REQUEST_{String(requestCount).padStart(4, "0")}
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
            <span
              className="status-dot"
              style={{
                background: phase === "RECORDING" ? "var(--accent2)" :
                  phase === "COMPLETE" ? "var(--success)" :
                  phase === "ERROR" ? "var(--danger)" :
                  isProcessing ? "var(--accent)" : "var(--success)",
              }}
            />
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.5625rem",
                letterSpacing: "0.08em",
                color: "var(--fg-muted)",
              }}
            >
              {PHASE_LABELS[phase]}
            </span>
          </div>
        </div>
      </div>

      {/* Main interactive area */}
      <div
        style={{
          padding: "2rem 1.5rem",
          display: "flex",
          flexDirection: "column",
          gap: "1.5rem",
        }}
      >
        {/* Mic + waveform */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "1rem",
          }}
        >
          <MicButton
            state={micState}
            disabled={isOffline || isProcessing}
            onClick={handleMicClick}
            recordingSeconds={recordingSeconds}
          />

          {/* Waveform */}
          <div
            style={{
              width: "100%",
              maxWidth: "320px",
              opacity: phase === "RECORDING" ? 1 : 0.3,
              transition: "opacity var(--duration-med)",
            }}
          >
            <Waveform
              isActive={phase === "RECORDING"}
              analyserNode={analyser}
              color="var(--accent2)"
            />
          </div>

          {/* Phase status */}
          <AnimatePresence mode="wait">
            <motion.div
              key={phase}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.15 }}
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.6875rem",
                letterSpacing: "0.12em",
                color: phase === "ERROR" ? "var(--danger)" :
                  phase === "COMPLETE" ? "var(--success)" :
                  "var(--fg-muted)",
                textAlign: "center",
              }}
              aria-live="polite"
            >
              {phase === "RECORDING" && recordingSeconds > 0
                ? `LISTENING  ${recordingSeconds}s`
                : PHASE_LABELS[phase]}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Live transcript */}
        <AnimatePresence>
          {liveTranscript && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              style={{
                padding: "0.625rem 0.875rem",
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-soft)",
                borderRadius: "2px",
                overflow: "hidden",
              }}
            >
              <div className="label-mono" style={{ marginBottom: "0.25rem" }}>
                TRANSCRIBED QUERY
              </div>
              <p
                style={{
                  fontFamily: "var(--font-body)",
                  fontSize: "0.9rem",
                  color: "var(--fg)",
                  fontStyle: "italic",
                }}
                aria-live="polite"
              >
                &ldquo;{liveTranscript}&rdquo;
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Error display */}
        <AnimatePresence>
          {error && phase === "ERROR" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                padding: "0.75rem",
                border: "1px solid var(--danger)",
                borderRadius: "2px",
                background: "var(--danger-bg)",
              }}
              role="alert"
            >
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.625rem",
                  fontWeight: 500,
                  letterSpacing: "0.1em",
                  color: "var(--danger)",
                  marginBottom: "0.25rem",
                }}
              >
                {error.includes("Microphone") ? "MIC UNAVAILABLE" :
                 error.includes("speech")      ? "STT UNAVAILABLE" :
                 error.includes("backend") || error.includes("Backend") ? "BACKEND ERROR" :
                 "ERROR"}
              </div>
              <p style={{ fontFamily: "var(--font-body)", fontSize: "0.8125rem", color: "var(--fg)" }}>
                {error}
              </p>
              <button
                className="btn btn-ghost"
                onClick={() => { setError(null); updatePhase("IDLE"); }}
                style={{ marginTop: "0.5rem", fontSize: "0.625rem" }}
                aria-label="Dismiss error and retry"
              >
                DISMISS
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Divider */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
          }}
        >
          <div style={{ flex: 1, height: "1px", background: "var(--border-soft)" }} />
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.5625rem",
              letterSpacing: "0.1em",
              color: "var(--fg-muted)",
            }}
          >
            OR TYPE
          </span>
          <div style={{ flex: 1, height: "1px", background: "var(--border-soft)" }} />
        </div>

        {/* Text input */}
        <div style={{ position: "relative" }}>
          <div
            style={{
              position: "absolute",
              left: "0.75rem",
              top: "0.6875rem",
              fontFamily: "var(--font-mono)",
              fontSize: "0.875rem",
              color: "var(--fg-muted)",
              pointerEvents: "none",
              userSelect: "none",
            }}
            aria-hidden="true"
          >
            &gt;
          </div>
          <textarea
            value={textQuery}
            onChange={(e) => setTextQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="TYPE A QUESTION OR USE VOICE"
            rows={2}
            disabled={isOffline || isProcessing}
            aria-label="Type your query here"
            style={{
              width: "100%",
              padding: "0.625rem 4rem 0.625rem 1.75rem",
              background: "var(--bg-secondary)",
              border: "1px solid var(--border)",
              borderRadius: "2px",
              color: "var(--fg)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.8125rem",
              resize: "none",
              outline: "none",
              lineHeight: 1.5,
            }}
            onFocus={(e) => (e.target.style.borderColor = "var(--accent)")}
            onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
          />
          <button
            onClick={() => handleTextSubmit(textQuery)}
            disabled={!textQuery.trim() || isOffline || isProcessing}
            className="btn btn-accent"
            aria-label="Submit text query"
            style={{
              position: "absolute",
              right: "0.5rem",
              top: "50%",
              transform: "translateY(-50%)",
              fontSize: "0.625rem",
              padding: "0.375rem 0.625rem",
              opacity: !textQuery.trim() || isProcessing ? 0.4 : 1,
            }}
          >
            RUN +
          </button>
        </div>

        {/* Example queries */}
        <div style={{ display: "flex", gap: "0.375rem", flexWrap: "wrap" }}>
          {EXAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              onClick={() => !isProcessing && !isOffline && handleTextSubmit(q)}
              disabled={isOffline || isProcessing}
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: "0.5625rem",
                padding: "0.25rem 0.5rem",
                border: "1px solid var(--border-soft)",
                borderRadius: "2px",
                background: "transparent",
                color: "var(--fg-muted)",
                cursor: isProcessing ? "not-allowed" : "pointer",
                transition: "color var(--duration-fast), border-color var(--duration-fast)",
              }}
              onMouseEnter={(e) => {
                if (!isProcessing) {
                  e.currentTarget.style.color = "var(--fg)";
                  e.currentTarget.style.borderColor = "var(--border)";
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.color = "var(--fg-muted)";
                e.currentTarget.style.borderColor = "var(--border-soft)";
              }}
              aria-label={`Try example: ${q}`}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Pipeline visualization */}
      <div
        style={{
          borderTop: "1px solid var(--border)",
          padding: "0.75rem 1rem",
          background: "var(--bg-secondary)",
        }}
      >
        <div className="label-mono" style={{ marginBottom: "0.5rem" }}>
          PIPELINE
        </div>
        <div className="pipeline-scroll">
          <PipelineViz stages={stages} />
        </div>
      </div>
    </div>
  );
}
