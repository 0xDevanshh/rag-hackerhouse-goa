"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { PipelineResult, RequestTrace } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const MOTION_MS = 200;
const AUDIO_BUFFER_SIZE = 512;

type BackendStatus = "checking" | "up" | "down";
type UiState = "idle" | "recording" | "processing" | "result" | "error";

/**
 * Total duration of the named spans, or null if none of them ran.
 *
 * null and 0 are deliberately different: "this stage never ran" (a cache hit
 * skips retrieval entirely) renders as "—", while "ran and cost nothing
 * measurable" renders as "0 ms".
 *
 * Defensive about the shape on purpose. This component is rendered from a
 * payload owned by a separate backend, and a renamed field previously took the
 * whole page down with an uncaught TypeError rather than degrading one table
 * cell. A missing span list should cost a dash, not the app.
 */
function spanTotal(trace: RequestTrace | undefined, ...names: string[]): number | null {
  const spans = trace?.spans;
  if (!Array.isArray(spans)) return null;
  const matched = spans.filter((s) => names.includes(s.name));
  return matched.length ? matched.reduce((sum, s) => sum + s.duration_ms, 0) : null;
}

/** A value from the trace's overlapping aggregates (e.g. `llm_ttft`). */
function detail(trace: RequestTrace | undefined, name: string): number | null {
  const value = trace?.details?.[name];
  return typeof value === "number" ? value : null;
}

/**
 * The measured wall clock, taken from the server rather than summed here — see
 * RequestTrace in ./types. Summing spans would quietly drop whatever the
 * backend didn't instrument.
 */
function totalMs(trace: RequestTrace | undefined): number | null {
  return typeof trace?.total_ms === "number" ? trace.total_ms : null;
}

function ragTotalMs(trace: RequestTrace | undefined): number | null {
  return spanTotal(
    trace,
    "query_preprocessing",
    "embedding_cache",
    "embedding_compute",
    "vector_search",
    "bm25",
    "fusion",
    "reranking",
    "retrieval_overhead",
    "relevance_guard",
    "context_build",
    "llm_network",
    "llm_client_wait",
    "llm_generation",
    "llm_retry_wait",
    "grounding_guard",
  );
}

function formatMs(ms: number | null): string {
  return ms === null ? "—" : `${ms.toFixed(0)} ms`;
}

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [uiState, setUiState] = useState<UiState>("idle");
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [streamingAnswer, setStreamingAnswer] = useState("");
  const [liveTranscript, setLiveTranscript] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const realtimeSocketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const sttTimingRef = useRef({
    audioAvailableMs: null as number | null,
    firstAudioSentMs: null as number | null,
    firstPartialReceivedMs: null as number | null,
    firstPartialText: "",
  });

  const sendPcmFrame = (samples: Float32Array, inputRate: number) => {
    const now = performance.now();
    if (sttTimingRef.current.audioAvailableMs === null) {
      sttTimingRef.current.audioAvailableMs = now;
    }
    const ratio = inputRate / 16000;
    const outputLength = Math.floor(samples.length / ratio);
    const pcm = new Int16Array(outputLength);
    for (let i = 0; i < outputLength; i += 1) {
      const sample = Math.max(-1, Math.min(1, samples[Math.floor(i * ratio)]));
      pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    let binary = "";
    const bytes = new Uint8Array(pcm.buffer);
    for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
    const socket = realtimeSocketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ event: "audio_input", audio: btoa(binary) }));
      if (sttTimingRef.current.firstAudioSentMs === null) {
        sttTimingRef.current.firstAudioSentMs = performance.now();
      }
    }
  };

  const finishRealtimeAudio = () => {
    workletRef.current?.disconnect();
    audioSourceRef.current?.disconnect();
    audioContextRef.current?.close();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    workletRef.current = null;
    audioSourceRef.current = null;
    audioContextRef.current = null;
  };

  const submitTranscript = async (transcript: string) => {
    // Route through the fast extractive path by default. The generative
    // endpoint (/query/text) calls the remote LLM and grounding guardrail,
    // adding 700-1300ms that is not reducible by any local optimization.
    // /query/realtime/text uses MockRealtimeSTT + ExtractiveProvider: the
    // same embedding/FAISS/BM25 pipeline, zero network calls, ~30-50ms.
    const res = await fetch(`${API_URL}/query/realtime/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: transcript }),
    });
    if (!res.ok) throw new Error(`RAG request failed with status ${res.status}`);
    setResult((await res.json()) as PipelineResult);
    setUiState("result");
  };

  const pingBackend = async () => {
    try {
      const res = await fetch(`${API_URL}/health`);
      setBackendStatus(res.ok ? "up" : "down");
    } catch {
      setBackendStatus("down");
    }
  };

  const retryBackendCheck = () => {
    setBackendStatus("checking");
    pingBackend();
  };

  useEffect(() => {
    let ignore = false;

    async function checkOnMount() {
      try {
        const res = await fetch(`${API_URL}/health`);
        if (!ignore) setBackendStatus(res.ok ? "up" : "down");
      } catch {
        if (!ignore) setBackendStatus("down");
      }
    }

    checkOnMount();
    return () => {
      ignore = true;
    };
  }, []);

  useLayoutEffect(() => {
    const timing = sttTimingRef.current;
    if (timing.firstPartialReceivedMs !== null && liveTranscript === timing.firstPartialText) {
      const renderedMs = performance.now();
      console.info("stt_first_partial_latency", {
        capture_to_render_ms: renderedMs - (timing.audioAvailableMs ?? renderedMs),
        capture_to_send_ms: (timing.firstAudioSentMs ?? renderedMs) - (timing.audioAvailableMs ?? renderedMs),
        send_to_receive_ms: timing.firstPartialReceivedMs - (timing.firstAudioSentMs ?? timing.firstPartialReceivedMs),
        receive_to_render_ms: renderedMs - timing.firstPartialReceivedMs,
      });
      timing.firstPartialText = "";
    }
  }, [liveTranscript]);

  const startRecording = async () => {
    setErrorMessage(null);
    if (typeof AudioContext === "undefined" || typeof WebSocket === "undefined") {
      setErrorMessage("This browser doesn't support realtime audio capture.");
      setUiState("error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setLiveTranscript("");
      sttTimingRef.current = {
        audioAvailableMs: null,
        firstAudioSentMs: null,
        firstPartialReceivedMs: null,
        firstPartialText: "",
      };
      const configResponse = await fetch(`${API_URL}/stt/realtime/config`);
      const realtimeConfig = configResponse.ok
        ? (await configResponse.json()) as {
            direct?: boolean;
            url?: string;
            protocol?: string;
            query?: Record<string, string>;
          }
        : null;
      const socketUrl = realtimeConfig?.direct && realtimeConfig.url && realtimeConfig.query
        ? `${realtimeConfig.url}?${new URLSearchParams(realtimeConfig.query)}`
        : `${API_URL.replace(/^http/, "ws")}/stt/realtime`;
      const socket = realtimeConfig?.direct && realtimeConfig.protocol
        ? new WebSocket(socketUrl, realtimeConfig.protocol)
        : new WebSocket(socketUrl);
      realtimeSocketRef.current = socket;
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data) as { event?: string; text?: string };
        if (message.event === "transcript.partial" && message.text) {
          if (sttTimingRef.current.firstPartialReceivedMs === null) {
            sttTimingRef.current.firstPartialReceivedMs = performance.now();
            sttTimingRef.current.firstPartialText = message.text;
          }
          setLiveTranscript(message.text);
        }
        if (message.event === "transcript.final" && message.text) {
          setLiveTranscript(message.text);
          finishRealtimeAudio();
          socket.close();
          submitTranscript(message.text).catch((error: unknown) => {
            setErrorMessage(error instanceof Error ? error.message : "Could not answer the transcript.");
            setUiState("error");
          });
        }
      };
      socket.onerror = () => {
        finishRealtimeAudio();
        setErrorMessage("Realtime speech recognition is unavailable.");
        setUiState("error");
      };
      await new Promise<void>((resolve, reject) => {
        socket.addEventListener("open", () => resolve(), { once: true });
        socket.addEventListener("error", () => reject(new Error("Realtime speech connection failed.")), { once: true });
      });
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      if (!audioContext.audioWorklet) throw new Error("AudioWorklet is required for realtime capture.");
      await audioContext.audioWorklet.addModule("/audio-capture-worklet.js");
      const worklet = new AudioWorkletNode(audioContext, "audio-capture-processor");
      worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
        sendPcmFrame(event.data, audioContext.sampleRate);
      };
      source.connect(worklet);
      // Keep the graph active without monitoring the microphone. The worklet
      // emits silent output, so captured audio cannot echo through speakers.
      worklet.connect(audioContext.destination);
      workletRef.current = worklet;
      audioContextRef.current = audioContext;
      audioSourceRef.current = source;
      requestAnimationFrame(() => {
        setUiState("recording");
      });
    } catch {
      setErrorMessage("Microphone access was denied or is unavailable.");
      setUiState("error");
    }
  };

  const stopRecording = () => {
    finishRealtimeAudio();
    realtimeSocketRef.current?.send(JSON.stringify({ event: "end" }));
    window.setTimeout(() => {
      setUiState("processing");
    }, MOTION_MS);
  };

  const handleRecordClick = () => {
    if (uiState === "recording") {
      stopRecording();
    } else {
      setResult(null);
      startRecording();
    }
  };

  const reset = () => {
    setResult(null);
    setStreamingAnswer("");
    setLiveTranscript("");
    setIsStreaming(false);
    setErrorMessage(null);
    setUiState("idle");
  };

  const backendDown = backendStatus === "down";

  return (
    <div className="container">
      <header className="header">
        <div>
          <h1>Voice RAG</h1>
          <p className="subtitle">Ask a question out loud, grounded in the document corpus.</p>
        </div>
        <span className="statusPill">
          <span className={`statusDot ${backendStatus === "up" ? "up" : backendStatus === "down" ? "down" : ""}`} />
          {backendStatus === "checking" && "Checking backend…"}
          {backendStatus === "up" && "Backend online"}
          {backendStatus === "down" && "Backend offline"}
        </span>
      </header>

      {backendDown && (
        <div className="backendDownBanner">
          <strong>Backend not running.</strong>
          <span>
            Couldn&apos;t reach {API_URL}. Start the FastAPI server (e.g. <code>python src/api.py</code>) and try again.
          </span>
          <button onClick={retryBackendCheck}>Retry connection</button>
        </div>
      )}

      <div className="recorderArea">
        <button
          className={`recordButton ${uiState === "recording" ? "recording" : ""}`}
          onClick={handleRecordClick}
          disabled={backendDown || uiState === "processing"}
        >
          {uiState === "recording" ? "Stop" : "Record"}
        </button>

        <div className="stateLabel">
          {uiState === "idle" && "Tap to ask a question"}
          {uiState === "recording" && "Listening… tap to stop"}
          {uiState === "processing" && <span className="spinner" aria-label="Processing" />}
          {uiState === "result" && !isStreaming && "Done — tap Record to ask another question"}
          {uiState === "result" && isStreaming && "Generating…"}
          {uiState === "error" && "Tap Record to try again"}
        </div>
      </div>

      {uiState === "recording" && liveTranscript && (
        <div className="card">
          <h2>Live transcript</h2>
          <p className="queryText">{liveTranscript}</p>
        </div>
      )}

      {uiState === "error" && errorMessage && (
        <div className="errorBox">
          <strong>Something went wrong.</strong>
          <p>{errorMessage}</p>
        </div>
      )}

      {uiState === "result" && !result && (
        <div className="resultSection">
          <div className="card">
            <h2>
              Answer
              {isStreaming && <span className="spinner streamingSpinner" aria-label="Generating" />}
            </h2>
            <p className="answerText">{streamingAnswer || "…"}</p>
          </div>
        </div>
      )}

      {uiState === "result" && result && (
        <div className="resultSection">
          <div className="card">
            <h2>Transcribed query</h2>
            <p className="queryText">&ldquo;{result.query_text}&rdquo;</p>
          </div>

          <div className="card">
            <h2>
              Answer
              {result.degraded && <span className="badge">degraded</span>}
              {result.cached && <span className="badge cached">cached</span>}
            </h2>
            <p className="answerText">{result.answer}</p>
          </div>

          <div className="card">
            <h2>Top passages</h2>
            <ul className="passageList">
              {result.sources.slice(0, 3).map((chunk, i) => (
                <li className="passageItem" key={i}>
                  <div className="passageScore">score: {result.scores[i] !== undefined ? result.scores[i].toFixed(3) : "—"}</div>
                  <div className="passageText">{chunk.text}</div>
                </li>
              ))}
              {result.sources.length === 0 && <li className="passageText">No passages were retrieved.</li>}
            </ul>
          </div>

          <div className="card">
            <h2>Latency breakdown</h2>
            <table className="latencyTable">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {/* ── Serial STT path (/query) ── */}
                {spanTotal(result.trace, "stt_network") !== null && (
                  <tr>
                    <td>Speech-to-text (batch)</td>
                    <td>{formatMs(spanTotal(result.trace, "stt_network"))}</td>
                  </tr>
                )}
                {/* ── Overlapped STT path (/query/realtime) ── */}
                {spanTotal(result.trace, "stt_final") !== null && (
                  <>
                    <tr>
                      <td>
                        STT → first partial
                        <span className="stageNote">time until first transcript.partial</span>
                      </td>
                      <td>{formatMs(spanTotal(result.trace, "stt_to_first_partial"))}</td>
                    </tr>
                    <tr>
                      <td>
                        STT total
                        <span className="stageNote">stream open → transcript.final</span>
                      </td>
                      <td>{formatMs(spanTotal(result.trace, "stt_final"))}</td>
                    </tr>
                    <tr>
                      <td>
                        Retrieval (on stable partial)
                        <span className="stageNote">ran concurrently with STT</span>
                      </td>
                      <td>{formatMs(spanTotal(result.trace, "retrieval_on_partial"))}</td>
                    </tr>
                    {spanTotal(result.trace, "stt_overlap_savings") !== null && (
                      <tr className="overlapSavingsRow">
                        <td>
                          ⚡ Overlap savings
                          <span className="stageNote">retrieval ms hidden inside STT</span>
                        </td>
                        <td>−{formatMs(spanTotal(result.trace, "stt_overlap_savings"))}</td>
                      </tr>
                    )}
                  </>
                )}
                {/* ── Common stages ── */}
                <tr>
                  <td>Embedding</td>
                  <td>{formatMs(spanTotal(result.trace, "embedding_cache", "embedding_compute"))}</td>
                </tr>
                {spanTotal(result.trace, "stt_final") === null && (
                  <tr>
                    <td>Retrieval</td>
                    <td>{formatMs(spanTotal(result.trace, "vector_search", "bm25", "fusion", "reranking", "retrieval_overhead"))}</td>
                  </tr>
                )}
                <tr>
                  <td>BM25 lexical search</td>
                  <td>{formatMs(spanTotal(result.trace, "bm25"))}</td>
                </tr>
                <tr>
                  <td>RRF fusion</td>
                  <td>{formatMs(spanTotal(result.trace, "fusion"))}</td>
                </tr>
                <tr>
                  <td>Guardrails</td>
                  <td>
                    {formatMs(spanTotal(result.trace, "query_preprocessing", "relevance_guard", "grounding_guard"))}
                  </td>
                </tr>
                <tr>
                  <td>
                    LLM<span className="stageNote">first token at {formatMs(detail(result.trace, "llm_ttft"))}</span>
                  </td>
                  <td>
                    {formatMs(
                      spanTotal(result.trace, "llm_network", "llm_client_wait", "llm_generation", "llm_retry_wait"),
                    )}
                  </td>
                </tr>
                <tr>
                  <td>
                    Server overhead
                    <span className="stageNote">middleware, body parse, serialization, flush</span>
                  </td>
                  <td>
                    {formatMs(
                      spanTotal(result.trace, "middleware", "body_parse", "serialization", "response_write"),
                    )}
                  </td>
                </tr>
                <tr>
                  <td>
                    Unaccounted<span className="stageNote">wall clock no stage claimed</span>
                  </td>
                  <td>
                    {formatMs(
                      typeof result.trace?.unaccounted_ms === "number" ? result.trace.unaccounted_ms : null,
                    )}
                  </td>
                </tr>
                <tr className="latencyTotalRow">
                  <td>Processing (excl. speech recognition)</td>
                  <td>{formatMs(ragTotalMs(result.trace))}</td>
                </tr>
                <tr className="latencyTotalRow">
                  <td>Full end-to-end total</td>
                  <td>{formatMs(totalMs(result.trace))}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <button className="retryButton" onClick={reset}>
            Ask another question
          </button>
        </div>
      )}
    </div>
  );
}
