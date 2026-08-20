"use client";

import { useEffect, useRef, useState } from "react";
import type { PipelineResult, RequestTrace } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type BackendStatus = "checking" | "up" | "down";
type UiState = "idle" | "recording" | "processing" | "result" | "error";

interface SentenceEvent {
  event: "sentence";
  text: string;
}

interface DoneEvent {
  event: "done";
  result: PipelineResult;
}

function parseSseFrame(frame: string): { event: string; data: string } | null {
  let eventType = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) eventType = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  return data ? { event: eventType, data } : null;
}

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
  const [isStreaming, setIsStreaming] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

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

  const sendAudio = async (blob: Blob) => {
    setUiState("processing");
    setErrorMessage(null);
    setResult(null);
    setStreamingAnswer("");
    setIsStreaming(true);

    try {
      const formData = new FormData();
      const extension = blob.type.includes("webm") ? "webm" : blob.type.includes("ogg") ? "ogg" : "wav";
      formData.append("audio", blob, `recording.${extension}`);

      const res = await fetch(`${API_URL}/query/stream`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok || !res.body) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `Request failed with status ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let frameEnd;
        while ((frameEnd = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, frameEnd);
          buffer = buffer.slice(frameEnd + 2);
          const parsed = parseSseFrame(frame);
          if (!parsed) continue;

          if (parsed.event === "sentence") {
            const payload = JSON.parse(parsed.data) as SentenceEvent;
            setStreamingAnswer((prev) => (prev ? `${prev} ${payload.text}` : payload.text));
            setUiState("result");
          } else if (parsed.event === "done") {
            const payload = JSON.parse(parsed.data) as DoneEvent;
            setResult(payload.result);
            setIsStreaming(false);
            setUiState("result");
          }
        }
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Something went wrong while contacting the backend.");
      setUiState("error");
    } finally {
      setIsStreaming(false);
    }
  };

  const startRecording = async () => {
    setErrorMessage(null);
    if (typeof MediaRecorder === "undefined") {
      setErrorMessage("This browser doesn't support audio recording (MediaRecorder API unavailable).");
      setUiState("error");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        const blob = new Blob(chunksRef.current, { type: mediaRecorder.mimeType || "audio/webm" });
        sendAudio(blob);
      };

      mediaRecorder.start();
      setUiState("recording");
    } catch {
      setErrorMessage("Microphone access was denied or is unavailable.");
      setUiState("error");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
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
                <tr>
                  <td>Speech-to-text</td>
                  <td>{formatMs(spanTotal(result.trace, "stt_network"))}</td>
                </tr>
                <tr>
                  <td>Embedding</td>
                  <td>{formatMs(spanTotal(result.trace, "embedding_cache", "embedding_compute"))}</td>
                </tr>
                <tr>
                  <td>Retrieval</td>
                  <td>{formatMs(spanTotal(result.trace, "vector_search", "bm25", "fusion", "reranking", "retrieval_overhead"))}</td>
                </tr>
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
                  <td>RAG total</td>
                  <td>{formatMs(ragTotalMs(result.trace))}</td>
                </tr>
                <tr className="latencyTotalRow">
                  <td>Full voice total</td>
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
