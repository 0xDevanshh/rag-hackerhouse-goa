"use client";

import { useEffect, useRef, useState } from "react";
import type { LatencyTrace, PipelineResult } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type BackendStatus = "checking" | "up" | "down";
type UiState = "idle" | "recording" | "processing" | "result" | "error";

function getStageDuration(trace: LatencyTrace, stage: string): number | null {
  const match = trace.stages.find((s) => s.stage === stage);
  return match ? match.duration_ms : null;
}

function getTotalDuration(trace: LatencyTrace): number {
  return trace.stages.reduce((sum, s) => sum + s.duration_ms, 0);
}

function formatMs(ms: number | null): string {
  return ms === null ? "—" : `${ms.toFixed(0)} ms`;
}

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [uiState, setUiState] = useState<UiState>("idle");
  const [result, setResult] = useState<PipelineResult | null>(null);
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
    try {
      const formData = new FormData();
      const extension = blob.type.includes("webm") ? "webm" : blob.type.includes("ogg") ? "ogg" : "wav";
      formData.append("audio", blob, `recording.${extension}`);

      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `Request failed with status ${res.status}`);
      }

      const data: PipelineResult = await res.json();
      setResult(data);
      setUiState("result");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Something went wrong while contacting the backend.");
      setUiState("error");
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
          {uiState === "result" && "Done — tap Record to ask another question"}
          {uiState === "error" && "Tap Record to try again"}
        </div>
      </div>

      {uiState === "error" && errorMessage && (
        <div className="errorBox">
          <strong>Something went wrong.</strong>
          <p>{errorMessage}</p>
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
                  <td>STT</td>
                  <td>{formatMs(getStageDuration(result.latency_trace, "stt"))}</td>
                </tr>
                <tr>
                  <td>Retrieval</td>
                  <td>{formatMs(getStageDuration(result.latency_trace, "retrieval"))}</td>
                </tr>
                <tr>
                  <td>Generation</td>
                  <td>{formatMs(getStageDuration(result.latency_trace, "generation"))}</td>
                </tr>
                <tr>
                  <td>Total</td>
                  <td>{formatMs(getTotalDuration(result.latency_trace))}</td>
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
