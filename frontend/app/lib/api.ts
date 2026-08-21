// ─── API adapter layer ─────────────────────────────────────────────────────
// All network calls go through here. No component touches fetch() directly.

import type { PipelineResult } from "./types";

export const API_URL =
  typeof process !== "undefined"
    ? (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000")
    : "http://localhost:8000";

// ── Health / readiness ────────────────────────────────────────────────────

export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { signal, cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}

// ── Text query — fast extractive path ─────────────────────────────────────
// POST /query/realtime/text — MockRealtimeSTT + ExtractiveProvider, ~30-50ms

export async function queryText(
  query: string,
  signal?: AbortSignal,
): Promise<PipelineResult> {
  const res = await fetch(`${API_URL}/query/realtime/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, `Text query failed (${res.status}): ${body}`);
  }
  return res.json() as Promise<PipelineResult>;
}

// ── Text query — LLM path ─────────────────────────────────────────────────
// POST /query/text/llm — hosted Groq/Anthropic, ~700-1300ms

export async function queryTextLlm(
  query: string,
  signal?: AbortSignal,
): Promise<PipelineResult> {
  const res = await fetch(`${API_URL}/query/text/llm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, `LLM query failed (${res.status}): ${body}`);
  }
  return res.json() as Promise<PipelineResult>;
}

// ── Voice query — batch audio path ────────────────────────────────────────
// POST /query — multipart audio upload → STT → RAG

export async function queryAudio(
  audioBlob: Blob,
  signal?: AbortSignal,
): Promise<PipelineResult> {
  const form = new FormData();
  form.append("audio", audioBlob, "recording.webm");
  const res = await fetch(`${API_URL}/query`, {
    method: "POST",
    body: form,
    signal,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, `Audio query failed (${res.status}): ${body}`);
  }
  return res.json() as Promise<PipelineResult>;
}

// ── Realtime STT config ───────────────────────────────────────────────────

export interface RealtimeSttConfig {
  direct: boolean;
  url?: string;
  protocol?: string;
  query?: Record<string, string>;
}

export async function fetchRealtimeSttConfig(): Promise<RealtimeSttConfig | null> {
  try {
    const res = await fetch(`${API_URL}/stt/realtime/config`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json() as Promise<RealtimeSttConfig>;
  } catch {
    return null;
  }
}

// ── Error type ────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function isAbortError(err: unknown): boolean {
  return err instanceof Error && err.name === "AbortError";
}
