import type { PipelineResult } from "../types";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function queryText(query: string): Promise<PipelineResult> {
  const response = await fetch(`${API_URL}/query/text`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

export async function queryVoice(audio: Blob): Promise<PipelineResult> {
  const form = new FormData();
  form.append("audio", audio, "voice-query.webm");
  const response = await fetch(`${API_URL}/query/voice`, { method: "POST", body: form });
  if (!response.ok) throw new Error(`Voice request failed (${response.status})`);
  return response.json();
}

export async function healthCheck() {
  try { return (await fetch(`${API_URL}/health`)).ok; } catch { return false; }
}

export { API_URL };
