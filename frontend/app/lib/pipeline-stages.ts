// ─── Pipeline stage state management ─────────────────────────────────────

import type { PipelineStage, UiPhase } from "./types";

export const STAGE_DEFS: Array<{ id: string; label: string; shortLabel: string }> = [
  { id: "speech",   label: "SPEECH",   shortLabel: "01" },
  { id: "query",    label: "QUERY",    shortLabel: "02" },
  { id: "embed",    label: "EMBED",    shortLabel: "03" },
  { id: "retrieve", label: "RETRIEVE", shortLabel: "04" },
  { id: "rerank",   label: "RERANK",   shortLabel: "05" },
  { id: "ground",   label: "GROUND",   shortLabel: "06" },
  { id: "answer",   label: "ANSWER",   shortLabel: "07" },
];

// Which stages are "active" (pulsing) or "complete" for each UiPhase
const PHASE_MAP: Record<UiPhase, { active: string[]; complete: string[] }> = {
  IDLE:         { active: [], complete: [] },
  RECORDING:    { active: ["speech"], complete: [] },
  UPLOADING:    { active: ["speech"], complete: [] },
  TRANSCRIBING: { active: ["speech", "query"], complete: [] },
  RETRIEVING:   { active: ["embed", "retrieve", "rerank"], complete: ["speech", "query"] },
  GROUNDING:    { active: ["ground"], complete: ["speech", "query", "embed", "retrieve", "rerank"] },
  GENERATING:   { active: ["answer"], complete: ["speech", "query", "embed", "retrieve", "rerank", "ground"] },
  COMPLETE:     { active: [], complete: ["speech", "query", "embed", "retrieve", "rerank", "ground", "answer"] },
  DEGRADED:     { active: [], complete: ["speech", "query", "embed", "retrieve", "rerank"] },
  ERROR:        { active: [], complete: [] },
  OFFLINE_DEMO: { active: [], complete: [] },
};

export function buildStages(
  phase: UiPhase,
  latency?: Record<string, number | null>,
): PipelineStage[] {
  const { active, complete } = PHASE_MAP[phase] ?? { active: [], complete: [] };
  const failed = phase === "ERROR" ? ["speech"] : [];

  return STAGE_DEFS.map((def) => {
    let status: PipelineStage["status"] = "PENDING";
    if (failed.includes(def.id)) status = "FAILED";
    else if (complete.includes(def.id)) status = "COMPLETE";
    else if (active.includes(def.id)) status = "ACTIVE";

    const durationMs = latency?.[def.id] ?? null;

    return {
      id: def.id,
      label: def.label,
      shortLabel: def.shortLabel,
      status,
      durationMs,
    };
  });
}
