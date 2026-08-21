"use client";

import styles from "./VoiceOrb.module.css";

export type OrbState = "idle" | "recording" | "processing" | "result" | "error";

/** A deliberately restrained audio visualizer: signal, not spectacle. */
export default function VoiceOrb({ state }: { state: OrbState }) {
  return (
    <div className={`${styles.voiceSignal} ${styles[state]}`} aria-hidden="true">
      <div className={styles.signalGrid} />
      <div className={styles.signalBars}>
        {Array.from({ length: 27 }, (_, index) => <i key={index} />)}
      </div>
      <div className={styles.signalReadout}>
        <span>INPUT SIGNAL</span>
        <b>{state === "recording" ? "REC" : state === "result" ? "VERIFIED" : state === "processing" ? "PROCESSING" : "READY"}</b>
      </div>
    </div>
  );
}
