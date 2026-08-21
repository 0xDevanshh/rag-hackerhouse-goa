"use client";

type MicState = "IDLE" | "HOVER" | "RECORDING" | "PROCESSING" | "SUCCESS" | "ERROR";

interface MicButtonProps {
  state: MicState;
  disabled?: boolean;
  onClick: () => void;
  recordingSeconds?: number;
}

export default function MicButton({
  state,
  disabled = false,
  onClick,
  recordingSeconds = 0,
}: MicButtonProps) {
  const isRecording = state === "RECORDING";
  const isProcessing = state === "PROCESSING";

  const bgColor =
    state === "RECORDING" ? "var(--accent2)" :
    state === "SUCCESS"   ? "var(--success)" :
    state === "ERROR"     ? "var(--danger)" :
    state === "PROCESSING" ? "var(--fg-muted)" :
    "var(--fg)";

  const iconColor =
    state === "RECORDING" ? "#fff" :
    state === "SUCCESS"   ? "#fff" :
    state === "ERROR"     ? "#fff" :
    "var(--bg)";

  const label =
    state === "RECORDING"  ? `Recording — ${recordingSeconds}s. Press to stop.` :
    state === "PROCESSING" ? "Processing your query" :
    state === "SUCCESS"    ? "Query completed" :
    state === "ERROR"      ? "Error — press to retry" :
    "Press to start voice recording";

  return (
    <button
      onClick={onClick}
      disabled={disabled || isProcessing}
      aria-label={label}
      aria-pressed={isRecording}
      style={{
        position: "relative",
        width: "88px",
        height: "88px",
        borderRadius: "50%",
        border: `2px solid ${bgColor}`,
        background: bgColor,
        color: iconColor,
        cursor: disabled || isProcessing ? "not-allowed" : "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
        opacity: disabled ? 0.5 : 1,
        transition: "background var(--duration-fast), border-color var(--duration-fast), transform var(--duration-fast)",
        boxShadow: isRecording
          ? "0 0 0 0 var(--accent2)"
          : "none",
        animation: isRecording ? "mic-pulse 2s ease-out infinite" : "none",
      }}
    >
      {isProcessing ? (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke={iconColor} strokeWidth="2" strokeDasharray="56" strokeDashoffset="14">
            <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur="0.8s" repeatCount="indefinite"/>
          </circle>
        </svg>
      ) : isRecording ? (
        // Stop square icon
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
          <rect x="4" y="4" width="12" height="12" fill={iconColor} rx="1"/>
        </svg>
      ) : (
        // Microphone icon
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <rect x="9" y="2" width="6" height="12" rx="3" fill={iconColor}/>
          <path d="M5 10a7 7 0 0 0 14 0" stroke={iconColor} strokeWidth="2" strokeLinecap="square"/>
          <line x1="12" y1="17" x2="12" y2="21" stroke={iconColor} strokeWidth="2" strokeLinecap="square"/>
          <line x1="8" y1="21" x2="16" y2="21" stroke={iconColor} strokeWidth="2" strokeLinecap="square"/>
        </svg>
      )}

      <style>{`
        @keyframes mic-pulse {
          0%   { box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent2) 60%, transparent); }
          70%  { box-shadow: 0 0 0 20px color-mix(in srgb, var(--accent2) 0%, transparent); }
          100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent2) 0%, transparent); }
        }
        @media (prefers-reduced-motion: reduce) {
          @keyframes mic-pulse { from { box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent2) 40%, transparent); } to { box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent2) 40%, transparent); } }
        }
      `}</style>
    </button>
  );
}
