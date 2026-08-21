"use client";

import { useEffect, useRef } from "react";

const BAR_COUNT = 32;

interface WaveformProps {
  isActive: boolean;
  analyserNode: AnalyserNode | null;
  color?: string;
}

export default function Waveform({ isActive, analyserNode, color = "var(--accent)" }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | null>(null);
  const dataRef = useRef<Uint8Array>(new Uint8Array(BAR_COUNT));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const data = dataRef.current;

      if (analyserNode && isActive) {
        const buffer = new Uint8Array(analyserNode.frequencyBinCount);
        analyserNode.getByteFrequencyData(buffer);
        // Down-sample to BAR_COUNT bars
        const step = Math.floor(buffer.length / BAR_COUNT);
        for (let i = 0; i < BAR_COUNT; i++) {
          data[i] = buffer[i * step] ?? 0;
        }
      } else if (!isActive) {
        // Decay bars to zero
        for (let i = 0; i < BAR_COUNT; i++) {
          data[i] = Math.max(0, data[i] - 12);
        }
      }

      const barW = (w / BAR_COUNT) - 1;
      const resolvedColor = getComputedStyle(canvas).getPropertyValue("--bar-color").trim() || color;

      for (let i = 0; i < BAR_COUNT; i++) {
        const normalized = data[i] / 255;
        const barH = Math.max(2, normalized * h * 0.85);
        const x = i * (barW + 1);
        const y = (h - barH) / 2;

        ctx.fillStyle = resolvedColor;
        ctx.globalAlpha = 0.3 + normalized * 0.7;
        ctx.fillRect(x, y, barW, barH);
      }
      ctx.globalAlpha = 1;

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [isActive, analyserNode, color]);

  return (
    <canvas
      ref={canvasRef}
      width={BAR_COUNT * 6}
      height={48}
      style={{
        width: "100%",
        height: "48px",
        display: "block",
        // Pass CSS variable to canvas via computed style trick
        "--bar-color": color,
      } as React.CSSProperties}
      aria-hidden="true"
    />
  );
}
