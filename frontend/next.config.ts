import type { NextConfig } from "next";

// Static export is NOT compatible with this app's use of browser APIs
// (localStorage, navigator.mediaDevices, AudioContext, WebSocket).
// Deploy as a Node.js Web Service on Render using render.yaml,
// not as a Static Site. `output: 'export'` must NOT be set here.
const nextConfig: NextConfig = {
  // No output: 'export' — this app requires a Node.js runtime.
};

export default nextConfig;
