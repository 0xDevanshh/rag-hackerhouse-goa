import type { NextConfig } from "next";

// Static export is NOT compatible with this app's use of browser APIs
// (localStorage, navigator.mediaDevices, AudioContext, WebSocket).
// Deploy as a Node.js Web Service on Render using render.yaml,
// not as a Static Site. `output: 'export'` must NOT be set here.
const nextConfig: NextConfig = {
  // No output: 'export' — this app requires a Node.js runtime.
};

if (process.env.VERCEL && !process.env.NEXT_PUBLIC_API_URL) {
  throw new Error(
    "NEXT_PUBLIC_API_URL is not set. It must point at the deployed FastAPI " +
      "backend, including the scheme and no trailing slash — e.g. " +
      "https://voice-rag-api.onrender.com. Add it under Project Settings → " +
      "Environment Variables and redeploy.",
  );
}

export default nextConfig;