import type { NextConfig } from "next";

if (process.env.VERCEL && !process.env.NEXT_PUBLIC_API_URL) {
  throw new Error(
    "NEXT_PUBLIC_API_URL is not set. It must point at the deployed FastAPI " +
      "backend, including the scheme and no trailing slash — e.g. " +
      "https://voice-rag-api.onrender.com. Add it under Project Settings → " +
      "Environment Variables and redeploy.",
  );
}

const nextConfig: NextConfig = {
  output: "export",
};

export default nextConfig;