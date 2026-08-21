import type { Metadata } from "next";
import "./globals.css";
import "./enhancements.css";

export const metadata: Metadata = {
  title: "VOICE RAG™ — Ask. Retrieve. Verify.",
  description:
    "A multilingual voice-first RAG system that retrieves evidence, checks relevance, verifies grounding, and returns answers you can inspect.",
  openGraph: {
    title: "VOICE RAG™",
    description: "Ask. Retrieve. Verify. Voice-first multilingual research.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>{children}</body>
    </html>
  );
}
