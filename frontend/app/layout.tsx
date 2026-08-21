import type { Metadata, Viewport } from "next";
import "./globals.css";
import "./enhancements.css";

export const metadata: Metadata = {
  title: "VoiceRAG",
  description: "Ask a question by voice, grounded in your document corpus.",
};

export const viewport: Viewport = {
  themeColor: "#050508",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning prevents false positives from browser extensions
    // injecting attributes (e.g. Grammarly, CRX emulators) into <html> after SSR.
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
