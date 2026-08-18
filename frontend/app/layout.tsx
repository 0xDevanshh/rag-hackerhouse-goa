import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Voice RAG",
  description: "Ask a question by voice, answered from the document corpus.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
