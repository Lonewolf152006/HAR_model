import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "ORBITAL-HAR — Avionics Telemetry Console",
  description: "Mission Control Telemetry & Human Activity Recognition System for Microgravity Operations",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} h-screen overflow-hidden antialiased dark`}
    >
      <body className="h-screen max-h-screen overflow-hidden flex flex-col bg-[#0B0D10] text-[#E6E9ED] selection:bg-[#00E08A]/20 selection:text-[#00E08A] font-sans">
        {children}
      </body>
    </html>
  );
}
