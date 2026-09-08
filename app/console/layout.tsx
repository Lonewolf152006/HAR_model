"use client";

import React from "react";
import { TelemetryProvider, useTelemetry } from "@/context/TelemetryContext";
import { Sidebar } from "@/components/shell/Sidebar";
import { TopBar } from "@/components/shell/TopBar";
import { BootSequence } from "@/components/shell/BootSequence";

function ConsoleShellInner({ children }: { children: React.ReactNode }) {
  const { isReplayingBoot, handleBootComplete } = useTelemetry();

  return (
    <div className="h-screen w-screen max-h-screen overflow-hidden technical-blueprint-bg text-[#E6E9ED] relative select-none flex">
      {/* Persistent 2-3% Film-Grain Noise Overlay Across Viewport */}
      <div className="film-grain" />

      {/* Terminal Boot Sequence Overlay (First load or replay trigger) */}
      <BootSequence
        key={isReplayingBoot ? "boot-replay" : "boot-normal"}
        forceReplay={isReplayingBoot}
        onComplete={handleBootComplete}
      />

      {/* Fixed Left Breaker Panel Sidebar */}
      <Sidebar />

      {/* Main Console Viewport */}
      <div className="flex-1 flex flex-col min-w-0 h-screen max-h-screen overflow-hidden">
        <TopBar />

        {/* Fixed Viewport Tab Content Container (Never Produces Outer Scrollbar) */}
        <main className="flex-1 h-[calc(100vh-3.5rem)] max-h-[calc(100vh-3.5rem)] overflow-hidden p-3 bg-[#0B0D10] technical-blueprint-bg relative flex flex-col">
          <div className="scanline-overlay absolute inset-0 pointer-events-none opacity-10" />
          <div className="relative z-10 h-full w-full overflow-hidden flex flex-col">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

export default function ConsoleLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <TelemetryProvider>
      <ConsoleShellInner>{children}</ConsoleShellInner>
    </TelemetryProvider>
  );
}
