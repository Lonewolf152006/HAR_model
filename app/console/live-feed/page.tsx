"use client";

import React from "react";
import { useTelemetry } from "@/context/TelemetryContext";
import { VideoCanvas } from "@/components/live-feed/VideoCanvas";
import { StateConfidenceCard } from "@/components/live-feed/StateConfidenceCard";
import { GateChecklistLive } from "@/components/live-feed/GateChecklistLive";
import { ContainmentBadgesCard } from "@/components/live-feed/ContainmentBadgesCard";
export default function LiveFeedPage() {
  const {
    currentState,
    expectedNext,
    confidence,
    confidenceHistory,
    gates,
    containment,
    frame,
  } = useTelemetry();

  return (
    <div className="h-full w-full max-h-full overflow-hidden select-none font-mono relative flex flex-col">
      {/* Fixed-Viewport CSS Grid: 100vh - TopBar, 0 outer scrollbar */}
      <div className="h-full w-full grid grid-cols-12 gap-3 overflow-hidden">
        {/* Left Column: 8 cols (67% Width) Dominant Video/HUD Canvas */}
        <div className="col-span-12 lg:col-span-8 h-full min-h-0 overflow-hidden flex flex-col">
          <VideoCanvas />
        </div>

        {/* Right Column: 4 cols (33% Width) Exactly 3 Panels Fitting Single Viewport */}
        <div className="col-span-12 lg:col-span-4 h-full min-h-0 overflow-hidden flex flex-col gap-2.5">
          {/* 1. Folded State Engine + 50-Sample Live Sparkline */}
          <StateConfidenceCard
            currentState={currentState}
            expectedNext={expectedNext}
            stateDurationMs={frame.state_duration_ms}
            confidence={confidence}
            confidenceHistory={confidenceHistory}
          />

          {/* 2. Compact 5 DecisionStabilizer Gates (with dedicated internal overflow-y-auto) */}
          <GateChecklistLive gates={gates} />

          {/* 3. Compact 2.5D Geometric Containment Engine Badges */}
          <ContainmentBadgesCard containment={containment} />
        </div>
      </div>
    </div>
  );
}
