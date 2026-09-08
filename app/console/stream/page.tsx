"use client";

import React from "react";
import { StreamTargetPanel } from "@/components/stream/StreamTargetPanel";
import { LocalRecordingPanel } from "@/components/stream/LocalRecordingPanel";
import { BandwidthLatencyPanel } from "@/components/stream/BandwidthLatencyPanel";

export default function StreamPage() {
  return (
    <div className="h-full w-full max-h-full overflow-hidden select-none font-mono relative">
      {/* Fixed Viewport 2-Column Grid (~60/40 split, strictly no outer scrollbar) */}
      <div className="h-full w-full grid grid-cols-1 lg:grid-cols-12 gap-3 overflow-hidden">
        {/* LEFT COLUMN: Dominant Stream Target & Link Health (~60% / 7 cols) */}
        <div className="lg:col-span-7 h-full min-h-0 overflow-hidden">
          <StreamTargetPanel />
        </div>

        {/* RIGHT COLUMN: Stacked Local Recording (top) + Bandwidth/Latency (bottom) (~40% / 5 cols) */}
        <div className="lg:col-span-5 h-full min-h-0 overflow-hidden flex flex-col gap-3">
          {/* Top: Local Recording Panel (Dominant in right column) */}
          <div className="flex-[1.35] min-h-0 overflow-hidden">
            <LocalRecordingPanel />
          </div>

          {/* Bottom: Bandwidth & Latency Readout (Smallest, quietest panel) */}
          <div className="flex-1 min-h-0 overflow-hidden">
            <BandwidthLatencyPanel />
          </div>
        </div>
      </div>
    </div>
  );
}
