"use client";

import React from "react";
import { MissionClock } from "./MissionClock";
import { SystemStatusStrip } from "./SystemStatusStrip";
import { useTelemetry } from "@/context/TelemetryContext";
import { Volume2, VolumeX } from "lucide-react";

export function TopBar() {
  const {
    isMuted,
    setIsMuted,
    activeAlert,
  } = useTelemetry();

  return (
    <header className="h-14 border-b border-white/10 bg-[#0B0D10] bezel-depth-subtle px-4 flex items-center justify-between z-30 shrink-0 select-none relative">
      {/* Left side: Mission Clock & Session Info */}
      <div className="flex items-center gap-4">
        <MissionClock />

        <div className="hidden lg:flex flex-col font-mono text-[10px] leading-tight text-[#565C66]">
          <span className="text-[#8A919C] font-semibold">
            NODE: ISS-COLUMBUS-HAR
          </span>
          <span>SESSION: EXP-2026-0924</span>
        </div>
      </div>

      {/* Middle: Active Alert Banner indicator if triggered */}
      {activeAlert?.active && (
        <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-[#1A0E10] border border-[#FF4D4F]/60 rounded-[2px] bezel-depth-subtle animate-pulse">
          <span className="w-2 h-2 rounded-full bg-[#FF4D4F] glow-critical" />
          <span className="font-mono text-xs font-bold text-[#FF4D4F] tracking-wide">
            {activeAlert.reason}
          </span>
        </div>
      )}

      {/* Right side: Subsystem Strip, Audio/Motion Toggles */}
      <div className="flex items-center gap-2.5">
        <div className="hidden sm:block">
          <SystemStatusStrip />
        </div>

        {/* Global Sound Toggle */}
        <button
          onClick={() => setIsMuted(!isMuted)}
          className={`p-1.5 rounded-[2px] border transition-colors bezel-depth-subtle ${
            isMuted
              ? "bg-[#12151A] border-white/10 text-[#565C66] hover:text-[#8A919C]"
              : "bg-[#12151A] border-[#00E08A]/40 text-[#00E08A] hover:bg-[#171B21]"
          }`}
          title={isMuted ? "Sound: Muted (Click to enable)" : "Sound: Active (Click to mute)"}
        >
          {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
        </button>
      </div>
    </header>
  );
}
