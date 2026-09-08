"use client";

import React from "react";
import { MissionClock } from "./MissionClock";
import { SystemStatusStrip } from "./SystemStatusStrip";
import { useTelemetry } from "@/context/TelemetryContext";
import { Volume2, VolumeX, Activity, Sparkles } from "lucide-react";

export function TopBar() {
  const {
    subsystems,
    isMuted,
    setIsMuted,
    isReducedMotion,
    setIsReducedMotion,
    activeAlert,
  } = useTelemetry();

  return (
    <header className="h-14 border-b border-white/10 bg-[#0B0D10] bezel-depth-subtle px-4 flex items-center justify-between z-30 shrink-0 select-none">
      {/* Left side: Mission Clock & Session Info */}
      <div className="flex items-center gap-4">
        <MissionClock />

        <div className="hidden lg:flex flex-col font-mono text-[10px] leading-tight text-[#565C66]">
          <span className="text-[#8A919C] font-semibold">
            NODE // ISS-COLUMBUS-HAR
          </span>
          <span>SESSION: EXP-2026-0924</span>
        </div>
      </div>

      {/* Middle: Active Alert Banner indicator if triggered */}
      {activeAlert && (
        <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-[#1A0E10] border border-[#FF4D4F]/60 rounded-[2px] bezel-depth-subtle animate-pulse">
          <span className="w-2 h-2 rounded-full bg-[#FF4D4F] glow-critical" />
          <span className="font-mono text-xs font-bold text-[#FF4D4F] tracking-wide uppercase">
            {activeAlert}
          </span>
        </div>
      )}

      {/* Right side: Subsystem Strip, Quick Anomaly Injector, Audio/Motion Toggles */}
      <div className="flex items-center gap-3">
        <div className="hidden sm:block">
          <SystemStatusStrip subsystems={subsystems} />
        </div>

        {/* Anomaly Injection Testing Control (Decommissioned / Offline) */}
        <div
          className="flex items-center gap-1.5 px-2.5 py-1 border border-white/5 rounded-[2px] text-xs font-mono bg-[#12151A]/40 text-[#565C66] opacity-40 cursor-not-allowed select-none bezel-depth-subtle"
          title="Fault Injection Offline // Diagnostics module unlinked"
        >
          <Activity className="w-3.5 h-3.5 text-[#565C66]" />
          <span className="hidden sm:inline">FAULT INJ (OFFLINE)</span>
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

        {/* Reduced Motion Toggle */}
        <button
          onClick={() => setIsReducedMotion(!isReducedMotion)}
          className={`p-1.5 rounded-[2px] border transition-colors bezel-depth-subtle ${
            isReducedMotion
              ? "bg-[#12151A] border-[#FFB020]/40 text-[#FFB020]"
              : "bg-[#12151A] border-white/10 text-[#8A919C] hover:text-[#E6E9ED]"
          }`}
          title={isReducedMotion ? "Reduced Motion: Enabled" : "Reduced Motion: Disabled"}
        >
          <Sparkles className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
