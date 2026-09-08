"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { Volume2, VolumeX, Sparkles, Terminal, ToggleLeft, ToggleRight } from "lucide-react";

export function SystemTogglesPanel() {
  const {
    isMuted,
    setIsMuted,
    isReducedMotion,
    setIsReducedMotion,
    replayBootSequence,
  } = useTelemetry();

  return (
    <AvionicsPanel
      title="SYSTEM PREFERENCES & AVIONICS BUS"
      className="p-3 font-mono flex flex-col justify-between"
    >
      <div className="space-y-2">
        <div className="flex items-center justify-between text-[9px] text-[#565C66] tracking-wider uppercase pb-1 border-b border-white/5">
          <span>GLOBAL SUBSYSTEM SWITCHES</span>
          <span>DISCRETE INSTRUMENT STATE</span>
        </div>

        {/* Toggle 1: Global Sound */}
        <div className="p-2 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className={`p-1 rounded-[2px] border ${
                !isMuted
                  ? "bg-[#00E08A]/10 text-[#00E08A] border-[#00E08A]/40 glow-nominal"
                  : "bg-white/5 text-[#565C66] border-white/10"
              }`}
            >
              {!isMuted ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
            </div>
            <div>
              <div className="text-[10px] font-bold text-[#E6E9ED]">
                AUDITORY TELEMETRY SYNTH
              </div>
              <div className="text-[8.5px] text-[#8A919C]">
                WebAudio synthesizer for state transitions & rejections
              </div>
            </div>
          </div>

          {/* Squared Instrument Switch (2-4px radius, no pills) */}
          <button
            type="button"
            onClick={() => setIsMuted(!isMuted)}
            className={`px-2.5 py-1 rounded-[2px] text-[10px] font-bold font-mono border transition-all duration-150 flex items-center gap-1.5 bezel-depth-subtle ${
              !isMuted
                ? "bg-[#0E1A14] text-[#00E08A] border-[#00E08A]/50 glow-nominal"
                : "bg-[#0E1015] text-[#565C66] border-white/10"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-[1px] ${
                !isMuted ? "bg-[#00E08A] animate-pulse" : "bg-[#565C66]"
              }`}
            />
            <span>{!isMuted ? "ACTIVE" : "MUTED"}</span>
          </button>
        </div>

        {/* Toggle 2: UI Motion / Animations */}
        <div className="p-2 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className={`p-1 rounded-[2px] border ${
                !isReducedMotion
                  ? "bg-[#4DA3FF]/10 text-[#4DA3FF] border-[#4DA3FF]/40 glow-info"
                  : "bg-[#FFB020]/10 text-[#FFB020] border-[#FFB020]/40 glow-caution"
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-[10px] font-bold text-[#E6E9ED]">
                DYNAMIC RADAR & PULSE MOTION
              </div>
              <div className="text-[8.5px] text-[#8A919C]">
                High-rate CSS animations, radar sweeps & stamp flashes
              </div>
            </div>
          </div>

          {/* Squared Instrument Switch */}
          <button
            type="button"
            onClick={() => setIsReducedMotion(!isReducedMotion)}
            className={`px-2.5 py-1 rounded-[2px] text-[10px] font-bold font-mono border transition-all duration-150 flex items-center gap-1.5 bezel-depth-subtle ${
              !isReducedMotion
                ? "bg-[#0E1A14] text-[#00E08A] border-[#00E08A]/50 glow-nominal"
                : "bg-[#1A140E] text-[#FFB020] border-[#FFB020]/50 glow-caution"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-[1px] ${
                !isReducedMotion ? "bg-[#00E08A]" : "bg-[#FFB020]"
              }`}
            />
            <span>{!isReducedMotion ? "ENABLED" : "REDUCED"}</span>
          </button>
        </div>
      </div>

      {/* Utility Action: Replay Boot Sequence */}
      <div className="pt-2 mt-2 border-t border-white/5 flex items-center justify-between">
        <span className="text-[8.5px] text-[#565C66]">
          DIAGNOSTIC UTILITY: CRT TERMINAL LOADER
        </span>

        <button
          type="button"
          onClick={replayBootSequence}
          className="flex items-center gap-1.5 px-2.5 py-1 text-[9.5px] font-mono font-bold text-[#8A919C] hover:text-[#4DA3FF] bg-[#0E1015] hover:bg-[#171B21] border border-white/15 hover:border-[#4DA3FF]/50 rounded-[2px] transition-colors bezel-depth-subtle"
        >
          <Terminal className="w-3 h-3 text-[#4DA3FF]" />
          <span>REPLAY BOOT SEQUENCE</span>
        </button>
      </div>
    </AvionicsPanel>
  );
}
