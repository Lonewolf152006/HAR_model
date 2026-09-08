"use client";

import React from "react";
import { AlertOctagon, Volume2 } from "lucide-react";

interface AlertBannerProps {
  alertText: string | null;
}

export function AlertBanner({ alertText }: AlertBannerProps) {
  if (!alertText) return null;

  return (
    <div className="relative z-50 w-full bg-[#1A0E10] bezel-depth border border-[#FF4D4F]/70 p-3 rounded-[2px] shadow-2xl flex items-center justify-between gap-3 animate-fadeIn select-none">
      <div className="flex items-center gap-2.5 min-w-0">
        <AlertOctagon className="w-5 h-5 text-[#FF4D4F] glow-critical shrink-0 animate-bounce" />
        <div className="min-w-0">
          <div className="font-mono text-[10px] uppercase tracking-widest text-[#FF4D4F] font-bold flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#FF4D4F] glow-critical animate-ping" />
            <span>AVIONICS TELEMETRY FAULT // VOICED ALERT</span>
          </div>
          <p className="font-mono text-xs text-[#E6E9ED] font-semibold truncate mt-0.5">
            {alertText}
          </p>
        </div>
      </div>

      {/* Animated Equalizer Waveform simulating synthesized voice TTS transmission */}
      <div className="flex items-center gap-2 shrink-0 bg-[#251214] px-2.5 py-1 rounded-[2px] border border-[#FF4D4F]/50 bezel-depth-subtle">
        <Volume2 className="w-3.5 h-3.5 text-[#FF4D4F] glow-critical" />
        <div className="flex items-end gap-1 h-3.5">
          <span className="w-1 bg-[#FF4D4F] rounded-full animate-wave-1" />
          <span className="w-1 bg-[#FF4D4F] rounded-full animate-wave-2" />
          <span className="w-1 bg-[#FF4D4F] rounded-full animate-wave-3" />
          <span className="w-1 bg-[#FF4D4F] rounded-full animate-wave-4" />
        </div>
        <span className="font-mono text-[9px] text-[#FF4D4F] font-bold uppercase hidden sm:inline">
          TTS ACTIVE
        </span>
      </div>
    </div>
  );
}
