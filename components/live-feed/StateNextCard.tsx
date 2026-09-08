"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { HarState } from "@/lib/types";
import { STATE_MAP } from "@/lib/constants";
import { ArrowRight, Clock, Activity } from "lucide-react";

interface StateNextCardProps {
  currentState: HarState;
  expectedNext: HarState;
  stateDurationMs: number;
}

export function StateNextCard({
  currentState,
  expectedNext,
  stateDurationMs,
}: StateNextCardProps) {
  const currentDef = STATE_MAP[currentState] || {
    name: currentState,
    description: "",
    index: 0,
  };
  const nextDef = STATE_MAP[expectedNext] || {
    name: expectedNext,
    description: "",
    index: 0,
  };

  const elapsedSec = (stateDurationMs / 1000).toFixed(1);

  return (
    <AvionicsPanel
      title="STATE ENGINE"
      indexTag="01 // FSM"
      badge={
        <span className="font-mono text-[10px] text-[#00E08A] bg-[#171B21] px-1.5 py-0.5 rounded-[2px] border border-[#00E08A]/40 flex items-center gap-1.5 glow-nominal bezel-depth-subtle">
          <Activity className="w-3 h-3 animate-pulse" />
          ACTIVE
        </span>
      }
    >
      <div className="space-y-3 font-mono">
        {/* Current State Highlight */}
        <div>
          <span className="text-[9px] text-[#565C66] tracking-wider block uppercase mb-0.5">
            Current Action
          </span>
          <div className="flex items-baseline justify-between">
            <span className="text-base font-bold text-[#00E08A] tracking-wide flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#00E08A] glow-nominal animate-pulse" />
              {currentDef.name}
            </span>
            <span className="text-[10px] text-[#8A919C] flex items-center gap-1">
              <Clock className="w-3 h-3 text-[#565C66]" />
              {elapsedSec}s
            </span>
          </div>
          <p className="font-sans text-[11px] text-[#8A919C] mt-0.5 leading-snug">
            {currentDef.description}
          </p>
        </div>

        {/* Expected Next Step */}
        <div className="p-2 bg-[#171B21] bezel-depth-subtle border border-white/5 rounded-[2px]">
          <div className="flex items-center justify-between text-[9px] text-[#565C66] uppercase mb-1">
            <span>Next Expected Sequence</span>
            <span className="text-[#4DA3FF] font-semibold">PREDICTED</span>
          </div>
          <div className="flex items-center gap-2 text-xs font-semibold text-[#E6E9ED]">
            <ArrowRight className="w-3.5 h-3.5 text-[#4DA3FF] glow-info shrink-0" />
            <span>{nextDef.name}</span>
          </div>
        </div>
      </div>
    </AvionicsPanel>
  );
}
