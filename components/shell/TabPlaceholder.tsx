"use client";

import React from "react";
import { useTelemetry } from "@/context/TelemetryContext";
import { Terminal, Shield, CheckCircle2, AlertCircle } from "lucide-react";

interface TabPlaceholderProps {
  tabIndex: string;
  tabTitle: string;
  tabDescription: string;
  priorityStep: string;
}

export function TabPlaceholder({
  tabIndex,
  tabTitle,
  tabDescription,
  priorityStep,
}: TabPlaceholderProps) {
  const { frame, currentState, expectedNext, confidence, gates, cyclesCompleted } =
    useTelemetry();

  return (
    <div className="h-full flex flex-col gap-4 font-mono">
      {/* Top Header Card */}
      <div className="p-4 bg-[#12151A] border border-white/10 rounded-[2px] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-[2px] bg-[#171B21] border border-white/10 flex items-center justify-center text-[#00E08A] text-xs font-bold">
            {tabIndex}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-bold text-[#E6E9ED] uppercase tracking-wider">
                {tabTitle}
              </h1>
              <span className="px-1.5 py-0.5 text-[9px] bg-[#FFB020]/10 border border-[#FFB020]/40 text-[#FFB020] rounded-[2px]">
                STUBBED ROUTE
              </span>
            </div>
            <p className="text-xs text-[#8A919C] mt-0.5">{tabDescription}</p>
          </div>
        </div>

        <div className="text-right text-[10px] text-[#565C66] hidden sm:block">
          <div>SCHEDULED BUILD PRIORITY: {priorityStep}</div>
          <div className="text-[#00E08A] font-semibold">ROUTER BUS OK // TELEMETRY SYNCED</div>
        </div>
      </div>

      {/* Main Grid Card: Telemetry Proof of Life */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Left: State Engine Telemetry */}
        <div className="p-4 bg-[#12151A] border border-white/10 rounded-[2px] flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-2 border-b border-white/10 text-xs text-[#8A919C]">
              <span>STATE ENGINE</span>
              <span className="text-[#00E08A] animate-pulse">● LIVE</span>
            </div>

            <div className="mt-4 space-y-3">
              <div>
                <span className="text-[10px] text-[#565C66] block">CURRENT ACTIVE STATE</span>
                <span className="text-base font-bold text-[#00E08A] tracking-wider">
                  {currentState}
                </span>
              </div>

              <div>
                <span className="text-[10px] text-[#565C66] block">NEXT EXPECTED TRANSITION</span>
                <span className="text-xs text-[#8A919C] tracking-wide">
                  {expectedNext}
                </span>
              </div>

              <div>
                <span className="text-[10px] text-[#565C66] block">PREDICTION CONFIDENCE</span>
                <div className="flex items-center gap-2 mt-1">
                  <div className="flex-1 h-1.5 bg-[#171B21] rounded-full overflow-hidden border border-white/5">
                    <div
                      className="h-full bg-[#00E08A] transition-all duration-200"
                      style={{ width: `${Math.min(100, confidence * 100)}%` }}
                    />
                  </div>
                  <span className="text-xs text-[#E6E9ED] font-semibold">
                    {(confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              <div className="pt-2 border-t border-white/5 text-[11px] text-[#8A919C] flex justify-between">
                <span>Cycles Completed:</span>
                <span className="text-[#4DA3FF] font-bold">#{cyclesCompleted}</span>
              </div>
            </div>
          </div>

          <div className="p-2.5 bg-[#171B21] border border-white/5 text-[10px] text-[#565C66]">
            FRAME ID: #{frame.frame_id} // FPS: {frame.fps}
          </div>
        </div>

        {/* Center: 5 DecisionStabilizer Gates status */}
        <div className="p-4 bg-[#12151A] border border-white/10 rounded-[2px]">
          <div className="flex items-center justify-between pb-2 border-b border-white/10 text-xs text-[#8A919C]">
            <span>STABILIZATION GATES</span>
            <Shield className="w-3.5 h-3.5 text-[#4DA3FF]" />
          </div>

          <div className="mt-3 space-y-2">
            {Object.values(gates).map((gate) => (
              <div
                key={gate.id}
                className="p-2 bg-[#171B21]/60 border border-white/5 rounded-[2px] flex items-center justify-between text-xs"
              >
                <div className="flex items-center gap-2">
                  {gate.passed ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-[#00E08A] shrink-0" />
                  ) : (
                    <AlertCircle className="w-3.5 h-3.5 text-[#FF4D4F] shrink-0" />
                  )}
                  <span className="text-[#E6E9ED]">{gate.name}</span>
                </div>
                <span
                  className={`text-[10px] font-semibold ${
                    gate.passed ? "text-[#00E08A]" : "text-[#FF4D4F]"
                  }`}
                >
                  {gate.passed ? "PASS" : "FAIL"}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Containment & FSM Booleans */}
        <div className="p-4 bg-[#12151A] border border-white/10 rounded-[2px]">
          <div className="flex items-center justify-between pb-2 border-b border-white/10 text-xs text-[#8A919C]">
            <span>CONTAINMENT & FSM</span>
            <Terminal className="w-3.5 h-3.5 text-[#8A919C]" />
          </div>

          <div className="mt-3 space-y-2 text-xs">
            <div className="flex items-center justify-between p-2 bg-[#171B21]/60 border border-white/5 rounded-[2px]">
              <span className="text-[#8A919C]">Main Container:</span>
              <span className="text-[#E6E9ED] font-semibold">{frame.containment.main_box}</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-[#171B21]/60 border border-white/5 rounded-[2px]">
              <span className="text-[#FF4D4F]">Red Sample Cube:</span>
              <span className="text-[#E6E9ED] font-semibold">{frame.containment.red_box}</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-[#171B21]/60 border border-white/5 rounded-[2px]">
              <span className="text-[#4DA3FF]">Blue Sample Cube:</span>
              <span className="text-[#E6E9ED] font-semibold">{frame.containment.blue_box}</span>
            </div>
            <div className="flex items-center justify-between p-2 bg-[#171B21]/60 border border-white/5 rounded-[2px]">
              <span className="text-[#8A919C]">box_open:</span>
              <span className={frame.fsm.box_open ? "text-[#00E08A]" : "text-[#565C66]"}>
                {frame.fsm.box_open ? "TRUE" : "FALSE"}
              </span>
            </div>
            <div className="flex items-center justify-between p-2 bg-[#171B21]/60 border border-white/5 rounded-[2px]">
              <span className="text-[#8A919C]">red_placed_out:</span>
              <span className={frame.fsm.red_placed_out ? "text-[#00E08A]" : "text-[#565C66]"}>
                {frame.fsm.red_placed_out ? "TRUE" : "FALSE"}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
