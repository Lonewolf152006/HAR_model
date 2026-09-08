"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { GateResults } from "@/lib/types";
import { useTelemetry } from "@/context/TelemetryContext";
import { CheckCircle2, XCircle, ShieldCheck } from "lucide-react";

interface GateChecklistLiveProps {
  gates: GateResults;
}

export function GateChecklistLive({ gates }: GateChecklistLiveProps) {
  const { isReducedMotion } = useTelemetry();
  const gateList = Object.values(gates);
  const passCount = gateList.filter((g) => g.passed).length;
  const allPassed = passCount === gateList.length;

  return (
    <AvionicsPanel
      title="DECISION STABILIZER"
      badge={
        <span
          className={`font-mono text-[9px] px-1.5 py-0.5 rounded-[2px] border font-bold bezel-depth-subtle ${
            allPassed
              ? "bg-[#171B21] text-[#00E08A] border-[#00E08A]/50 glow-nominal"
              : "bg-[#1A0E10] text-[#FF4D4F] border-[#FF4D4F]/60"
          }`}
        >
          {passCount}/5 GATES ARMED
        </span>
      }
      bracketColor={allPassed ? "border-[#00E08A]/60" : "border-[#FF4D4F]"}
      className="flex-1 min-h-0 flex flex-col overflow-hidden p-2.5"
    >
      {/* Scrollable Gate List with dedicated internal overflow-y-auto */}
      <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-1 font-mono">
        {gateList.map((gate) => {
          const isCausalViolation = gate.id === "causal_logic" && !gate.passed;

          return (
            <div
              key={gate.id}
              className={`px-2 py-1 rounded-[2px] border transition-colors flex items-center justify-between gap-2 bezel-depth-subtle ${
                isCausalViolation
                  ? `bg-[#241113] border-[#FF4D4F] ${!isReducedMotion ? "animate-alert-row-flash" : ""}`
                  : gate.passed
                  ? "bg-[#171B21] border-white/5 hover:border-white/10"
                  : "bg-[#1A0E10] border-[#FF4D4F]/30"
              }`}
            >
              <div className="flex items-center gap-1.5 min-w-0">
                {gate.passed ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#00E08A] glow-nominal shrink-0 animate-glow-snap" />
                ) : isCausalViolation ? (
                  <XCircle className="w-3.5 h-3.5 text-[#FF4D4F] glow-critical shrink-0" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 text-[#FF4D4F]/80 shrink-0" />
                )}
                <div className="min-w-0">
                  <div className="text-[11px] font-semibold text-[#E6E9ED] truncate leading-tight">
                    {gate.name}
                  </div>
                  <div
                    className={`text-[9px] leading-tight truncate ${
                      gate.passed
                        ? "text-[#8A919C]"
                        : isCausalViolation
                        ? "text-[#FF4D4F] font-semibold"
                        : "text-[#FF4D4F]"
                    }`}
                  >
                    {gate.reason}
                  </div>
                </div>
              </div>

              <div className="text-right shrink-0">
                <span
                  className={`text-[9px] font-bold px-1 py-0.2 rounded-[1px] ${
                    gate.passed
                      ? "text-[#00E08A] glow-nominal bg-[#00E08A]/10"
                      : isCausalViolation
                      ? "text-[#FF4D4F] glow-critical bg-[#FF4D4F]/20 border border-[#FF4D4F]/50"
                      : "text-[#FF4D4F] bg-[#FF4D4F]/10"
                  }`}
                >
                  {gate.passed ? "PASS" : "FAIL"}
                </span>
                <div className="text-[8px] text-[#565C66]">
                  {gate.value}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-1.5 pt-1.5 border-t border-white/5 flex items-center justify-between font-mono text-[8px] text-[#565C66] shrink-0">
        <span className="flex items-center gap-1 text-[#00E08A]">
          <ShieldCheck className="w-3 h-3 text-[#00E08A] glow-nominal" />
          <span>ARMED AT 10Hz</span>
        </span>
        <span className="text-[#565C66]">5-GATE VERIFICATION</span>
      </div>
    </AvionicsPanel>
  );
}
