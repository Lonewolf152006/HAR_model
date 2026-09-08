"use client";

import React, { useMemo } from "react";
import { useTelemetry } from "@/context/TelemetryContext";
import { HAR_STATES, STATE_MAP } from "@/lib/constants";
import { HarState } from "@/lib/types";
import { Check, ShieldAlert, Disc, Clock, ChevronRight } from "lucide-react";

export function FlightPlanStrip() {
  const {
    currentState,
    expectedNext,
    confidence,
    cyclesCompleted,
    frame,
    containment,
    activeRejection,
    lastAcceptedState,
  } = useTelemetry();

  const currentIndex = useMemo(() => {
    return HAR_STATES.findIndex((s) => s.id === currentState);
  }, [currentState]);

  // Real hold-time / stability progress toward the gate threshold (14 frames minimum)
  const activeGateProgress = useMemo(() => {
    const rawProgress = (frame.stability_count / 14) * 100;
    return Math.min(100, Math.max(6, Math.round(rawProgress)));
  }, [frame.stability_count]);

  const getContainmentBadge = (stepId: HarState) => {
    if (stepId === "pick_red" || stepId === "place_red_out") {
      const status = containment.red_box;
      const statusColor =
        status === "HELD"
          ? "text-[#4DA3FF] border-[#4DA3FF]/50 glow-info"
          : status === "INSIDE"
          ? "text-[#00E08A] border-[#00E08A]/50 glow-nominal"
          : "text-[#FFB020] border-[#FFB020]/50 glow-caution";

      return (
        <div className="flex items-center gap-1 px-1.5 py-0.5 bg-[#171B21] border border-white/10 rounded-[2px] bezel-depth-subtle font-mono text-[9px]">
          <span className="w-1.5 h-1.5 rounded-[1px] bg-[#FF4D4F] shrink-0" />
          <span className="text-[#8A919C]">RED:</span>
          <span className={`font-bold ${statusColor}`}>{status}</span>
        </div>
      );
    }

    if (stepId === "pick_blue" || stepId === "place_blue_in") {
      const status = containment.blue_box;
      const statusColor =
        status === "HELD"
          ? "text-[#4DA3FF] border-[#4DA3FF]/50 glow-info"
          : status === "INSIDE"
          ? "text-[#00E08A] border-[#00E08A]/50 glow-nominal"
          : "text-[#FFB020] border-[#FFB020]/50 glow-caution";

      return (
        <div className="flex items-center gap-1 px-1.5 py-0.5 bg-[#171B21] border border-white/10 rounded-[2px] bezel-depth-subtle font-mono text-[9px]">
          <span className="w-1.5 h-1.5 rounded-[1px] bg-[#4DA3FF] shrink-0" />
          <span className="text-[#8A919C]">BLUE:</span>
          <span className={`font-bold ${statusColor}`}>{status}</span>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="relative bg-[#12151A] bezel-depth border border-white/10 rounded-[2px] p-3 flex flex-col justify-between h-full w-full overflow-hidden">
      {/* 4 Precision HUD Corner Bracket Accents */}
      <span className="absolute -top-[1px] -left-[1px] w-2.5 h-2.5 border-t-[1.5px] border-l-[1.5px] border-[#00E08A]/70 pointer-events-none" />
      <span className="absolute -top-[1px] -right-[1px] w-2.5 h-2.5 border-t-[1.5px] border-r-[1.5px] border-[#00E08A]/70 pointer-events-none" />
      <span className="absolute -bottom-[1px] -left-[1px] w-2.5 h-2.5 border-b-[1.5px] border-l-[1.5px] border-[#00E08A]/70 pointer-events-none" />
      <span className="absolute -bottom-[1px] -right-[1px] w-2.5 h-2.5 border-b-[1.5px] border-r-[1.5px] border-[#00E08A]/70 pointer-events-none" />

      {/* Signature Sequence Condensed Single-Line Header */}
      <div className="flex items-center justify-between pb-2.5 border-b border-white/10 shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="font-mono text-[9px] text-[#565C66] tracking-wider uppercase">
              02 // SIGNATURE SEQUENCE
            </span>
            <span className="text-[#565C66] font-mono text-[10px]">|</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-[1px] bg-[#00E08A] glow-nominal animate-pulse" />
            <h2 className="font-mono text-sm md:text-base font-bold text-[#E6E9ED] tracking-wide uppercase">
              {STATE_MAP[currentState]?.name || currentState}
            </h2>
          </div>

          <div className="hidden lg:flex items-center gap-1 text-[11px] font-mono text-[#8A919C] bg-[#171B21] px-2 py-0.5 rounded-[2px] border border-white/5">
            <span className="text-[#565C66]">EXPECTED:</span>
            <span className="text-[#4DA3FF] font-semibold">
              {expectedNext.toUpperCase()}
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-1 text-[11px] font-mono bg-[#171B21] px-2 py-0.5 rounded-[2px] border border-white/5">
            <span className="text-[#565C66]">CONF:</span>
            <span
              className={`font-semibold ${
                confidence >= 0.75 ? "text-[#00E08A]" : "text-[#FFB020]"
              }`}
            >
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        {/* Top-Right Cycle Counter Badge */}
        <div className="flex items-center gap-2 bg-[#171B21] px-2.5 py-1 rounded-[2px] border border-white/10 bezel-depth-subtle shrink-0">
          <span className="text-[10px] font-mono text-[#8A919C] tracking-wider">
            CYCLES COMPLETED
          </span>
          <span className="font-mono text-xs font-bold text-[#00E08A] glow-nominal">
            #{cyclesCompleted}
          </span>
        </div>
      </div>

      {/* Main Rail Area (7 connected steps on a single horizontal rail) */}
      <div className="flex-1 flex items-center justify-center px-2 py-2 overflow-x-auto overflow-y-hidden">
        <div className="flex items-center w-full min-w-[780px] justify-between relative">
          {HAR_STATES.map((step, index) => {
            const isDone = index < currentIndex;
            const isActive = index === currentIndex;
            const isPending = index > currentIndex;
            const isRejected = activeRejection?.step === step.id;
            const isJustAccepted = lastAcceptedState === step.id;

            // Step status styling
            let nodeBorder = "border-white/10";
            let nodeBg = "bg-[#171B21]";
            let nodeText = "text-[#565C66]";
            let glowClass = "";
            let animationClass = "";

            if (isRejected) {
              nodeBorder = "border-[#FF4D4F]";
              nodeBg = "bg-[#FF4D4F]/10";
              nodeText = "text-[#FF4D4F]";
              glowClass = "glow-critical";
              animationClass = "animate-reject-pulse";
            } else if (isJustAccepted) {
              nodeBorder = "border-[#00E08A]";
              nodeBg = "bg-[#00E08A]";
              nodeText = "text-[#0B0D10]";
              animationClass = "animate-stamp-flash";
            } else if (isDone) {
              nodeBorder = "border-[#00E08A]/60";
              nodeBg = "bg-[#00E08A]/10";
              nodeText = "text-[#00E08A]";
              glowClass = "glow-nominal";
            } else if (isActive) {
              nodeBorder = "border-[#FFB020]";
              nodeBg = "bg-[#FFB020]/10";
              nodeText = "text-[#FFB020]";
              glowClass = "glow-caution";
            }

            const containmentBadge = getContainmentBadge(step.id);

            return (
              <React.Fragment key={step.id}>
                {/* Node Column */}
                <div className="flex flex-col items-center relative z-10 w-[110px] md:w-[124px] shrink-0">
                  {/* Step Index Monospace Tag */}
                  <div className="flex items-center gap-1 mb-1.5 font-mono text-[10px]">
                    <span
                      className={`font-bold ${
                        isDone
                          ? "text-[#00E08A]"
                          : isActive
                          ? "text-[#FFB020]"
                          : isRejected
                          ? "text-[#FF4D4F]"
                          : "text-[#565C66]"
                      }`}
                    >
                      STEP 0{step.index}
                    </span>
                  </div>

                  {/* Squared-Off Node Box (No Pill, Raised-Bezel Depth) */}
                  <div
                    id={`sop-node-${step.id}`}
                    className={`relative w-full h-[62px] p-1.5 rounded-[2px] border bezel-depth flex flex-col items-center justify-between text-center select-none transition-colors duration-150 ${nodeBorder} ${nodeBg} ${glowClass} ${animationClass}`}
                  >
                    {/* Status Glyph Top */}
                    <div className="flex items-center justify-center">
                      {isRejected ? (
                        <ShieldAlert className="w-3.5 h-3.5 text-[#FF4D4F]" />
                      ) : isDone ? (
                        <Check className="w-3.5 h-3.5 text-[#00E08A]" />
                      ) : isActive ? (
                        <Disc className="w-3.5 h-3.5 text-[#FFB020] animate-spin" />
                      ) : (
                        <Clock className="w-3.5 h-3.5 text-[#565C66]" />
                      )}
                    </div>

                    {/* Step Label (Monospace) */}
                    <div className="w-full truncate">
                      <span
                        className={`font-mono text-[11px] font-bold tracking-tight block truncate ${
                          isDone
                            ? "text-[#E6E9ED]"
                            : isActive
                            ? "text-[#FFB020]"
                            : isRejected
                            ? "text-[#FF4D4F]"
                            : "text-[#565C66]"
                        }`}
                      >
                        {step.id.toUpperCase()}
                      </span>
                      <span className="font-mono text-[8px] text-[#8A919C] tracking-tighter block truncate">
                        {step.name.split("// ")[1] || step.id}
                      </span>
                    </div>
                  </div>

                  {/* Below-Node Area: Inline Containment Badges or Rejection Caption */}
                  <div className="h-[34px] flex flex-col items-center justify-start mt-1.5 w-full">
                    {isRejected && activeRejection ? (
                      <div
                        id={`rejection-caption-${step.id}`}
                        className="font-mono text-[8.5px] text-[#FF4D4F] bg-[#171B21] border border-[#FF4D4F]/50 px-1.5 py-0.5 rounded-[2px] text-center leading-tight shadow-sm max-w-[130px] truncate"
                        title={activeRejection.reason}
                      >
                        {activeRejection.reason}
                      </div>
                    ) : containmentBadge ? (
                      containmentBadge
                    ) : (
                      <span className="font-mono text-[8px] text-[#343A46] tracking-widest uppercase">
                        {isDone ? "VERIFIED" : isActive ? "IN-FLIGHT" : "QUEUED"}
                      </span>
                    )}
                  </div>
                </div>

                {/* Connecting Rail Segment Between Nodes (Rendered 6 times) */}
                {index < HAR_STATES.length - 1 && (
                  <div className="flex-1 min-w-[20px] max-w-[60px] h-[3px] bg-[#1E232B] relative mx-1 self-center -mt-[38px]">
                    {/* Done Segment: Solid Nominal */}
                    {index < currentIndex && (
                      <div className="absolute inset-0 bg-[#00E08A] shadow-[0_0_3px_rgba(0,224,138,0.5)]" />
                    )}

                    {/* Active Segment: Progress-Wipe Fill driven by gate stability progress */}
                    {index === currentIndex && (
                      <div
                        className="absolute top-0 left-0 bottom-0 bg-[#FFB020] shadow-[0_0_3px_rgba(255,176,32,0.6)]"
                        style={{
                          width: `${activeGateProgress}%`,
                          transition:
                            "width 180ms cubic-bezier(0.4, 0, 0.2, 1)",
                        }}
                      />
                    )}

                    {/* Directional Chevron Accent */}
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
                      <ChevronRight
                        className={`w-2.5 h-2.5 ${
                          index < currentIndex
                            ? "text-[#00E08A]"
                            : index === currentIndex
                            ? "text-[#FFB020]"
                            : "text-[#343A46]"
                        }`}
                      />
                    </div>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Flight Plan Rail Status Footer */}
      <div className="flex items-center justify-between pt-2 border-t border-white/5 font-mono text-[9px] text-[#565C66] shrink-0">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-[1px] bg-[#00E08A]" />
            <span>NOMINAL [DONE]</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-[1px] bg-[#FFB020]" />
            <span>ACTIVE [EVALUATING]</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-[1px] bg-[#FF4D4F]" />
            <span>TRANSIENT REJECT</span>
          </span>
        </div>
        <div className="flex items-center gap-2 text-[#8A919C]">
          <span>RAIL TELEMETRY: 10Hz SYNC</span>
          <span className="text-[#343A46]">•</span>
          <span>GATE HOLD: {frame.stability_count}/14 FRAMES</span>
        </div>
      </div>
    </div>
  );
}
