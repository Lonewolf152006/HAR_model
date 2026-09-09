"use client";

import React, { useId } from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { HarState } from "@/lib/types";
import { STATE_MAP, GATE_THRESHOLDS } from "@/lib/constants";
import { useTelemetry } from "@/context/TelemetryContext";
import { ArrowRight, Clock, Activity, TrendingUp, AlertTriangle } from "lucide-react";

interface StateConfidenceCardProps {
  currentState: HarState;
  expectedNext: HarState;
  stateDurationMs: number;
  confidence: number;
  confidenceHistory: number[];
}

export function StateConfidenceCard({
  currentState,
  expectedNext,
  stateDurationMs,
  confidence,
  confidenceHistory,
}: StateConfidenceCardProps) {
  const { thresholds } = useTelemetry();
  const gradientId = useId();
  const threshold = thresholds?.confidence ?? GATE_THRESHOLDS.CONFIDENCE_MIN;
  const isPassing = confidence >= threshold;

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

  // SVG Sparkline dimensions (compact, reclaimed vertical height)
  const width = 320;
  const height = 46;
  const paddingX = 4;
  const paddingTop = 6;
  const paddingBottom = 6;
  const effectiveHeight = height - paddingTop - paddingBottom;

  const minY = 0.4;
  const maxY = 1.0;

  const getY = (val: number) => {
    const clamped = Math.max(minY, Math.min(maxY, val));
    const normalized = (clamped - minY) / (maxY - minY);
    return height - paddingBottom - normalized * effectiveHeight;
  };

  const points = confidenceHistory.map((val, idx) => {
    const x =
      paddingX +
      (idx / Math.max(1, confidenceHistory.length - 1)) * (width - paddingX * 2);
    const y = getY(val);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const pathD = `M ${points.join(" L ")}`;
  const areaD = `${pathD} L ${width - paddingX},${height - paddingBottom} L ${paddingX},${height - paddingBottom} Z`;
  const thresholdY = getY(threshold);

  const minVal = Math.min(...confidenceHistory);
  const maxVal = Math.max(...confidenceHistory);

  return (
    <AvionicsPanel
      title="STATE ENGINE & CONFIDENCE"
      badge={
        <span className="font-mono text-[10px] text-[#00E08A] bg-[#171B21] px-1.5 py-0.5 rounded-[2px] border border-[#00E08A]/40 flex items-center gap-1.5 glow-nominal bezel-depth-subtle">
          <Activity className="w-2.5 h-2.5 animate-pulse" />
          ACTIVE
        </span>
      }
      bracketColor="border-[#00E08A]/60"
      className="shrink-0 p-2.5"
    >
      <div className="space-y-2 font-mono">
        {/* State and Next Step in single compact section */}
        <div className="flex items-center justify-between pb-1.5 border-b border-white/5">
          <div className="min-w-0">
            <span className="text-[8px] text-[#565C66] tracking-wider block uppercase">
              CURRENT ACTION
            </span>
            <div className="text-sm font-bold text-[#00E08A] tracking-wide flex items-center gap-1.5 truncate">
              <span className="w-1.5 h-1.5 rounded-full bg-[#00E08A] glow-nominal shrink-0 animate-pulse" />
              <span className="truncate">{currentDef.name}</span>
            </div>
          </div>

          <div className="text-right shrink-0">
            <span className="text-[8px] text-[#565C66] tracking-wider block uppercase">
              NEXT EXPECTED
            </span>
            <div className="text-xs text-[#E6E9ED] font-semibold flex items-center justify-end gap-1">
              <ArrowRight className="w-3 h-3 text-[#4DA3FF] glow-info shrink-0" />
              <span>{nextDef.name}</span>
              <span className="text-[9px] text-[#8A919C]">({elapsedSec}s)</span>
            </div>
          </div>
        </div>

        {/* Embedded Compact Confidence Sparkline with Threshold Line */}
        <div>
          <div className="flex items-center justify-between text-[9px] text-[#8A919C] mb-1">
            <div className="flex items-center gap-2">
              <span className="text-[#E6E9ED] font-semibold">
                P = {confidence.toFixed(2)}
              </span>
              <span className="text-[#565C66]">
                [MIN: {(minVal * 100).toFixed(0)}% | MAX: {(maxVal * 100).toFixed(0)}%]
              </span>
            </div>
            <div className="flex items-center gap-1 font-bold text-[#FFB020]">
              <span>GATE THR: {Math.round(threshold * 100)}%</span>
            </div>
          </div>

          {/* SVG Sparkline Graph */}
          <div className="relative w-full bg-[#0E1116] bezel-depth-subtle border border-white/10 rounded-[2px] p-0.5 overflow-hidden">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="w-full h-12 overflow-visible"
              preserveAspectRatio="none"
            >
              <defs>
                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="0%"
                    stopColor={isPassing ? "#00E08A" : "#FF4D4F"}
                    stopOpacity="0.25"
                  />
                  <stop
                    offset="100%"
                    stopColor={isPassing ? "#00E08A" : "#FF4D4F"}
                    stopOpacity="0.0"
                  />
                </linearGradient>
              </defs>

              {/* Horizontal Reference Line (Threshold 0.75) */}
              <line
                x1={paddingX}
                y1={thresholdY}
                x2={width - paddingX}
                y2={thresholdY}
                stroke="#FFB020"
                strokeWidth="1.2"
                strokeDasharray="3 2"
                opacity="0.9"
              />

              {/* Area Fill */}
              <path d={areaD} fill={`url(#${gradientId})`} />

              {/* Sparkline Stroke */}
              <path
                d={pathD}
                fill="none"
                stroke={isPassing ? "#00E08A" : "#FF4D4F"}
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Current Value Head Dot */}
              {points.length > 0 && (
                <circle
                  cx={points[points.length - 1].split(",")[0]}
                  cy={points[points.length - 1].split(",")[1]}
                  r="3"
                  fill={isPassing ? "#00E08A" : "#FF4D4F"}
                  stroke="#0E1116"
                  strokeWidth="1"
                />
              )}
            </svg>

            {/* Rolling Buffer Indicator Footer */}
            <div className="flex items-center justify-between px-1 text-[8px] text-[#565C66]">
              <span>T -5.0s (50 SAMPLES @ 10Hz)</span>
            </div>
          </div>
        </div>
      </div>
    </AvionicsPanel>
  );
}
