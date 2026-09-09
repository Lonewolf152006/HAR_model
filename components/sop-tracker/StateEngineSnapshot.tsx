"use client";

import React, { useId } from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { GATE_THRESHOLDS } from "@/lib/constants";
import { Check, X, Gauge, Activity } from "lucide-react";

export function StateEngineSnapshot() {
  const { confidence, confidenceHistory, gates, frame, thresholds } = useTelemetry();
  const gradientId = useId();

  const threshold = thresholds?.confidence ?? GATE_THRESHOLDS.CONFIDENCE_MIN;
  const isPassing = confidence >= threshold;

  // Compute SVG dimensions and sparkline coordinates
  const width = 300;
  const height = 65;
  const paddingX = 4;
  const paddingTop = 8;
  const paddingBottom = 12;
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
  const areaD = `${pathD} L ${width - paddingX},${
    height - paddingBottom
  } L ${paddingX},${height - paddingBottom} Z`;
  const thresholdY = getY(threshold);

  const gateKeys = [
    { key: "confidence", label: "CONF", gate: gates.confidence },
    { key: "stability", label: "STAB", gate: gates.stability },
    { key: "cooldown", label: "COOL", gate: gates.cooldown },
    { key: "causal_logic", label: "FSM", gate: gates.causal_logic },
    { key: "motion", label: "MOTN", gate: gates.motion },
  ] as const;

  return (
    <AvionicsPanel
      title="STATE ENGINE SNAPSHOT"
      className="h-full flex flex-col justify-between overflow-hidden p-3 pb-3.5"
    >
      {/* Sparkline Numeric Readout + Compact Graph */}
      <div className="flex flex-col space-y-1 shrink-0">
        <div className="flex items-baseline justify-between font-mono">
          <div className="flex items-baseline gap-2">
            <span
              className={`text-xl font-bold tracking-tight ${
                isPassing ? "text-[#00E08A]" : "text-[#FF4D4F]"
              }`}
            >
              {(confidence * 100).toFixed(0)}%
            </span>
            <span className="text-[9px] text-[#8A919C]">
              p = {confidence.toFixed(2)}
            </span>
          </div>

          <div className="flex items-center gap-2 text-[8px] text-[#565C66]">
            <span>THR: {threshold.toFixed(2)}</span>
            <span className="text-[#343A46]">|</span>
            <span>BUFFER: 50 SAMPLES</span>
          </div>
        </div>

        {/* Compact SVG Sparkline */}
        <div className="relative w-full bg-[#0E1015] border border-white/5 rounded-[2px] bezel-depth-subtle overflow-hidden">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="w-full h-[62px] overflow-visible"
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="0%"
                  stopColor={isPassing ? "#00E08A" : "#FF4D4F"}
                  stopOpacity="0.2"
                />
                <stop
                  offset="100%"
                  stopColor={isPassing ? "#00E08A" : "#FF4D4F"}
                  stopOpacity="0.0"
                />
              </linearGradient>
            </defs>

            {/* Threshold Reference Line */}
            <line
              x1={paddingX}
              y1={thresholdY}
              x2={width - paddingX}
              y2={thresholdY}
              stroke="#FFB020"
              strokeWidth="1"
              strokeDasharray="3 2"
              opacity="0.8"
            />

            {/* Area Fill */}
            <path d={areaD} fill={`url(#${gradientId})`} />

            {/* Line Path */}
            <path
              d={pathD}
              fill="none"
              stroke={isPassing ? "#00E08A" : "#FF4D4F"}
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Head point */}
            {points.length > 0 && (
              <circle
                cx={points[points.length - 1].split(",")[0]}
                cy={points[points.length - 1].split(",")[1]}
                r="2.5"
                fill={isPassing ? "#00E08A" : "#FF4D4F"}
              />
            )}
          </svg>
        </div>
      </div>

      {/* Single-Line 5-Gate Pass/Fail Summary Strip */}
      <div className="flex flex-col space-y-1 mt-2">
        <div className="flex items-center justify-between font-mono text-[8px] text-[#565C66]">
          <span>DECISION STABILIZER GATES (SINGLE-STRIP)</span>
          <span>THRESHOLD PASS</span>
        </div>

        {/* 5 horizontal compact chips */}
        <div className="grid grid-cols-5 gap-1 font-mono text-[9px]">
          {gateKeys.map(({ key, label, gate }) => {
            const passed = gate.passed;
            return (
              <div
                key={key}
                id={`sop-gate-${key}`}
                className={`flex flex-col items-center justify-center py-1 px-0.5 rounded-[2px] border bezel-depth-subtle ${
                  passed
                    ? "bg-[#171B21] border-[#00E08A]/40 text-[#00E08A]"
                    : "bg-[#FF4D4F]/10 border-[#FF4D4F]/60 text-[#FF4D4F] animate-pulse"
                }`}
                title={`${label}: ${gate.reason}`}
              >
                <div className="flex items-center gap-0.5">
                  {passed ? (
                    <Check className="w-2.5 h-2.5 text-[#00E08A]" />
                  ) : (
                    <X className="w-2.5 h-2.5 text-[#FF4D4F]" />
                  )}
                  <span className="font-bold text-[8.5px]">{label}</span>
                </div>
                <span className="text-[7.5px] text-[#8A919C] tracking-tighter truncate max-w-[45px]">
                  {passed ? "PASS" : "FAIL"}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Dwell / Motion Kinematics Telemetry Strip */}
      <div className="flex items-center justify-between pt-2 pb-1 border-t border-white/5 font-mono text-[8.5px] text-[#8A919C] shrink-0 mb-0.5">
        <div className="flex items-center gap-1">
          <Activity className="w-3 h-3 text-[#4DA3FF]" />
          <span>DWELL: {(frame.state_duration_ms / 1000).toFixed(1)}s</span>
        </div>
        <div className="flex items-center gap-1">
          <Gauge className="w-3 h-3 text-[#FFB020]" />
          <span>ENERGY: {frame.motion_energy.toFixed(3)} J</span>
        </div>
      </div>
    </AvionicsPanel>
  );
}
