"use client";

import React, { useId } from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { GATE_THRESHOLDS } from "@/lib/constants";
import { TrendingUp, AlertTriangle } from "lucide-react";

interface ConfidenceSparklineProps {
  confidence: number;
  history: number[];
}

export function ConfidenceSparkline({
  confidence,
  history,
}: ConfidenceSparklineProps) {
  const gradientId = useId();
  const threshold = GATE_THRESHOLDS.CONFIDENCE_MIN; // 0.75
  const isPassing = confidence >= threshold;

  // Compute SVG dimensions and points
  const width = 320;
  const height = 90;
  const paddingX = 4;
  const paddingTop = 12;
  const paddingBottom = 16;
  const effectiveHeight = height - paddingTop - paddingBottom;

  // Scale: 0.40 to 1.00
  const minY = 0.4;
  const maxY = 1.0;

  const getY = (val: number) => {
    const clamped = Math.max(minY, Math.min(maxY, val));
    const normalized = (clamped - minY) / (maxY - minY);
    return height - paddingBottom - normalized * effectiveHeight;
  };

  const points = history.map((val, idx) => {
    const x =
      paddingX +
      (idx / Math.max(1, history.length - 1)) * (width - paddingX * 2);
    const y = getY(val);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  const pathD = `M ${points.join(" L ")}`;
  const areaD = `${pathD} L ${width - paddingX},${height - paddingBottom} L ${paddingX},${height - paddingBottom} Z`;

  const thresholdY = getY(threshold);

  const minVal = Math.min(...history);
  const maxVal = Math.max(...history);

  return (
    <AvionicsPanel
      title="CONFIDENCE SPARKLINE"
      indexTag="02 // TELEMETRY"
      badge={
        <div className="flex items-center gap-1.5 font-mono text-[10px]">
          {isPassing ? (
            <span className="text-[#00E08A] glow-nominal font-bold flex items-center gap-1 bg-[#00E08A]/10 px-1.5 py-0.5 rounded-[2px] border border-[#00E08A]/40">
              <TrendingUp className="w-3 h-3" />
              PASS
            </span>
          ) : (
            <span className="text-[#FF4D4F] glow-critical font-bold flex items-center gap-1 bg-[#FF4D4F]/10 px-1.5 py-0.5 rounded-[2px] border border-[#FF4D4F]/50 animate-pulse">
              <AlertTriangle className="w-3 h-3" />
              DIP DETECTED
            </span>
          )}
        </div>
      }
      bracketColor={isPassing ? "border-[#00E08A]/50" : "border-[#FF4D4F]"}
    >
      {/* Real-time Numeric Readout Header */}
      <div className="flex items-baseline justify-between mb-2">
        <div className="flex items-baseline gap-2">
          <span
            className={`font-mono text-2xl font-bold tracking-tight ${
              isPassing ? "text-[#00E08A]" : "text-[#FF4D4F]"
            }`}
          >
            {(confidence * 100).toFixed(0)}%
          </span>
          <span className="font-mono text-[10px] text-[#8A919C]">
            (p = {confidence.toFixed(2)})
          </span>
        </div>

        <div className="flex items-center gap-3 font-mono text-[9px] text-[#565C66]">
          <span>MIN: {(minVal * 100).toFixed(0)}%</span>
          <span>MAX: {(maxVal * 100).toFixed(0)}%</span>
          <span className="text-[#FFB020]">THR: 75%</span>
        </div>
      </div>

      {/* SVG Sparkline Graph */}
      <div className="relative w-full bg-[#0E1116] bezel-depth-subtle border border-white/10 rounded-[2px] p-1 overflow-hidden">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-20 overflow-visible"
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

          {/* Background Grid Lines */}
          <line
            x1={paddingX}
            y1={getY(0.9)}
            x2={width - paddingX}
            y2={getY(0.9)}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="1"
          />
          <line
            x1={paddingX}
            y1={getY(0.6)}
            x2={width - paddingX}
            y2={getY(0.6)}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="1"
          />

          {/* Horizontal Reference Line (Threshold 0.75) */}
          <line
            x1={paddingX}
            y1={thresholdY}
            x2={width - paddingX}
            y2={thresholdY}
            stroke="#FFB020"
            strokeWidth="1.2"
            strokeDasharray="4 3"
            opacity="0.9"
          />

          {/* Threshold Label Text */}
          <text
            x={width - 54}
            y={thresholdY - 3}
            fill="#FFB020"
            fontSize="8"
            fontFamily="monospace"
            fontWeight="bold"
          >
            GATE 0.75
          </text>

          {/* Area Fill */}
          <path d={areaD} fill={`url(#${gradientId})`} />

          {/* Sparkline Stroke */}
          <path
            d={pathD}
            fill="none"
            stroke={isPassing ? "#00E08A" : "#FF4D4F"}
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Current Value Head Dot */}
          {points.length > 0 && (
            <circle
              cx={points[points.length - 1].split(",")[0]}
              cy={points[points.length - 1].split(",")[1]}
              r="3.5"
              fill={isPassing ? "#00E08A" : "#FF4D4F"}
              stroke="#0E1116"
              strokeWidth="1"
            />
          )}
        </svg>

        {/* Rolling Buffer Indicator Footer */}
        <div className="flex items-center justify-between mt-1 px-1 font-mono text-[9px] text-[#565C66]">
          <span>T -5.0s (50 SAMPLES @ 10Hz)</span>
          <span>NOW</span>
        </div>
      </div>
    </AvionicsPanel>
  );
}
