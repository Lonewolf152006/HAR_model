"use client";

import React, { useState } from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { Sliders, RotateCcw, ShieldCheck } from "lucide-react";

interface SliderRowProps {
  id: string;
  label: string;
  sublabel: string;
  value: number;
  min: number;
  max: number;
  step: number;
  displayValue: string;
  consequenceNote: string;
  ticks: { value: number; label: string }[];
  onChange: (val: number) => void;
}

function TuningSliderRow({
  id,
  label,
  sublabel,
  value,
  min,
  max,
  step,
  displayValue,
  consequenceNote,
  ticks,
  onChange,
}: SliderRowProps) {
  const [isDragging, setIsDragging] = useState(false);

  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));

  return (
    <div className="p-2 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle space-y-1">
      {/* Header: Parameter Name + Value */}
      <div className="flex items-center justify-between">
        <div>
          <label htmlFor={id} className="text-[10px] font-bold text-[#E6E9ED] block tracking-wide">
            {label}
          </label>
          <span className="text-[8.5px] text-[#565C66] tracking-wider uppercase">
            {sublabel}
          </span>
        </div>
        <div className="text-right">
          <span className="font-mono text-sm sm:text-base font-bold text-[#4DA3FF] tracking-tight">
            {displayValue}
          </span>
        </div>
      </div>

      {/* Slider Container with Custom Track & Ticks */}
      <div className="relative pt-0.5 pb-1">
        {/* Visual Track Background */}
        <div className="h-1.5 w-full bg-[#0E1015] border border-white/10 rounded-[1px] relative overflow-hidden pointer-events-none">
          {/* Filled Track in --accent-info */}
          <div
            className="h-full bg-[#4DA3FF] transition-all duration-75"
            style={{ width: `${pct}%` }}
          />
        </div>

        {/* Real Range Input */}
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onMouseDown={() => setIsDragging(true)}
          onMouseUp={() => setIsDragging(false)}
          onTouchStart={() => setIsDragging(true)}
          onTouchEnd={() => setIsDragging(false)}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className={`avionics-slider absolute top-0.5 inset-x-0 ${
            isDragging ? "glow-info" : ""
          }`}
        />

        {/* Ticks & Interval Markers */}
        <div className="flex justify-between items-center text-[7.5px] font-mono text-[#565C66] mt-1 select-none">
          {ticks.map((t, idx) => (
            <span key={idx} className="flex flex-col items-center">
              <span className="h-1 w-[1px] bg-white/15 mb-0.5" />
              <span>{t.label}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Engineering Consequence Note */}
      <div className="text-[8.5px] text-[#8A919C] leading-snug border-t border-white/5 pt-1">
        <span className="text-[#565C66] mr-1">// IMPACT:</span>
        <span>{consequenceNote}</span>
      </div>
    </div>
  );
}

export function GateThresholdTuningPanel() {
  const { thresholds, updateThresholds, restoreDefaultThresholds } = useTelemetry();

  return (
    <AvionicsPanel
      title="DECISION STABILIZER // GATE THRESHOLD TUNING"
      indexTag="05 // CALIBRATION"
      badge={
        <span className="px-2 py-0.5 font-mono text-[9px] font-bold rounded-[2px] border bg-[#4DA3FF]/10 text-[#4DA3FF] border-[#4DA3FF]/50 glow-info flex items-center gap-1">
          <Sliders className="w-2.5 h-2.5" />
          MUTABLE REGISTERS
        </span>
      }
      className="h-full flex flex-col justify-between p-3 font-mono overflow-hidden"
    >
      <div className="flex-1 min-h-0 flex flex-col justify-between gap-2 overflow-y-auto pr-0.5">
        {/* Row 1: Confidence Threshold */}
        <TuningSliderRow
          id="conf-threshold"
          label="CONFIDENCE THRESHOLD (GATE 1)"
          sublabel="MINIMUM POSTERIOR CLASS PROBABILITY"
          value={thresholds.confidence}
          min={0.3}
          max={0.95}
          step={0.01}
          displayValue={`${(thresholds.confidence * 100).toFixed(0)}% (${thresholds.confidence.toFixed(2)})`}
          consequenceNote="Lower = faster state trigger, higher risk of false-positive detections"
          ticks={[
            { value: 0.3, label: "30%" },
            { value: 0.52, label: "52% (DEF)" },
            { value: 0.75, label: "75%" },
            { value: 0.95, label: "95%" },
          ]}
          onChange={(val) => updateThresholds({ confidence: val })}
        />

        {/* Row 2: Stability Window */}
        <TuningSliderRow
          id="stability-window"
          label="STABILITY WINDOW (GATE 2)"
          sublabel="CONSECUTIVE FRAMES BUFFER REQUIREMENT"
          value={thresholds.stabilityWindow}
          min={2}
          max={20}
          step={1}
          displayValue={`${thresholds.stabilityWindow} FRAMES (${thresholds.stabilityWindow * 100}ms)`}
          consequenceNote="Higher = suppresses momentary flicker, adds hold-time latency before state lock"
          ticks={[
            { value: 2, label: "2F" },
            { value: 5, label: "5F (DEF)" },
            { value: 10, label: "10F" },
            { value: 15, label: "15F" },
            { value: 20, label: "20F" },
          ]}
          onChange={(val) => updateThresholds({ stabilityWindow: val })}
        />

        {/* Row 3: Cooldown Duration */}
        <TuningSliderRow
          id="cooldown-duration"
          label="TRANSITION COOLDOWN (GATE 3)"
          sublabel="MINIMUM LOCKOUT TIME BETWEEN ACCEPTED STATES"
          value={thresholds.cooldown}
          min={0.2}
          max={3.0}
          step={0.05}
          displayValue={`${thresholds.cooldown.toFixed(2)}s`}
          consequenceNote="Lower = rapid successive transitions permitted, risk of double-counting steps"
          ticks={[
            { value: 0.2, label: "0.2s" },
            { value: 0.85, label: "0.85s (DEF)" },
            { value: 1.5, label: "1.5s" },
            { value: 2.2, label: "2.2s" },
            { value: 3.0, label: "3.0s" },
          ]}
          onChange={(val) => updateThresholds({ cooldown: val })}
        />

        {/* Row 4: Motion Energy Floor */}
        <TuningSliderRow
          id="motion-floor"
          label="MOTION ENERGY FLOOR (GATE 4)"
          sublabel="MINIMUM KINEMATIC DISPLACEMENT TO PREVENT STATIC HALLUCINATION"
          value={thresholds.motionFloor}
          min={0.02}
          max={0.4}
          step={0.01}
          displayValue={`${thresholds.motionFloor.toFixed(2)} J`}
          consequenceNote="Lower = accepts delicate micro-movements, higher risk of static posture hallucination"
          ticks={[
            { value: 0.02, label: "0.02 J" },
            { value: 0.12, label: "0.12 J (DEF)" },
            { value: 0.25, label: "0.25 J" },
            { value: 0.4, label: "0.40 J" },
          ]}
          onChange={(val) => updateThresholds({ motionFloor: val })}
        />
      </div>

      {/* Footer: Restore Defaults Text Action */}
      <div className="pt-2 mt-1 border-t border-white/5 flex items-center justify-between text-[9px] text-[#565C66] shrink-0">
        <span className="flex items-center gap-1 text-[#8A919C]">
          <ShieldCheck className="w-3 h-3 text-[#00E08A]" />
          <span>REAL-TIME SHARED REGISTERS (LIVE FEED & SOP LINKED)</span>
        </span>

        <button
          type="button"
          onClick={restoreDefaultThresholds}
          className="flex items-center gap-1 text-[#8A919C] hover:text-[#E6E9ED] underline underline-offset-4 cursor-pointer transition-colors duration-150"
        >
          <RotateCcw className="w-2.5 h-2.5" />
          <span>RESTORE DEFAULTS</span>
        </button>
      </div>
    </AvionicsPanel>
  );
}
