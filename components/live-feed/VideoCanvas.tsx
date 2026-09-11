"use client";

import React, { useState, useRef } from "react";
import { useTelemetry } from "@/context/TelemetryContext";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { Crosshair, Radio, Target, Activity, Cpu, ShieldCheck, AlertTriangle } from "lucide-react";

export function VideoCanvas() {
  const {
    currentState,
    expectedNext,
    confidence,
    boundingBoxes,
    frame,
    cyclesCompleted,
    activeAlert,
    isReducedMotion,
  } = useTelemetry();

  const containerRef = useRef<HTMLDivElement>(null);
  const [reticlePos, setReticlePos] = useState({ x: 50, y: 50 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    setReticlePos({
      x: +Math.max(0, Math.min(100, x)).toFixed(1),
      y: +Math.max(0, Math.min(100, y)).toFixed(1),
    });
  };

  return (
    <AvionicsPanel
      key="avionics-video-canvas"
      className={`p-0 overflow-hidden bg-[#0A0C0F] border h-full flex flex-col min-h-0 shadow-2xl transition-colors duration-150 ${
        activeAlert?.active
          ? !isReducedMotion
            ? "animate-alert-border-sustained border-[#FF4D4F]"
            : "border-[#FF4D4F]"
          : "border-white/15"
      }`}
      bracketColor={activeAlert?.active ? "border-[#FF4D4F]" : "border-[#00E08A]"}
    >
      {/* Top Header Flight Strip */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-[#12151A] border-b border-white/10 font-mono text-xs select-none shrink-0">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-[#00E08A] glow-nominal animate-pulse" />
          <span className="font-bold text-[#E6E9ED] tracking-wider text-xs">
            PRIMARY CAMERA FEED 01
          </span>
          <span className="text-[10px] text-[#565C66]">
            [1080p60 | SONY IMX477 MIPI-CSI2]
          </span>
        </div>

        <div className="flex items-center gap-3 text-[10px]">
          <span className="text-[#8A919C] flex items-center gap-1">
            <Radio className="w-3 h-3 text-[#00E08A]" />
            <span>59.8 FPS</span>
          </span>
          <span className="px-1.5 py-0.5 bg-[#1A0E10] border border-[#FF4D4F]/50 text-[#FF4D4F] font-bold rounded-[2px] flex items-center gap-1.5 bezel-depth-subtle">
            <span className="w-1.5 h-1.5 rounded-full bg-[#FF4D4F] glow-critical animate-ping" />
            REC
          </span>
        </div>
      </div>

      {/* Main Video Viewport (Takes flex-1 min-h-0 to perfectly fill available vertical height) */}
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className="flex-1 min-h-0 relative w-full bg-[#07080B] lens-vignette select-none overflow-hidden cursor-crosshair group flex items-center justify-center"
      >
        {/* Live Camera Stream from Edge Vision Hub (Camo Studio / Video Feed) */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/video_feed"
          alt="Live Camera Feed"
          className="absolute inset-0 w-full h-full object-cover z-0"
          onError={(e) => {
            (e.currentTarget as HTMLElement).style.display = "none";
          }}
          onLoad={(e) => {
            (e.currentTarget as HTMLElement).style.display = "block";
          }}
        />

        {/* Subtle Background Perspective Grid & Microgravity Workstation Simulation */}
        <div className="absolute inset-0 bg-dot-grid opacity-25 pointer-events-none" />

        {/* Experiment Workstation 3D Wireframe Depth Overlay */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none opacity-25"
          preserveAspectRatio="none"
          viewBox="0 0 1000 560"
        >
          {/* Workstation table outline */}
          <polygon
            points="100,480 900,480 820,240 180,240"
            fill="none"
            stroke="#4DA3FF"
            strokeWidth="1"
            strokeDasharray="4 4"
          />
          {/* Central alignment cross */}
          <line
            x1="500"
            y1="0"
            x2="500"
            y2="560"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="1"
          />
          <line
            x1="0"
            y1="280"
            x2="1000"
            y2="280"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="1"
          />
          {/* Containment Zone Bracket Lines */}
          <rect
            x="180"
            y="170"
            width="400"
            height="260"
            fill="rgba(77, 163, 255, 0.02)"
            stroke="rgba(77, 163, 255, 0.2)"
            strokeWidth="1"
          />
          {/* Exterior Staging Bracket */}
          <rect
            x="680"
            y="230"
            width="180"
            height="180"
            fill="rgba(255, 176, 32, 0.02)"
            stroke="rgba(255, 176, 32, 0.25)"
            strokeWidth="1"
            strokeDasharray="3 3"
          />
          <text
            x="690"
            y="250"
            fill="#8A919C"
            fontSize="11"
            fontFamily="monospace"
          >
            EXTERIOR BRACKET
          </text>
        </svg>

        {/* Simulated Microgravity Astronaut Arm / Glove Rig Silhouette */}
        <div
          className={`absolute transition-all duration-700 ease-out pointer-events-none opacity-30 ${
            currentState === "pick_red" || currentState === "pick_blue"
              ? "right-1/4 top-1/3"
              : currentState === "place_red_out"
              ? "right-[15%] top-1/3"
              : "right-1/3 top-1/2"
          }`}
        >
          <div className="w-32 h-20 border-t border-l border-dashed border-white/20 rounded-tl-3xl transform rotate-12" />
        </div>

        {/* Dynamic AI YOLOv8 Bounding Boxes with Tight Glow */}
        {boundingBoxes.map((box) => {
          const glowClass =
            box.color === "#FF4D4F"
              ? "glow-critical"
              : box.color === "#00E08A"
              ? "glow-nominal"
              : "glow-info";

          return (
            <div
              key={box.id}
              className="absolute transition-all duration-300 ease-out pointer-events-none"
              style={{
                left: `${box.x}%`,
                top: `${box.y}%`,
                width: `${box.w}%`,
                height: `${box.h}%`,
              }}
            >
              {/* Box Border & Corner Tabs */}
              <div
                className="w-full h-full border relative"
                style={{ borderColor: box.color }}
              >
                {/* Corner tick marks */}
                <span
                  className="absolute -top-1 -left-1 w-1.5 h-1.5 border-t border-l"
                  style={{ borderColor: box.color }}
                />
                <span
                  className="absolute -top-1 -right-1 w-1.5 h-1.5 border-t border-r"
                  style={{ borderColor: box.color }}
                />
                <span
                  className="absolute -bottom-1 -left-1 w-1.5 h-1.5 border-b border-l"
                  style={{ borderColor: box.color }}
                />
                <span
                  className="absolute -bottom-1 -right-1 w-1.5 h-1.5 border-b border-r"
                  style={{ borderColor: box.color }}
                />

                {/* Box Tag Label with tight box-shadow glow (right-aligned if on right half to prevent overflow clipping) */}
                <div
                  className={`absolute -top-5 ${
                    box.x > 50 ? "right-0" : "left-0"
                  } px-1.5 py-0.5 text-[9px] font-mono font-bold tracking-tight whitespace-nowrap rounded-[1px] flex items-center gap-1.5 bezel-depth-subtle z-20 ${glowClass}`}
                  style={{
                    backgroundColor: "#12151A",
                    color: box.color,
                    border: `1px solid ${box.color}60`,
                  }}
                >
                  <span>{box.label}</span>
                  <span className="opacity-80">
                    {(box.confidence * 100).toFixed(0)}%
                  </span>
                  <span
                    className={`px-1 py-0.2 text-[8px] rounded-[1px] border font-bold uppercase shrink-0 ${
                      box.status === "OUTSIDE"
                        ? "bg-[#FFB020]/20 text-[#FFB020] border-[#FFB020]/50 glow-caution"
                        : box.status === "HELD"
                        ? "bg-[#4DA3FF]/20 text-[#4DA3FF] border-[#4DA3FF]/50 glow-info animate-pulse"
                        : "bg-[#00E08A]/20 text-[#00E08A] border-[#00E08A]/50 glow-nominal"
                    }`}
                  >
                    {box.status || "INSIDE"}
                  </span>
                </div>
              </div>
            </div>
          );
        })}

        {/* Interactive Reticle Tracking Cursor */}
        <div
          className={`absolute pointer-events-none transition-opacity duration-150 ${
            isHovered ? "opacity-100" : "opacity-0"
          }`}
          style={{
            left: `${reticlePos.x}%`,
            top: `${reticlePos.y}%`,
            transform: "translate(-50%, -50%)",
          }}
        >
          <div className="relative w-8 h-8 flex items-center justify-center">
            <Crosshair className="w-6 h-6 text-[#00E08A]/80" />
            <span className="absolute -top-3 text-[8px] font-mono text-[#00E08A] whitespace-nowrap glow-nominal px-1 bg-[#12151A]">
              [{reticlePos.x}, {reticlePos.y}]
            </span>
          </div>
        </div>

        {/* Corner HUD Telemetry Overlays (Solid #0B0D10 Bezel Depth, No Glassmorphism) */}
        {/* Top-Left: State & Prediction */}
        <div className="absolute top-2.5 left-2.5 bg-[#0B0D10] bezel-depth border border-white/20 p-2 rounded-[2px] font-mono text-xs pointer-events-none space-y-0.5 shadow-xl">
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] text-[#8A919C]">STATE:</span>
            <span className="text-[#00E08A] font-bold tracking-wider flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#00E08A] glow-nominal" />
              {currentState.toUpperCase()}
            </span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px]">
            <span className="text-[9px] text-[#8A919C]">NEXT:</span>
            <span className="text-[#E6E9ED]">{expectedNext}</span>
          </div>
          <div className="flex items-center gap-1.5 text-[10px]">
            <span className="text-[9px] text-[#8A919C]">CONF:</span>
            <span className="text-[#00E08A] font-semibold">{confidence}</span>
            <span className="text-[8px] text-[#565C66]">(P ≥ 0.75 REQ)</span>
          </div>
        </div>

        {/* In-Panel HUD Alert Caption directly under State Box (Visible during alert, in --accent-critical) */}
        {activeAlert?.active && (
          <div
            id="hud-causal-violation-caption"
            className="absolute top-[84px] left-2.5 max-w-[500px] bg-[#1A0E10] border border-[#FF4D4F] px-2.5 py-1 rounded-[2px] bezel-depth font-mono text-[10px] text-[#FF4D4F] pointer-events-none flex items-center gap-1.5 shadow-2xl z-20"
          >
            <AlertTriangle className="w-3.5 h-3.5 text-[#FF4D4F] shrink-0" />
            <span className="font-bold tracking-tight leading-tight" title={activeAlert.reason}>
              {activeAlert.reason}
            </span>
          </div>
        )}

        {/* Top-Right: Target Calibration */}
        <div className="absolute top-2.5 right-2.5 bg-[#0B0D10] bezel-depth border border-white/15 p-2 rounded-[2px] font-mono text-[9px] text-right pointer-events-none space-y-0.5 shadow-xl">
          <div className="text-[#8A919C]">STATION: ISS-COLUMBUS</div>
          <div className="text-[#E6E9ED]">FRAME: #{frame.frame_id}</div>
          <div className="text-[#00E08A] flex items-center justify-end gap-1">
            <span className="w-1 h-1 rounded-full bg-[#00E08A] glow-nominal" />
            <span>POSE: 33 PTS LOCKED</span>
          </div>
        </div>

        {/* Bottom-Left: Motion & Kinematics */}
        <div className="absolute bottom-2.5 left-2.5 bg-[#0B0D10] bezel-depth border border-white/15 px-2 py-1 rounded-[2px] font-mono text-[9px] pointer-events-none flex items-center gap-2">
          <Target className="w-3 h-3 text-[#4DA3FF]" />
          <span className="text-[#8A919C]">ENERGY:</span>
          <span className="text-[#E6E9ED] font-semibold">
            {frame.motion_energy} J
          </span>
          <span className="text-[#00E08A] flex items-center gap-1">
            <span className="w-1 h-1 rounded-full bg-[#00E08A] glow-nominal" />
            TRACKING
          </span>
        </div>

        {/* Bottom-Right: Reticle Coordinate Readout */}
        <div className="absolute bottom-2.5 right-2.5 bg-[#0B0D10] bezel-depth border border-white/15 px-2 py-1 rounded-[2px] font-mono text-[9px] text-[#8A919C] pointer-events-none">
          X: {reticlePos.x.toFixed(1)}% Y: {reticlePos.y.toFixed(1)}%
        </div>

        {/* CRT Scanline Overlay */}
        <div className="scanline-overlay absolute inset-0 pointer-events-none opacity-20" />
      </div>

      {/* Trimmed Bottom Stat Strip: ENERGY, LATENCY, CYCLES (Equally distributed, no dead empty space) */}
      <div className="grid grid-cols-3 divide-x divide-white/10 bg-[#12151A] bezel-depth-subtle border-t border-white/10 py-2 font-mono text-xs select-none shrink-0">
        <div className="flex items-center justify-center gap-2 px-3">
          <Activity className="w-3.5 h-3.5 text-[#00E08A] glow-nominal shrink-0" />
          <span className="text-[10px] text-[#565C66] tracking-wider uppercase">ENERGY:</span>
          <span className="text-[#00E08A] font-bold tracking-tight">
            {frame.motion_energy} J
          </span>
        </div>

        <div className="flex items-center justify-center gap-2 px-3">
          <Cpu className="w-3.5 h-3.5 text-[#4DA3FF] glow-info shrink-0" />
          <span className="text-[10px] text-[#565C66] tracking-wider uppercase">LATENCY:</span>
          <span className="text-[#4DA3FF] font-bold tracking-tight">
            {frame.latency_ms.toFixed(1)} ms
          </span>
        </div>

        <div className="flex items-center justify-center gap-2 px-3">
          <ShieldCheck className="w-3.5 h-3.5 text-[#00E08A] glow-nominal shrink-0" />
          <span className="text-[10px] text-[#565C66] tracking-wider uppercase">CYCLES:</span>
          <span className="text-[#E6E9ED] font-bold tracking-tight">
            #{cyclesCompleted}
          </span>
        </div>
      </div>
    </AvionicsPanel>
  );
}
