"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { CAMERA_SOURCES } from "@/lib/constants";
import { CameraSourceId } from "@/lib/types";
import { Camera, Check, Cpu } from "lucide-react";

export function CameraSourcePanel() {
  const { cameraSource, setCameraSource } = useTelemetry();

  const activeConfig =
    CAMERA_SOURCES.find((c) => c.id === cameraSource) || CAMERA_SOURCES[0];

  return (
    <AvionicsPanel
      title="OPTICAL SENSOR CAMERA SOURCE"
      badge={
        <span className="px-2 py-0.5 font-mono text-[9px] font-bold rounded-[2px] border bg-[#00E08A]/10 text-[#00E08A] border-[#00E08A]/50 glow-nominal flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00E08A] animate-pulse" />
          ACTIVE BUS
        </span>
      }
      className="flex flex-col justify-between p-3 font-mono"
    >
      <div className="space-y-2.5">
        <div className="flex items-center justify-between text-[9px] text-[#565C66] tracking-wider uppercase pb-1 border-b border-white/5">
          <span>HARDWARE FEED BUS</span>
          <span>SELECT SWITCH</span>
        </div>

        {/* Segmented Instrument Switch (No Dropdowns, Squared Segments, 2-4px radius) */}
        <div className="grid grid-cols-3 gap-1.5 p-1 bg-[#0E1015] border border-white/10 rounded-[2px] bezel-depth-subtle">
          {CAMERA_SOURCES.map((src) => {
            const isSelected = src.id === cameraSource;
            return (
              <button
                key={src.id}
                type="button"
                onClick={() => setCameraSource(src.id as CameraSourceId)}
                className={`py-1.5 px-2 text-center text-[10px] font-mono font-bold tracking-tight rounded-[2px] border transition-all duration-150 flex items-center justify-center gap-1.5 ${
                  isSelected
                    ? "bg-[#171B21] text-[#00E08A] border-[#00E08A]/70 shadow-[inset_0_1px_0_rgba(255,255,255,0.1),0_0_6px_rgba(0,224,138,0.25)]"
                    : "bg-transparent text-[#8A919C] border-transparent hover:text-[#E6E9ED] hover:bg-white/[0.03]"
                }`}
              >
                {isSelected && <Check className="w-2.5 h-2.5 text-[#00E08A]" />}
                <span className="truncate">{src.label}</span>
              </button>
            );
          })}
        </div>

        {/* Selected Sensor Telemetry & Specification Readout */}
        <div className="p-2.5 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle">
          <div className="flex items-center justify-between text-[8.5px] text-[#565C66] tracking-wider uppercase mb-1">
            <span className="flex items-center gap-1">
              <Camera className="w-3 h-3 text-[#8A919C]" />
              <span>SENSOR INTERFACE SPECIFICATION</span>
            </span>
            <span className="text-[#4DA3FF]">LOCKED</span>
          </div>

          <div className="text-xs sm:text-sm font-bold text-[#E6E9ED] tracking-wide">
            {activeConfig.resolutionFps} | {activeConfig.sensor}
          </div>

          <div className="flex items-center justify-between text-[9px] text-[#8A919C] mt-1.5 pt-1.5 border-t border-white/5">
            <span className="flex items-center gap-1">
              <Cpu className="w-2.5 h-2.5 text-[#565C66]" />
              <span>CARRIER LINK: {activeConfig.bus}</span>
            </span>
            <span className="text-[#00E08A]">SYNCHRONIZED</span>
          </div>
        </div>
      </div>
    </AvionicsPanel>
  );
}
