"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { ContainmentStatus } from "@/lib/types";
import { Box, Layers, Cuboid } from "lucide-react";

export function ContainmentQuickView() {
  const { containment } = useTelemetry();

  const getBadgeStyle = (status: ContainmentStatus | string) => {
    switch (status) {
      case "INSIDE":
        return "bg-[#171B21] text-[#00E08A] border-[#00E08A]/50 glow-nominal";
      case "OUTSIDE":
        return "bg-[#171B21] text-[#FFB020] border-[#FFB020]/50 glow-caution";
      case "HELD":
        return "bg-[#171B21] text-[#4DA3FF] border-[#4DA3FF]/60 glow-info animate-pulse";
      case "OPEN":
        return "bg-[#171B21] text-[#00E08A] border-[#00E08A]/50 glow-nominal";
      case "CLOSED":
        return "bg-[#171B21] text-[#8A919C] border-white/10";
      default:
        return "bg-[#171B21] text-[#8A919C] border-white/10";
    }
  };

  return (
    <AvionicsPanel
      title="CONTAINMENT QUICK-VIEW"
      indexTag="04 // 2.5D GEOMETRY"
      badge={
        <div className="flex items-center gap-1 font-mono text-[9px] text-[#4DA3FF]">
          <Layers className="w-3 h-3 glow-info" />
          <span>CALIBRATED</span>
        </div>
      }
      className="h-full flex flex-col justify-between overflow-hidden p-2.5"
    >
      {/* 3-Row Compact Single-Column List */}
      <div className="space-y-2 font-mono text-xs overflow-y-auto pr-0.5">
        {/* Row 1: Main Stowage Box */}
        <div className="flex items-center justify-between px-2.5 py-2 bg-[#171B21] bezel-depth-subtle border border-white/5 rounded-[2px]">
          <div className="flex items-center gap-2">
            <Box className="w-3.5 h-3.5 text-[#8A919C]" />
            <div>
              <span className="text-[11px] font-semibold text-[#E6E9ED] block">
                Main Stowage Box
              </span>
              <span className="text-[8px] text-[#565C66] block">
                PRIMARY VOLUME
              </span>
            </div>
          </div>
          <span
            id="containment-main-box"
            className={`px-2 py-0.5 text-[9.5px] font-bold rounded-[2px] border bezel-depth-subtle ${getBadgeStyle(
              containment.main_box
            )}`}
          >
            {containment.main_box}
          </span>
        </div>

        {/* Row 2: Red Sample Cube */}
        <div className="flex items-center justify-between px-2.5 py-2 bg-[#171B21] bezel-depth-subtle border border-white/5 rounded-[2px]">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-[1px] bg-[#FF4D4F] glow-critical shrink-0" />
            <div>
              <span className="text-[11px] font-semibold text-[#E6E9ED] block">
                Red Cube [Sample-A]
              </span>
              <span className="text-[8px] text-[#565C66] block">
                PRIMARY EXTRACT
              </span>
            </div>
          </div>
          <span
            id="containment-red-box"
            className={`px-2 py-0.5 text-[9.5px] font-bold rounded-[2px] border bezel-depth-subtle ${getBadgeStyle(
              containment.red_box
            )}`}
          >
            {containment.red_box}
          </span>
        </div>

        {/* Row 3: Blue Sample Cube */}
        <div className="flex items-center justify-between px-2.5 py-2 bg-[#171B21] bezel-depth-subtle border border-white/5 rounded-[2px]">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-[1px] bg-[#4DA3FF] glow-info shrink-0" />
            <div>
              <span className="text-[11px] font-semibold text-[#E6E9ED] block">
                Blue Cube [Sample-B]
              </span>
              <span className="text-[8px] text-[#565C66] block">
                SECONDARY REPLACEMENT
              </span>
            </div>
          </div>
          <span
            id="containment-blue-box"
            className={`px-2 py-0.5 text-[9.5px] font-bold rounded-[2px] border bezel-depth-subtle ${getBadgeStyle(
              containment.blue_box
            )}`}
          >
            {containment.blue_box}
          </span>
        </div>
      </div>

      {/* 2.5D Volumetric Geometry Coordinate Status Footer */}
      <div className="flex flex-col space-y-0.5 pt-1.5 border-t border-white/5 font-mono text-[8px] text-[#565C66] shrink-0">
        <div className="flex items-center justify-between">
          <span className="flex items-center gap-1">
            <Cuboid className="w-2.5 h-2.5 text-[#4DA3FF]" />
            <span>IOU THRESHOLD: &gt;0.35 INTERSECTION</span>
          </span>
          <span className="text-[#00E08A]">SYNCHRONIZED</span>
        </div>
        <div className="text-[#8A919C] text-[7.5px] truncate">
          BOUNDS: X[-0.42..+0.42] Y[-0.30..+0.30] Z[0..0.50]
        </div>
      </div>
    </AvionicsPanel>
  );
}
