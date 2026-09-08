"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { ContainmentStatus } from "@/lib/types";
import { Box } from "lucide-react";

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
      className="h-full flex flex-col justify-between overflow-hidden p-2.5"
    >
      {/* Single Containment Reference Row */}
      <div className="font-mono text-xs overflow-y-auto pr-0.5">
        {/* Main Stowage Box */}
        <div className="flex items-center justify-between px-2.5 py-2 bg-[#171B21] bezel-depth-subtle border border-white/5 rounded-[2px]">
          <div className="flex items-center gap-2">
            <Box className="w-3.5 h-3.5 text-[#8A919C]" />
            <div>
              <span className="text-[11px] font-semibold text-[#E6E9ED] block">
                Main Stowage Box
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
      </div>
    </AvionicsPanel>
  );
}
