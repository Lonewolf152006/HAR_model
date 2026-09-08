"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { ContainmentState, ContainmentStatus } from "@/lib/types";
import { Box } from "lucide-react";

interface ContainmentBadgesCardProps {
  containment: ContainmentState;
}

export function ContainmentBadgesCard({ containment }: ContainmentBadgesCardProps) {
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
      title="GEOMETRIC CONTAINMENT"
      className="shrink-0 p-2.5"
    >
      <div className="space-y-1.5 font-mono text-xs overflow-y-auto max-h-[130px] pr-0.5">
        {/* Main Container */}
        <div className="flex items-center justify-between px-2 py-1 bg-[#171B21] bezel-depth-subtle border border-white/5 rounded-[2px]">
          <div className="flex items-center gap-1.5">
            <Box className="w-3 h-3 text-[#8A919C]" />
            <span className="text-[11px] text-[#E6E9ED]">Main Stowage Box</span>
          </div>
          <span
            className={`px-1.5 py-0.5 text-[9px] font-bold rounded-[2px] border bezel-depth-subtle ${getBadgeStyle(
              containment.main_box
            )}`}
          >
            {containment.main_box}
          </span>
        </div>

        {/* Red Sample Cube */}
        <div className="flex items-center justify-between px-2 py-1 bg-[#171B21] bezel-depth-subtle border border-white/5 rounded-[2px]">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-[1px] bg-[#FF4D4F] glow-critical shrink-0" />
            <span className="text-[11px] text-[#E6E9ED]">Red Cube [Sample-A]</span>
          </div>
          <span
            className={`px-1.5 py-0.5 text-[9px] font-bold rounded-[2px] border bezel-depth-subtle ${getBadgeStyle(
              containment.red_box
            )}`}
          >
            {containment.red_box}
          </span>
        </div>

        {/* Blue Sample Cube */}
        <div className="flex items-center justify-between px-2 py-1 bg-[#171B21] bezel-depth-subtle border border-white/5 rounded-[2px]">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-[1px] bg-[#00E08A] glow-nominal shrink-0" />
            <span className="text-[11px] text-[#E6E9ED]">Blue Cube [Sample-B]</span>
          </div>
          <span
            className={`px-1.5 py-0.5 text-[9px] font-bold rounded-[2px] border bezel-depth-subtle ${getBadgeStyle(
              containment.blue_box
            )}`}
          >
            {containment.blue_box}
          </span>
        </div>
      </div>
    </AvionicsPanel>
  );
}
