"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { ArrowUp, ArrowDown, Minus, Activity, Gauge } from "lucide-react";

export function BandwidthLatencyPanel() {
  const { networkStats, streamStatus } = useTelemetry();
  const { bandwidthMbps, bandwidthTrend, latencyMs, latencyTrend } = networkStats;

  const isConnected = streamStatus === "CONNECTED";

  const renderTrendIcon = (trend: "up" | "down" | "stable", positiveIsUp: boolean) => {
    if (!isConnected || trend === "stable") {
      return <Minus className="w-3 h-3 text-[#565C66]" />;
    }
    if (trend === "up") {
      return (
        <ArrowUp
          className={`w-3 h-3 ${positiveIsUp ? "text-[#00E08A]" : "text-[#FF4D4F]"}`}
        />
      );
    }
    return (
      <ArrowDown
        className={`w-3 h-3 ${positiveIsUp ? "text-[#FF4D4F]" : "text-[#00E08A]"}`}
      />
    );
  };

  return (
    <AvionicsPanel
      title="BANDWIDTH & LATENCY"
      className="h-full flex flex-col justify-between overflow-hidden p-3"
    >
      <div className="grid grid-cols-2 gap-2 h-full font-mono">
        {/* Bandwidth Readout */}
        <div className="p-2.5 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle flex flex-col justify-between">
          <div className="flex items-center justify-between text-[9px] text-[#8A919C] uppercase pb-1 border-b border-white/5">
            <div className="flex items-center gap-1">
              <Gauge className="w-3 h-3 text-[#565C66]" />
              <span>CARRIER BITRATE</span>
            </div>
            {renderTrendIcon(bandwidthTrend, true)}
          </div>

          <div className="my-auto py-1">
            <div className="text-xl sm:text-2xl font-bold tracking-tight text-[#E6E9ED]">
              {isConnected ? bandwidthMbps.toFixed(1) : "0.0"}
              <span className="text-[10px] font-normal text-[#8A919C] ml-1">Mbps</span>
            </div>
            <div className="text-[8.5px] text-[#565C66] mt-0.5">
              {isConnected ? "HEVC ENCODED UPLINK" : "CARRIER UNLINKED"}
            </div>
          </div>

          <div className="text-[8px] text-[#565C66] pt-1 border-t border-white/5 flex items-center justify-between">
            <span>BURST LIMIT: 20.0</span>
            <span className={isConnected ? "text-[#00E08A]" : "text-[#565C66]"}>
              {isConnected ? "NOMINAL" : "ZERO"}
            </span>
          </div>
        </div>

        {/* Latency Readout */}
        <div className="p-2.5 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle flex flex-col justify-between">
          <div className="flex items-center justify-between text-[9px] text-[#8A919C] uppercase pb-1 border-b border-white/5">
            <div className="flex items-center gap-1">
              <Activity className="w-3 h-3 text-[#565C66]" />
              <span>ROUND-TRIP (RTT)</span>
            </div>
            {renderTrendIcon(latencyTrend, false)}
          </div>

          <div className="my-auto py-1">
            <div className="text-xl sm:text-2xl font-bold tracking-tight text-[#E6E9ED]">
              {isConnected ? latencyMs : "---"}
              <span className="text-[10px] font-normal text-[#8A919C] ml-1">ms</span>
            </div>
            <div className="text-[8.5px] text-[#565C66] mt-0.5">
              {isConnected ? "GROUND RELAY HOP 1" : "NO HANDSHAKE"}
            </div>
          </div>

          <div className="text-[8px] text-[#565C66] pt-1 border-t border-white/5 flex items-center justify-between">
            <span>TARGET: &lt;50ms</span>
            <span
              className={
                !isConnected
                  ? "text-[#565C66]"
                  : latencyMs < 30
                  ? "text-[#00E08A]"
                  : "text-[#FFB020]"
              }
            >
              {!isConnected ? "DISCONNECTED" : latencyMs < 30 ? "LOW JITTER" : "ELEVATED"}
            </span>
          </div>
        </div>
      </div>
    </AvionicsPanel>
  );
}
