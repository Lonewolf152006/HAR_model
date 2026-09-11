"use client";

import React, { useState } from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { Wifi, WifiOff, RefreshCw, ShieldAlert } from "lucide-react";

export function StreamTargetPanel() {
  const {
    streamStatus,
    streamConfig,
    streamEvents,
    signalQuality,
    toggleStreamConnect,
    updateStreamConfig,
  } = useTelemetry();

  const [inputIp, setInputIp] = useState(streamConfig.ip);
  const [inputPort, setInputPort] = useState(streamConfig.port);

  const handleApplyConfig = (e: React.FormEvent) => {
    e.preventDefault();
    updateStreamConfig({ ip: inputIp, port: inputPort });
  };

  const getStatusBadge = () => {
    switch (streamStatus) {
      case "CONNECTED":
        return (
          <span className="px-2 py-0.5 font-mono text-[9.5px] font-bold rounded-[2px] border bg-[#00E08A]/10 text-[#00E08A] border-[#00E08A]/50 glow-nominal flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00E08A] animate-pulse" />
            CONNECTED
          </span>
        );
      case "RECONNECTING":
        return (
          <span className="px-2 py-0.5 font-mono text-[9.5px] font-bold rounded-[2px] border bg-[#FFB020]/10 text-[#FFB020] border-[#FFB020]/50 glow-caution flex items-center gap-1 animate-pulse">
            <RefreshCw className="w-2.5 h-2.5 animate-spin text-[#FFB020]" />
            RECONNECTING
          </span>
        );
      case "OFFLINE":
      default:
        return (
          <span className="px-2 py-0.5 font-mono text-[9.5px] font-bold rounded-[2px] border bg-[#FF4D4F]/10 text-[#FF4D4F] border-[#FF4D4F]/50 glow-critical flex items-center gap-1">
            <WifiOff className="w-2.5 h-2.5 text-[#FF4D4F]" />
            OFFLINE
          </span>
        );
    }
  };

  return (
    <AvionicsPanel
      title="STREAM TARGET & LINK HEALTH"
      badge={getStatusBadge()}
      className="h-full flex flex-col justify-between overflow-hidden p-3"
    >
      {/* Target Config Instrument Row */}
      <div className="p-2.5 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle shrink-0 font-mono">
        <div className="flex items-center justify-between pb-1.5 border-b border-white/5 text-[9px] text-[#565C66] tracking-wider uppercase mb-2">
          <span>UPLINK CONFIGURATION</span>
          <span>PROTOCOL: {streamConfig.protocol}</span>
        </div>

        <form
          onSubmit={handleApplyConfig}
          className="flex flex-wrap sm:flex-nowrap items-center gap-2"
        >
          {/* IP Input */}
          <div className="flex-1 min-w-[140px]">
            <label className="text-[8.5px] text-[#8A919C] block mb-0.5">
              TARGET IP HOST
            </label>
            <input
              type="text"
              value={inputIp}
              onChange={(e) => setInputIp(e.target.value)}
              className="w-full bg-[#0E1015] border border-white/15 px-2 py-1 rounded-[2px] text-xs text-[#E6E9ED] font-mono focus:border-[#00E08A]/60 outline-none bezel-depth-subtle"
              placeholder="10.0.4.128"
            />
          </div>

          {/* Port Input */}
          <div className="w-20">
            <label className="text-[8.5px] text-[#8A919C] block mb-0.5">
              PORT
            </label>
            <input
              type="text"
              value={inputPort}
              onChange={(e) => setInputPort(e.target.value)}
              className="w-full bg-[#0E1015] border border-white/15 px-2 py-1 rounded-[2px] text-xs text-[#E6E9ED] font-mono focus:border-[#00E08A]/60 outline-none bezel-depth-subtle"
              placeholder="8554"
            />
          </div>

          {/* Connect / Disconnect Action Button */}
          <div className="self-end shrink-0 pt-3 sm:pt-0">
            <button
              type="button"
              onClick={toggleStreamConnect}
              className={`flex items-center gap-1.5 px-3 py-1 rounded-[2px] font-mono text-xs font-bold border transition-colors bezel-depth-subtle ${
                streamStatus === "CONNECTED"
                  ? "bg-[#1A0E10] hover:bg-[#2A1417] text-[#FF4D4F] border-[#FF4D4F]/50 hover:border-[#FF4D4F]"
                  : "bg-[#0E1A14] hover:bg-[#14261D] text-[#00E08A] border-[#00E08A]/50 hover:border-[#00E08A]"
              }`}
            >
              {streamStatus === "CONNECTED" ? (
                <>
                  <WifiOff className="w-3 h-3" />
                  <span>DISCONNECT</span>
                </>
              ) : streamStatus === "RECONNECTING" ? (
                <>
                  <RefreshCw className="w-3 h-3 animate-spin" />
                  <span>CONNECTING</span>
                </>
              ) : (
                <>
                  <Wifi className="w-3 h-3" />
                  <span>CONNECT</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Discrete Hardware Signal / Packet Health Visual */}
      <div className="p-2.5 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle shrink-0 my-2 font-mono">
        <div className="flex items-center justify-between pb-1.5 border-b border-white/5 text-[9px] text-[#565C66] tracking-wider uppercase mb-2">
          <span>HARDWARE RF / CARRIER HEALTH</span>
        </div>

        {/* 6 Discrete Horizontal Indicator Segments */}
        <div className="grid grid-cols-6 gap-1.5 mb-2.5">
          {Array.from({ length: signalQuality.maxBars }).map((_, idx) => {
            const isActive = idx < signalQuality.bars;
            return (
              <div
                key={idx}
                className={`h-4 rounded-[1px] border transition-colors duration-200 bezel-depth-subtle ${
                  isActive
                    ? streamStatus === "CONNECTED"
                      ? "bg-[#00E08A] border-[#00E08A]/80 shadow-[0_0_4px_rgba(0,224,138,0.4)]"
                      : "bg-[#FFB020] border-[#FFB020]/80 shadow-[0_0_4px_rgba(255,176,32,0.4)]"
                    : "bg-[#0E1015] border-white/5"
                }`}
              />
            );
          })}
        </div>

        {/* 3 Metrics: SNR, Packet Loss, Jitter */}
        <div className="grid grid-cols-3 gap-2 text-center text-xs">
          <div className="p-1.5 bg-[#0E1015] border border-white/5 rounded-[2px] bezel-depth-subtle">
            <span className="text-[8.5px] text-[#8A919C] block">SNR MARGIN</span>
            <span
              className={`font-bold ${
                streamStatus === "CONNECTED" ? "text-[#00E08A]" : "text-[#565C66]"
              }`}
            >
              {streamStatus === "CONNECTED" ? `${signalQuality.snrDb} dB` : "0.0 dB"}
            </span>
          </div>

          <div className="p-1.5 bg-[#0E1015] border border-white/5 rounded-[2px] bezel-depth-subtle">
            <span className="text-[8.5px] text-[#8A919C] block">PACKET LOSS</span>
            <span
              className={`font-bold ${
                streamStatus === "CONNECTED" ? "text-[#00E08A]" : "text-[#FF4D4F]"
              }`}
            >
              {streamStatus === "CONNECTED"
                ? `${signalQuality.packetLossPct}%`
                : "100.0%"}
            </span>
          </div>

          <div className="p-1.5 bg-[#0E1015] border border-white/5 rounded-[2px] bezel-depth-subtle">
            <span className="text-[8.5px] text-[#8A919C] block">LINK JITTER</span>
            <span
              className={`font-bold ${
                streamStatus === "CONNECTED" ? "text-[#4DA3FF]" : "text-[#565C66]"
              }`}
            >
              {streamStatus === "CONNECTED" ? `${signalQuality.jitterMs} ms` : "---"}
            </span>
          </div>
        </div>
      </div>

      {/* Live Connection Log-Style Strip (Last 3-4 Events, No Independent Scroll) */}
      <div className="p-2.5 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle shrink-0 mb-2 font-mono">
        <div className="flex items-center justify-between pb-1 border-b border-white/5 text-[8.5px] text-[#565C66] tracking-wider uppercase mb-1.5">
          <span>CARRIER LINK EVENTS (LAST 3 CYCLES)</span>
        </div>

        <div className="space-y-1 text-[10px]">
          {streamEvents.slice(0, 3).map((event) => (
            <div
              key={event.id}
              className="flex items-center gap-2 truncate text-[#8A919C]"
            >
              <span className="text-[#565C66] select-none text-[8.5px]">
                [{event.timestamp}]
              </span>
              <span
                className={`font-bold text-[9px] ${
                  event.type === "CONNECT"
                    ? "text-[#00E08A]"
                    : event.type === "DISCONNECT"
                    ? "text-[#FF4D4F]"
                    : "text-[#4DA3FF]"
                }`}
              >
                [{event.type}]
              </span>
              <span className="text-[#E6E9ED] truncate">{event.message}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Mission Context Framing Note (Why This Tab Exists) */}
      <div className="p-2.5 bg-[#0E1015] border border-white/10 rounded-[2px] bezel-depth-subtle font-mono text-[9.5px] text-[#8A919C] shrink-0">
        <div className="flex items-center gap-1.5 text-[#E6E9ED] font-bold text-[10px] mb-1">
          <ShieldAlert className="w-3.5 h-3.5 text-[#4DA3FF]" />
          <span>AUTONOMY CONTEXT: GROUND COMMUNICATIONS LATENCY</span>
        </div>
        <p className="leading-relaxed text-[#8A919C]">
          Orbital transit delay and communication blackouts prevent remote ground teleoperation.
          The on-board TAR DecisionStabilizer executes 100% locally on edge hardware to verify safety
          invariants regardless of ground link state.
        </p>
      </div>
    </AvionicsPanel>
  );
}
