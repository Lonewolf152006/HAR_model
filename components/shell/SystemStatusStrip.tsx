"use client";

import React, { useState, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { useTelemetry } from "@/context/TelemetryContext";
import { SubsystemHealth } from "@/lib/types";

export function SystemStatusStrip() {
  const { isPaused, recordingState, streamStatus, streamConfig } = useTelemetry();
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [pillRect, setPillRect] = useState<DOMRect | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Update rect on scroll or resize if hovered
  const updateRect = useCallback(() => {
    if (hoveredKey) {
      const el = document.getElementById(`subsystem-pill-${hoveredKey}`);
      if (el) {
        setPillRect(el.getBoundingClientRect());
      }
    }
  }, [hoveredKey]);

  useEffect(() => {
    if (!hoveredKey) return;
    window.addEventListener("scroll", updateRect, true);
    window.addEventListener("resize", updateRect);
    return () => {
      window.removeEventListener("scroll", updateRect, true);
      window.removeEventListener("resize", updateRect);
    };
  }, [hoveredKey, updateRect]);

  // Derive real camera health from frame ingestion and recording state
  const cameraHealth: SubsystemHealth = isPaused
    ? "fault"
    : recordingState.isRecording
    ? "nominal"
    : "degraded";

  const cameraDesc = isPaused
    ? "Video ingestion paused | Sensor bus standby"
    : recordingState.isRecording
    ? "Sony IMX477 | 1080p60 | MIPI CSI-2 | Receiving frames (REC active)"
    : "Sony IMX477 | 1080p60 | MIPI CSI-2 | Standby preview";

  // Derive real stream health from active stream connection state
  const streamHealth: SubsystemHealth =
    streamStatus === "CONNECTED"
      ? "nominal"
      : streamStatus === "RECONNECTING"
      ? "degraded"
      : "fault";

  const streamDesc =
    streamStatus === "CONNECTED"
      ? `${streamConfig.protocol} | ${streamConfig.ip}:${streamConfig.port} | 14.8 Mbps | Link nominal`
      : streamStatus === "RECONNECTING"
      ? `Re-negotiating WebRTC SDP with ${streamConfig.ip}:${streamConfig.port}...`
      : "Ground uplink severed | Carrier offline | NVRAM primary";

  const items = [
    {
      key: "camera",
      label: "CAMERA",
      health: cameraHealth,
      desc: cameraDesc,
    },
    {
      key: "stream",
      label: "STREAM",
      health: streamHealth,
      desc: streamDesc,
    },
  ];

  const getStatusStyle = (health: SubsystemHealth) => {
    switch (health) {
      case "nominal":
        return {
          dotBg: "bg-[#00E08A]",
          glow: "glow-nominal",
          text: "text-[#E6E9ED]",
          border: "border-white/10",
          statusTag: "NOMINAL",
        };
      case "degraded":
        return {
          dotBg: "bg-[#FFB020]",
          glow: "glow-caution",
          text: "text-[#FFB020]",
          border: "border-[#FFB020]/40",
          statusTag: "DEGRADED",
        };
      case "fault":
        return {
          dotBg: "bg-[#FF4D4F]",
          glow: "glow-critical",
          text: "text-[#FF4D4F]",
          border: "border-[#FF4D4F]/50",
          statusTag: "FAULT",
        };
    }
  };

  // Position-independent viewport placement calculation
  const getTooltipStyle = (rect: DOMRect) => {
    const tooltipWidth = 260;
    const margin = 12;
    const top = rect.bottom + 6;

    // If aligning to pill's left edge would overflow the right viewport boundary,
    // align the tooltip's right edge to the pill's right edge.
    let left = rect.left;
    if (left + tooltipWidth > window.innerWidth - margin) {
      left = rect.right - tooltipWidth;
    }

    // Strictly clamp within viewport boundaries so clipping is mathematically impossible
    left = Math.max(margin, Math.min(left, window.innerWidth - tooltipWidth - margin));

    return {
      top: `${top}px`,
      left: `${left}px`,
      width: `${tooltipWidth}px`,
    };
  };

  const activeItem = items.find((it) => it.key === hoveredKey);
  const activeStyle = activeItem ? getStatusStyle(activeItem.health) : null;

  return (
    <>
      <div className="flex items-center gap-1.5 bg-[#12151A] bezel-depth-subtle border border-white/10 p-1 rounded-[2px]">
        {items.map((sub) => {
          const styles = getStatusStyle(sub.health);

          return (
            <div
              key={sub.key}
              id={`subsystem-pill-${sub.key}`}
              className="relative"
              onMouseEnter={(e) => {
                setPillRect(e.currentTarget.getBoundingClientRect());
                setHoveredKey(sub.key);
              }}
              onMouseLeave={() => {
                setHoveredKey(null);
                setPillRect(null);
              }}
            >
              <div
                className={`flex items-center gap-1.5 px-2 py-1 rounded-[2px] transition-all duration-150 cursor-help border ${styles.border} bg-[#171B21] hover:bg-[#1E232B] hover:border-white/30`}
              >
                {/* Status Dot with tight box-shadow glow */}
                <span
                  className={`w-1.5 h-1.5 rounded-full ${styles.dotBg} ${styles.glow} ${
                    sub.health === "nominal"
                      ? "animate-status-pulse"
                      : sub.health === "degraded"
                      ? "animate-pulse"
                      : "animate-ping"
                  }`}
                />
                <span className="font-mono text-[11px] font-medium tracking-wider text-[#8A919C]">
                  {sub.label}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Flight Telemetry Tooltip via Portal (Escapes any ancestor overflow, clipping, or z-index constraints) */}
      {mounted && activeItem && activeStyle && pillRect && createPortal(
        <div
          style={{
            position: "fixed",
            ...getTooltipStyle(pillRect),
            zIndex: 9999,
          }}
          className="p-2 bg-[#171B21] bezel-depth border border-white/20 shadow-2xl rounded-[2px] pointer-events-none font-mono select-none animate-in fade-in-50 duration-100"
        >
          <div className="flex items-center justify-between pb-1 mb-1 border-b border-white/10 text-[10px]">
            <span className="font-bold text-[#E6E9ED]">
              {activeItem.label} SUBSYSTEM
            </span>
            <span
              className={`font-semibold ${
                activeItem.health === "nominal"
                  ? "text-[#00E08A]"
                  : activeItem.health === "degraded"
                  ? "text-[#FFB020]"
                  : "text-[#FF4D4F]"
              }`}
            >
              [{activeStyle.statusTag}]
            </span>
          </div>
          <p className="text-[10.5px] text-[#8A919C] leading-snug">
            {activeItem.desc}
          </p>
        </div>,
        document.body
      )}
    </>
  );
}
