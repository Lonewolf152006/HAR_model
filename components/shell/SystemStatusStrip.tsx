"use client";

import React, { useState } from "react";
import { SubsystemStatusMap, SubsystemHealth } from "@/lib/types";

interface SystemStatusStripProps {
  subsystems: SubsystemStatusMap;
}

interface SubsystemMeta {
  key: keyof SubsystemStatusMap;
  label: string;
  nominalDesc: string;
  degradedDesc: string;
  faultDesc: string;
}

const SUBSYSTEMS_META: SubsystemMeta[] = [
  {
    key: "camera",
    label: "CAMERA",
    nominalDesc: "Sony IMX477 // 1080p60 // Exposure: 1/120s // FPS: 59.9",
    degradedDesc: "Dropped frames detected // FPS: 28.4",
    faultDesc: "Video link lost // MIPI CSI-2 bus offline",
  },
  {
    key: "tar_model",
    label: "TAR MODEL",
    nominalDesc: "BiLSTM+Attention // Latency: 11.2ms // Weight checksum: OK",
    degradedDesc: "Inference latency spike > 45ms",
    faultDesc: "Model execution fault // Tensor memory error",
  },
  {
    key: "yolo",
    label: "YOLO",
    nominalDesc: "YOLOv8-Nano TensorRT // 3 classes tracked // IoU: 0.88",
    degradedDesc: "Bounding box jitter / Low IoU match",
    faultDesc: "Detector process crashed",
  },
  {
    key: "stream",
    label: "STREAM",
    nominalDesc: "WebRTC RTSP Sink // 192.168.1.140:8554 // Bitrate: 4.2 Mbps",
    degradedDesc: "Bandwidth throttling active // Buffer: 85%",
    faultDesc: "Sink unreachable // Connection timed out",
  },
];

export function SystemStatusStrip({ subsystems }: SystemStatusStripProps) {
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);

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

  return (
    <div className="flex items-center gap-1.5 bg-[#12151A] bezel-depth-subtle border border-white/10 p-1 rounded-[2px]">
      {SUBSYSTEMS_META.map((sub) => {
        const health = subsystems[sub.key] || "nominal";
        const styles = getStatusStyle(health);
        const isHovered = hoveredKey === sub.key;

        const description =
          health === "nominal"
            ? sub.nominalDesc
            : health === "degraded"
            ? sub.degradedDesc
            : sub.faultDesc;

        return (
          <div
            key={sub.key}
            className="relative"
            onMouseEnter={() => setHoveredKey(sub.key)}
            onMouseLeave={() => setHoveredKey(null)}
          >
            <div
              className={`flex items-center gap-1.5 px-2 py-1 rounded-[2px] transition-colors cursor-help border ${styles.border} bg-[#171B21] hover:bg-[#1E232B]`}
            >
              {/* Status Dot with tight box-shadow glow */}
              <span
                className={`w-1.5 h-1.5 rounded-full ${styles.dotBg} ${styles.glow} ${
                  health === "nominal" ? "animate-status-pulse" : "animate-ping"
                }`}
              />
              <span className="font-mono text-[11px] font-medium tracking-wider text-[#8A919C]">
                {sub.label}
              </span>
            </div>

            {/* Flight Telemetry Tooltip */}
            {isHovered && (
              <div className="absolute top-full mt-1.5 left-0 z-50 w-64 p-2 bg-[#171B21] bezel-depth border border-white/20 shadow-xl rounded-[2px] pointer-events-none">
                <div className="flex items-center justify-between pb-1 mb-1 border-b border-white/10 font-mono text-[10px]">
                  <span className="font-bold text-[#E6E9ED]">{sub.label} SUBSYSTEM</span>
                  <span
                    className={`font-semibold ${
                      health === "nominal"
                        ? "text-[#00E08A]"
                        : health === "degraded"
                        ? "text-[#FFB020]"
                        : "text-[#FF4D4F]"
                    }`}
                  >
                    [{styles.statusTag}]
                  </span>
                </div>
                <p className="font-mono text-[11px] text-[#8A919C] leading-snug">
                  {description}
                </p>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
