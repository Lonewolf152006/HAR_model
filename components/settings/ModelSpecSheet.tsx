"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { TELEMETRY_SESSION_ID } from "@/lib/constants";
import { FileCode, Database } from "lucide-react";

export function ModelSpecSheet() {
  const specs = [
    { label: "MODEL ARCHITECTURE", value: "BiLSTM + Multi-Head Self-Attention (2-layer, 128 hidden)" },
    { label: "ACTION CLASSES (7)", value: "idle, open_box, pick_red, place_red_out, pick_blue, place_blue_in, close_box" },
    { label: "OBJECT DETECTOR", value: "YOLOv8-Nano (custom 3-class bounding: box, red, blue)" },
    { label: "POSE ESTIMATION", value: "MediaPipe BlazePose (33 keypoints // 2.5D normalized)" },
    { label: "WEIGHTS FILENAME", value: "best_tar_model.pth (SHA256: 9f8a...c10e)" },
    { label: "TEMPORAL WINDOW", value: "32 frames @ 10Hz sampling (3.2s receptive field)" },
    { label: "EDGE INFERENCE", value: "11.4 ms @ FP16 TensorRT (NVIDIA Jetson Orin NX target)" },
    { label: "MISSION SESSION ID", value: TELEMETRY_SESSION_ID || "ORB-HAR-MISSION-2026-09" },
  ];

  return (
    <AvionicsPanel
      title="NEURAL PIPELINE // MODEL SPEC SHEET"
      indexTag="05 // ARCHITECTURE SPEC"
      className="h-full flex flex-col justify-between p-3 font-mono"
    >
      <div className="flex flex-col h-full justify-between">
        <div className="flex items-center justify-between text-[9px] text-[#565C66] tracking-wider uppercase pb-1 border-b border-white/5">
          <div className="flex items-center gap-1">
            <FileCode className="w-3 h-3 text-[#565C66]" />
            <span>READ-ONLY SPECIFICATION MANIFEST</span>
          </div>
          <span>AUTONOMOUS PIPELINE</span>
        </div>

        {/* Spec Grid / Table */}
        <div className="space-y-1.5 my-auto py-1">
          {specs.map((spec, idx) => (
            <div
              key={idx}
              className="p-1.5 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle flex flex-col sm:flex-row sm:items-baseline justify-between gap-1 text-[9.5px]"
            >
              <span className="text-[#8A919C] text-[8.5px] uppercase shrink-0 font-medium">
                {spec.label}
              </span>
              <span className="text-[#E6E9ED] font-mono truncate text-right">
                {spec.value}
              </span>
            </div>
          ))}
        </div>

        {/* Static Footer Note */}
        <div className="pt-1.5 border-t border-white/5 flex items-center justify-between text-[8px] text-[#565C66]">
          <span className="flex items-center gap-1">
            <Database className="w-2.5 h-2.5" />
            <span>IN-MEMORY CHECKPOINT LOADED</span>
          </span>
          <span>STATIONARY VALIDATED</span>
        </div>
      </div>
    </AvionicsPanel>
  );
}
