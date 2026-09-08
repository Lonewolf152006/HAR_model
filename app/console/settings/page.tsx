"use client";

import React from "react";
import { CameraSourcePanel } from "@/components/settings/CameraSourcePanel";
import { GateThresholdTuningPanel } from "@/components/settings/GateThresholdTuningPanel";
import { SystemTogglesPanel } from "@/components/settings/SystemTogglesPanel";
import { ModelSpecSheet } from "@/components/settings/ModelSpecSheet";

export default function SettingsPage() {
  return (
    <div className="h-full w-full max-h-full overflow-hidden select-none font-mono relative">
      {/* Fixed Viewport 2-Column Grid (roughly 50/50 split, strictly no outer scrollbar) */}
      <div className="h-full w-full grid grid-cols-1 lg:grid-cols-12 gap-3 overflow-hidden">
        {/* LEFT COLUMN: Camera Source (top) + Gate Threshold Tuning (dominant, bottom) */}
        <div className="lg:col-span-6 h-full min-h-0 overflow-hidden flex flex-col gap-3">
          {/* Top: Camera Source Selection Switch */}
          <div className="shrink-0">
            <CameraSourcePanel />
          </div>

          {/* Bottom: Signature Gate Threshold Tuning Sliders */}
          <div className="flex-1 min-h-0 overflow-hidden">
            <GateThresholdTuningPanel />
          </div>
        </div>

        {/* RIGHT COLUMN: System Toggles (top) + Model Spec Sheet (bottom) */}
        <div className="lg:col-span-6 h-full min-h-0 overflow-hidden flex flex-col gap-3">
          {/* Top: Sound & Animation Toggles + Boot Replay */}
          <div className="shrink-0">
            <SystemTogglesPanel />
          </div>

          {/* Bottom: Read-Only Model Architecture Spec Sheet */}
          <div className="flex-1 min-h-0 overflow-hidden">
            <ModelSpecSheet />
          </div>
        </div>
      </div>
    </div>
  );
}
