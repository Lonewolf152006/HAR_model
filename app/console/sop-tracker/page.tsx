"use client";

import React from "react";
import { FlightPlanStrip } from "@/components/sop-tracker/FlightPlanStrip";
import { FsmGraphPanel } from "@/components/sop-tracker/FsmGraphPanel";
import { StateEngineSnapshot } from "@/components/sop-tracker/StateEngineSnapshot";
import { ContainmentQuickView } from "@/components/sop-tracker/ContainmentQuickView";

export default function SopTrackerPage() {
  return (
    <div className="h-full w-full grid grid-rows-[minmax(0,1.3fr)_minmax(0,1fr)] gap-3 overflow-hidden">
      {/* Row 1: Dominant Flight-Plan Strip (Full Width, Horizontal Rail) */}
      <div className="min-h-0 w-full overflow-hidden">
        <FlightPlanStrip />
      </div>

      {/* Row 2: 3-Column Avionics Telemetry Analysis Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 min-h-0 overflow-hidden">
        {/* Col 1: Causal FSM Topology Directed Graph + Monospace Boolean Registry */}
        <div className="min-h-0 overflow-hidden">
          <FsmGraphPanel />
        </div>

        {/* Col 2: State Engine Snapshot (Confidence Sparkline + Single-Line 5-Gate Strip) */}
        <div className="min-h-0 overflow-hidden">
          <StateEngineSnapshot />
        </div>

        {/* Col 3: Containment Quick-View (3-Row Reference for Box, Red, Blue) */}
        <div className="min-h-0 overflow-hidden">
          <ContainmentQuickView />
        </div>
      </div>
    </div>
  );
}
