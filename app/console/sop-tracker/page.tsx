"use client";

import React from "react";
import { FlightPlanStrip } from "@/components/sop-tracker/FlightPlanStrip";
import { FsmGraphPanel } from "@/components/sop-tracker/FsmGraphPanel";
import { StateEngineSnapshot } from "@/components/sop-tracker/StateEngineSnapshot";

export default function SopTrackerPage() {
  return (
    <div className="h-full w-full grid grid-rows-[minmax(0,1.3fr)_minmax(0,1fr)] gap-3 overflow-hidden">
      {/* Row 1: Dominant Flight-Plan Strip (Full Width, Horizontal Rail) */}
      <div className="min-h-0 w-full overflow-hidden">
        <FlightPlanStrip />
      </div>

      {/* Row 2: 2-Column Avionics Telemetry Analysis Row (FSM Topology + State Engine Snapshot) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 min-h-0 overflow-hidden">
        {/* Col 1: Causal FSM Topology Directed Graph + Monospace Boolean Registry */}
        <div className="min-h-0 overflow-hidden">
          <FsmGraphPanel />
        </div>

        {/* Col 2: State Engine Snapshot (Confidence Sparkline + Single-Line 5-Gate Strip) */}
        <div className="min-h-0 overflow-hidden">
          <StateEngineSnapshot />
        </div>
      </div>
    </div>
  );
}
