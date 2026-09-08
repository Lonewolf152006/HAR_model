"use client";

import React, { useState, useMemo } from "react";
import { useTelemetry } from "@/context/TelemetryContext";
import { LogLevel } from "@/lib/types";
import { LogFilterBar } from "@/components/logs/LogFilterBar";
import { LogTerminal } from "@/components/logs/LogTerminal";

export default function LogsPage() {
  const { logs } = useTelemetry();
  const [currentFilter, setCurrentFilter] = useState<LogLevel | "ALL">("ALL");

  const counts = useMemo(() => {
    return {
      ALL: logs.length,
      ACCEPTED: logs.filter((l) => l.level === "ACCEPTED").length,
      REJECTED: logs.filter((l) => l.level === "REJECTED").length,
      ALERT: logs.filter((l) => l.level === "ALERT").length,
      INFO: logs.filter((l) => l.level === "INFO").length,
    };
  }, [logs]);

  return (
    <div className="h-full w-full flex flex-col gap-2.5 overflow-hidden">
      {/* Row 1: Filter Chips + Export Button (Single Row, ~48px) */}
      <LogFilterBar
        currentFilter={currentFilter}
        onFilterChange={setCurrentFilter}
        counts={counts}
        logs={logs}
      />

      {/* Row 2: Dominant Full-Width Log Terminal (~90% Height) */}
      <div className="flex-1 min-h-0 w-full overflow-hidden">
        <LogTerminal logs={logs} currentFilter={currentFilter} />
      </div>
    </div>
  );
}
