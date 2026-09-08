"use client";

import React, { createContext, useContext } from "react";
import { useTelemetryStream } from "@/lib/useTelemetryStream";

type TelemetryContextType = ReturnType<typeof useTelemetryStream>;

const TelemetryContext = createContext<TelemetryContextType | null>(null);

export function TelemetryProvider({ children }: { children: React.ReactNode }) {
  const telemetry = useTelemetryStream();

  return (
    <TelemetryContext.Provider value={telemetry}>
      {children}
    </TelemetryContext.Provider>
  );
}

export function useTelemetry() {
  const context = useContext(TelemetryContext);
  if (!context) {
    throw new Error("useTelemetry must be used within a TelemetryProvider");
  }
  return context;
}
