"use client";

import React, { useState, useEffect } from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import type { CameraSourceId } from "@/lib/types";
import { Camera, Check, RefreshCw } from "lucide-react";

interface DeviceItem {
  id: string;
  index: number;
  name: string;
  type: string;
  status: string;
  resolution?: string;
}

export function CameraSourcePanel() {
  const { setCameraSource } = useTelemetry();
  const [devices, setDevices] = useState<DeviceItem[]>([
    { id: "0", index: 0, name: "Camo Studio", type: "hardware", status: "connected", resolution: "1280x720" },
    { id: "1", index: 1, name: "NVIDIA Broadcast", type: "hardware", status: "connected", resolution: "640x480" },
    { id: "2", index: 2, name: "OBS Virtual Camera", type: "hardware", status: "connected", resolution: "640x480" },
    { id: "file", index: -1, name: "Synthetic / Test Loop", type: "file", status: "ready", resolution: "1280x720" },
  ]);
  const [selectedId, setSelectedId] = useState<string>("0");
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const fetchDevices = () => {
    setIsLoading(true);
    fetch("http://127.0.0.1:8080/api/v1/camera/devices")
      .then((res) => res.json())
      .then((data) => {
        if (data.devices && Array.isArray(data.devices)) {
          setDevices(data.devices);
          if (data.active_source !== undefined) {
            setSelectedId(String(data.active_source));
          }
        }
      })
      .catch(() => {})
      .finally(() => {
        setIsLoading(false);
      });
  };

  useEffect(() => {
    let active = true;
    fetch("http://127.0.0.1:8080/api/v1/camera/devices")
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        if (data.devices && Array.isArray(data.devices)) {
          setDevices(data.devices);
          if (data.active_source !== undefined) {
            setSelectedId(String(data.active_source));
          }
        }
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, []);

  const handleSelect = (dev: DeviceItem) => {
    setSelectedId(dev.id);
    setCameraSource(dev.id as CameraSourceId);
    fetch("http://127.0.0.1:8080/api/v1/camera/select", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: dev.id }),
    }).catch(() => {});
  };

  const activeDevice = devices.find((d) => d.id === selectedId) || devices[0];

  return (
    <AvionicsPanel
      title="OPTICAL SENSOR CAMERA SOURCE"
      badge={
        <button
          onClick={fetchDevices}
          className="px-2 py-0.5 font-mono text-[9.5px] font-bold rounded-[2px] border bg-[#00E08A]/10 text-[#00E08A] border-[#00E08A]/50 glow-nominal flex items-center gap-1.5 hover:bg-[#00E08A]/20 transition-colors"
          title="Refresh connected camera list"
        >
          <RefreshCw className={`w-2.5 h-2.5 ${isLoading ? "animate-spin" : ""}`} />
          <span>CONNECTED ({devices.length} SOURCES)</span>
        </button>
      }
      className="flex flex-col justify-between p-3 font-mono"
    >
      <div className="space-y-2.5">
        <div className="flex items-center justify-between text-[9px] text-[#565C66] tracking-wider uppercase pb-1 border-b border-white/5">
          <span>ALL DETECTED VIDEO DEVICES</span>
          <span>SELECT ACTIVE INPUT</span>
        </div>

        {/* Dynamic Multi-Camera Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 p-1 bg-[#0E1015] border border-white/10 rounded-[2px] bezel-depth-subtle">
          {devices.map((dev) => {
            const isSelected = dev.id === selectedId;
            return (
              <button
                key={dev.id}
                type="button"
                onClick={() => handleSelect(dev)}
                className={`flex items-center justify-between px-2.5 py-2 rounded-[2px] text-[10px] font-bold tracking-tight transition-all duration-150 border ${
                  isSelected
                    ? "bg-[#171B21] border-[#00E08A] text-[#00E08A] glow-nominal"
                    : "border-transparent text-[#8A919C] hover:text-[#E6E9ED] hover:border-white/10"
                }`}
              >
                <div className="flex items-center gap-2 truncate">
                  <Camera className={`w-3.5 h-3.5 shrink-0 ${isSelected ? "text-[#00E08A]" : "text-[#565C66]"}`} />
                  <div className="text-left truncate">
                    <div className="truncate font-mono">{dev.name}</div>
                    <div className="text-[8.5px] text-[#565C66] font-normal">
                      {dev.type === "file" ? "SIMULATION LOOP" : `DEVICE #${dev.index} • ${dev.resolution || "ACTIVE"}`}
                    </div>
                  </div>
                </div>
                {isSelected && <Check className="w-3.5 h-3.5 text-[#00E08A] shrink-0 ml-1" />}
              </button>
            );
          })}
        </div>

        {/* Selected Sensor Interface Specification Readout */}
        <div className="pt-2 border-t border-white/5 space-y-1 text-[10px]">
          <div className="flex justify-between items-center text-[#8A919C]">
            <span>ACTIVE CAPTURE DEVICE:</span>
            <span className="text-[#00E08A] font-bold">{activeDevice?.name}</span>
          </div>
          <div className="flex justify-between items-center text-[#8A919C]">
            <span>INTERFACE / BUS:</span>
            <span className="text-[#E6E9ED]">
              {activeDevice?.type === "file" ? "INTERNAL SYNTHETIC GENERATOR" : "DIRECTSHOW / UVC STREAM"}
            </span>
          </div>
          <div className="flex justify-between items-center text-[#8A919C]">
            <span>STREAM RESOLUTION:</span>
            <span className="text-[#4DA3FF] font-bold">{activeDevice?.resolution || "1280x720"}</span>
          </div>
        </div>
      </div>
    </AvionicsPanel>
  );
}
