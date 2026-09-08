"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { Disc, Square, HardDrive, AlertTriangle, Clock, Folder } from "lucide-react";

export function LocalRecordingPanel() {
  const { recordingState, streamStatus, toggleRecording } = useTelemetry();
  const {
    isRecording,
    elapsedMs,
    filePath,
    fileSizeBytes,
    remainingDiskBytes,
    totalDiskBytes,
  } = recordingState;

  // Format elapsed time as HH:MM:SS.t
  const formatTimer = (ms: number) => {
    const totalSecs = Math.floor(ms / 1000);
    const hrs = Math.floor(totalSecs / 3600);
    const mins = Math.floor((totalSecs % 3600) / 60);
    const secs = totalSecs % 60;
    const tenths = Math.floor((ms % 1000) / 100);
    return `${String(hrs).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${tenths}`;
  };

  const formatGb = (bytes: number) => {
    return (bytes / (1024 * 1024 * 1024)).toFixed(2);
  };

  const usedDiskBytes = Math.max(0, totalDiskBytes - remainingDiskBytes);
  const usedDiskGb = formatGb(usedDiskBytes);
  const remainingDiskGb = formatGb(remainingDiskBytes);
  const totalDiskGb = (totalDiskBytes / (1024 * 1024 * 1024)).toFixed(0);
  const currentFileSizeGb = formatGb(fileSizeBytes);
  const diskUsagePct = Math.min(100, Math.round((usedDiskBytes / totalDiskBytes) * 100));

  // Estimate remaining recording time at ~280 KB/s encoding bitrate
  const estRemainingHours = (remainingDiskBytes / (280000 * 3600)).toFixed(1);

  const isStreamDown = streamStatus !== "CONNECTED";

  const getRecBadge = () => {
    if (isRecording) {
      return (
        <span className="px-2 py-0.5 font-mono text-[9.5px] font-bold rounded-[2px] border bg-[#FF4D4F]/10 text-[#FF4D4F] border-[#FF4D4F]/50 glow-critical flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[#FF4D4F] animate-pulse" />
          REC ACTIVE
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 font-mono text-[9.5px] font-bold rounded-[2px] border bg-[#565C66]/20 text-[#8A919C] border-white/10 flex items-center gap-1.5">
        <span className="w-1.5 h-1.5 rounded-full bg-[#565C66]" />
        STANDBY
      </span>
    );
  };

  return (
    <AvionicsPanel
      title="LOCAL EDGE RECORDER"
      badge={getRecBadge()}
      className={`h-full flex flex-col justify-between overflow-hidden p-3 transition-colors duration-200 ${
        isStreamDown
          ? "border-[#FFB020]/70 glow-caution"
          : "border-white/10"
      }`}
    >
      {/* Top Banner: Cross-panel link notice when ground carrier is down */}
      {isStreamDown && (
        <div className="flex items-center gap-2 px-2.5 py-1.5 mb-2 bg-[#FFB020]/10 border border-[#FFB020]/40 rounded-[2px] text-[9.5px] font-mono text-[#FFB020] shrink-0">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0 animate-pulse text-[#FFB020]" />
          <span className="leading-tight">
            GROUND LINK {streamStatus} — ON-BOARD NVRAM IS PRIMARY MISSION RECORD
          </span>
        </div>
      )}

      {/* Primary Readout: Large Elapsed Timer & Control */}
      <div className="p-3 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle shrink-0">
        <div className="flex items-center justify-between pb-1.5 border-b border-white/5 text-[9px] font-mono text-[#565C66] tracking-wider uppercase mb-2">
          <div className="flex items-center gap-1">
            <Clock className="w-3 h-3 text-[#8A919C]" />
            <span>SESSION RUNTIME</span>
          </div>
          <span className="text-[#8A919C]">H.265 / HEVC CBR</span>
        </div>

        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="font-mono text-2xl sm:text-3xl font-bold tracking-tight text-[#E6E9ED]">
              {formatTimer(elapsedMs)}
            </div>
            <div className="text-[9px] font-mono text-[#8A919C] mt-0.5">
              CODEC: 1080p60 | ON-BOARD DISK CACHE
            </div>
          </div>

          <button
            type="button"
            onClick={toggleRecording}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-[2px] font-mono text-xs font-bold border transition-colors bezel-depth-subtle ${
              isRecording
                ? "bg-[#1A0E10] hover:bg-[#2A1417] text-[#FF4D4F] border-[#FF4D4F]/60 hover:border-[#FF4D4F]"
                : "bg-[#0E1A14] hover:bg-[#14261D] text-[#00E08A] border-[#00E08A]/60 hover:border-[#00E08A]"
            }`}
          >
            {isRecording ? (
              <>
                <Square className="w-3.5 h-3.5 fill-[#FF4D4F]" />
                <span>STOP REC</span>
              </>
            ) : (
              <>
                <Disc className="w-3.5 h-3.5 text-[#00E08A] animate-pulse" />
                <span>START REC</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Storage Information & Remaining Estimates */}
      <div className="p-2.5 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle shrink-0 my-2 font-mono space-y-2">
        <div className="flex items-center justify-between pb-1 border-b border-white/5 text-[9px] text-[#565C66] tracking-wider uppercase">
          <div className="flex items-center gap-1">
            <HardDrive className="w-3 h-3 text-[#8A919C]" />
            <span>NVRAM SOLID STATE STORAGE</span>
          </div>
          <span>SLOT 01 : PARTITION B</span>
        </div>

        {/* File Path & Current Size */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-1.5 text-[#8A919C] truncate max-w-[70%]">
            <Folder className="w-3 h-3 shrink-0 text-[#565C66]" />
            <span className="truncate text-[10px] text-[#E6E9ED]">{filePath}</span>
          </div>
          <div className="text-right">
            <span className="text-[8.5px] text-[#8A919C] block">FILE SIZE</span>
            <span className="text-xs font-bold text-[#00E08A]">{currentFileSizeGb} GB</span>
          </div>
        </div>

        {/* Storage Bar */}
        <div>
          <div className="flex items-center justify-between text-[9px] text-[#8A919C] mb-1">
            <span>USED: {usedDiskGb} GB / {totalDiskGb} GB</span>
            <span className="font-bold text-[#E6E9ED]">{diskUsagePct}%</span>
          </div>
          <div className="w-full h-2 bg-[#0E1015] border border-white/10 rounded-[1px] overflow-hidden p-0.5 bezel-depth-subtle">
            <div
              className={`h-full transition-all duration-300 rounded-[1px] ${
                diskUsagePct > 85 ? "bg-[#FF4D4F]" : diskUsagePct > 70 ? "bg-[#FFB020]" : "bg-[#4DA3FF]"
              }`}
              style={{ width: `${diskUsagePct}%` }}
            />
          </div>
        </div>

        {/* Remaining Storage Estimate */}
        <div className="p-1.5 bg-[#0E1015] border border-white/5 rounded-[2px] flex items-center justify-between text-xs">
          <div>
            <span className="text-[8.5px] text-[#8A919C] block">FREE CAPACITY</span>
            <span className="font-bold text-[#E6E9ED]">{remainingDiskGb} GB</span>
          </div>
          <div className="text-right">
            <span className="text-[8.5px] text-[#8A919C] block">ESTIMATED RUNTIME</span>
            <span className="font-bold text-[#4DA3FF]">~{estRemainingHours} HRS REMAINING</span>
          </div>
        </div>
      </div>
    </AvionicsPanel>
  );
}
