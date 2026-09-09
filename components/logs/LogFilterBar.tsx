"use client";

import React, { useState, useRef, useEffect } from "react";
import { LogLevel, LogEntry } from "@/lib/types";
import { Download, ChevronDown, Check, FileCode, FileText } from "lucide-react";

interface LogFilterBarProps {
  currentFilter: LogLevel | "ALL";
  onFilterChange: (filter: LogLevel | "ALL") => void;
  counts: {
    ALL: number;
    ACCEPTED: number;
    REJECTED: number;
    ALERT: number;
    CONTAINMENT: number;
    INFO: number;
  };
  logs: LogEntry[];
}

export function LogFilterBar({
  currentFilter,
  onFilterChange,
  counts,
  logs,
}: LogFilterBarProps) {
  const [exportOpen, setExportOpen] = useState(false);
  const [exportedFormat, setExportedFormat] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setExportOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleExport = (format: "json" | "md") => {
    const timestamp = new Date()
      .toISOString()
      .replace(/[:.]/g, "-")
      .slice(0, 19);

    if (format === "json") {
      const dataStr = JSON.stringify(logs, null, 2);
      const blob = new Blob([dataStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `orbital_har_blackbox_${timestamp}.json`;
      link.click();
      URL.revokeObjectURL(url);
    } else {
      const lines = [
        "# ORBITAL-HAR Flight Recorder Black-Box Log Export",
        `Exported At: ${new Date().toISOString()}`,
        `Total Log Lines: ${logs.length}`,
        "",
        "| Timestamp | Level | State | Reason / Message | Confidence |",
        "| :--- | :--- | :--- | :--- | :--- |",
        ...logs.map(
          (l) =>
            `| ${l.timestamp} | ${l.level} | ${l.state} | ${l.reason.replace(
              /\|/g,
              "\\|"
            )} | ${(l.confidence * 100).toFixed(0)}% |`
        ),
      ].join("\n");

      const blob = new Blob([lines], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `orbital_har_blackbox_${timestamp}.md`;
      link.click();
      URL.revokeObjectURL(url);
    }

    setExportedFormat(format.toUpperCase());
    setTimeout(() => {
      setExportedFormat(null);
      setExportOpen(false);
    }, 1200);
  };

  const filterChips: Array<{
    id: LogLevel | "ALL";
    label: string;
    count: number;
    activeColor: string;
    activeBorder: string;
    activeBg: string;
  }> = [
    {
      id: "ALL",
      label: "ALL",
      count: counts.ALL,
      activeColor: "text-[#E6E9ED]",
      activeBorder: "border-white/40",
      activeBg: "bg-white/10",
    },
    {
      id: "ACCEPTED",
      label: "ACCEPTED",
      count: counts.ACCEPTED,
      activeColor: "text-[#00E08A]",
      activeBorder: "border-[#00E08A]/60",
      activeBg: "bg-[#00E08A]/15",
    },
    {
      id: "REJECTED",
      label: "REJECTED",
      count: counts.REJECTED,
      activeColor: "text-[#FF4D4F]",
      activeBorder: "border-[#FF4D4F]/60",
      activeBg: "bg-[#FF4D4F]/15",
    },
    {
      id: "ALERT",
      label: "ALERT",
      count: counts.ALERT,
      activeColor: "text-[#FF4D4F]",
      activeBorder: "border-[#FF4D4F]/60",
      activeBg: "bg-[#FF4D4F]/15",
    },
    {
      id: "CONTAINMENT",
      label: "CONTAINMENT",
      count: counts.CONTAINMENT,
      activeColor: "text-[#4DA3FF]",
      activeBorder: "border-[#4DA3FF]/60",
      activeBg: "bg-[#4DA3FF]/15",
    },
    {
      id: "INFO",
      label: "INFO",
      count: counts.INFO,
      activeColor: "text-[#8A919C]",
      activeBorder: "border-white/30",
      activeBg: "bg-white/10",
    },
  ];

  return (
    <div className="flex items-center justify-between px-3 py-2 bg-[#12151A] border border-white/10 rounded-[2px] bezel-depth shrink-0 h-12 select-none">
      {/* Left Side: Filter Chips */}
      <div className="flex items-center gap-2 overflow-x-auto pr-2">
        <span className="font-mono text-[9px] text-[#565C66] tracking-wider uppercase mr-1 hidden sm:inline">
          FILTER:
        </span>
        {filterChips.map((chip) => {
          const isActive = currentFilter === chip.id;
          return (
            <button
              key={chip.id}
              onClick={() => onFilterChange(chip.id)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-[2px] font-mono text-[11px] border transition-colors duration-150 bezel-depth-subtle ${
                isActive
                  ? `${chip.activeBg} ${chip.activeBorder} ${chip.activeColor} font-bold`
                  : "bg-[#171B21] border-white/10 text-[#8A919C] hover:border-white/20 hover:text-[#E6E9ED]"
              }`}
            >
              <span>{chip.label}</span>
              <span
                className={`text-[9px] px-1 py-0.2 rounded-[1px] ${
                  isActive
                    ? "bg-black/40 text-current"
                    : "bg-white/5 text-[#565C66]"
                }`}
              >
                {chip.count}
              </span>
            </button>
          );
        })}
      </div>

      {/* Right Side: Export Black Box Data Button + Dropdown */}
      <div className="relative shrink-0" ref={dropdownRef}>
        <button
          onClick={() => setExportOpen((prev) => !prev)}
          className="flex items-center gap-2 px-3 py-1.5 bg-[#171B21] border border-white/15 rounded-[2px] text-[#E6E9ED] font-mono text-[11px] font-bold tracking-wider hover:border-[#00E08A]/60 hover:text-[#00E08A] transition-colors duration-150 bezel-depth-subtle"
        >
          <Download className="w-3.5 h-3.5 text-[#00E08A]" />
          <span>EXPORT BLACK BOX DATA</span>
          <ChevronDown
            className={`w-3 h-3 text-[#8A919C] transition-transform duration-150 ${
              exportOpen ? "rotate-180" : ""
            }`}
          />
        </button>

        {/* Tiny Dropdown / Segmented Control */}
        {exportOpen && (
          <div className="absolute right-0 top-full mt-1 w-44 bg-[#12151A] border border-white/15 rounded-[2px] bezel-depth shadow-2xl py-1 z-50 animate-in fade-in slide-in-from-top-1 duration-150">
            <div className="px-2.5 py-1 border-b border-white/5 font-mono text-[9px] text-[#565C66] tracking-wider uppercase">
              SELECT FORMAT
            </div>

            <button
              onClick={() => handleExport("json")}
              className="w-full flex items-center justify-between px-2.5 py-1.5 font-mono text-xs text-[#E6E9ED] hover:bg-[#171B21] hover:text-[#00E08A] transition-colors text-left"
            >
              <div className="flex items-center gap-2">
                <FileCode className="w-3.5 h-3.5 text-[#4DA3FF]" />
                <span>JSON Flight Buffer</span>
              </div>
              {exportedFormat === "JSON" && (
                <Check className="w-3 h-3 text-[#00E08A]" />
              )}
            </button>

            <button
              onClick={() => handleExport("md")}
              className="w-full flex items-center justify-between px-2.5 py-1.5 font-mono text-xs text-[#E6E9ED] hover:bg-[#171B21] hover:text-[#00E08A] transition-colors text-left"
            >
              <div className="flex items-center gap-2">
                <FileText className="w-3.5 h-3.5 text-[#FFB020]" />
                <span>Markdown Report</span>
              </div>
              {exportedFormat === "MD" && (
                <Check className="w-3 h-3 text-[#00E08A]" />
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
