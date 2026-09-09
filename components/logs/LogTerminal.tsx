"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import { LogEntry, LogLevel } from "@/lib/types";
import { Terminal, ChevronDown, Radio } from "lucide-react";

interface LogTerminalProps {
  logs: LogEntry[];
  currentFilter: LogLevel | "ALL";
}

export function LogTerminal({ logs, currentFilter }: LogTerminalProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isUserScrolledUp, setIsUserScrolledUp] = useState(false);
  const [unseenCount, setUnseenCount] = useState(0);
  const [isLivePulse, setIsLivePulse] = useState(false);
  const pulseTimerRef = useRef<NodeJS.Timeout | null>(null);
  const prevLogsLengthRef = useRef(logs.length);

  // Filter logs client-side
  const filteredLogs = React.useMemo(() => {
    if (currentFilter === "ALL") return logs;
    return logs.filter((log) => log.level === currentFilter);
  }, [logs, currentFilter]);

  // Pulse event for incoming lines
  const triggerLivePulse = useCallback(() => {
    setIsLivePulse(true);
    if (pulseTimerRef.current) clearTimeout(pulseTimerRef.current);
    pulseTimerRef.current = setTimeout(() => {
      setIsLivePulse(false);
    }, 280);
  }, []);

  // Track scroll position
  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight <= 30;

    if (isAtBottom) {
      setIsUserScrolledUp(false);
      setUnseenCount(0);
    } else {
      setIsUserScrolledUp(true);
    }
  };

  // Jump to bottom action
  const scrollToBottom = () => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    setIsUserScrolledUp(false);
    setUnseenCount(0);
  };

  // React to new logs
  useEffect(() => {
    const prevLen = prevLogsLengthRef.current;
    const diff = logs.length - prevLen;
    prevLogsLengthRef.current = logs.length;

    if (diff > 0) {
      triggerLivePulse();

      if (isUserScrolledUp) {
        setUnseenCount((prev) => prev + diff);
      } else {
        if (scrollRef.current) {
          scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
      }
    }
  }, [logs.length, isUserScrolledUp, triggerLivePulse]);

  // When filter changes, scroll to bottom if not manually scrolled up
  useEffect(() => {
    if (!isUserScrolledUp && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [currentFilter, isUserScrolledUp]);

  // Initial scroll to bottom on mount
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, []);

  const getTagColor = (level: LogLevel) => {
    switch (level) {
      case "ACCEPTED":
        return "text-[#00E08A]";
      case "REJECTED":
        return "text-[#FF4D4F]";
      case "ALERT":
        return "text-[#FF4D4F]";
      case "CONTAINMENT":
        return "text-[#4DA3FF]";
      case "INFO":
      default:
        return "text-[#8A919C]";
    }
  };

  const getMessageColor = (level: LogLevel) => {
    switch (level) {
      case "ACCEPTED":
        return "text-[#E6E9ED]";
      case "REJECTED":
        return "text-[#FF4D4F]";
      case "ALERT":
        return "text-[#FF4D4F]";
      case "CONTAINMENT":
        return "text-[#8A919C]";
      case "INFO":
      default:
        return "text-[#8A919C]";
    }
  };

  return (
    <div className="relative bg-[#0E1015] border border-white/10 rounded-[2px] bezel-depth h-full w-full flex flex-col justify-between overflow-hidden">
      {/* 4 Precision HUD Corner Bracket Accents */}
      <span className="absolute -top-[1px] -left-[1px] w-2.5 h-2.5 border-t-[1.5px] border-l-[1.5px] border-[#00E08A]/60 pointer-events-none" />
      <span className="absolute -top-[1px] -right-[1px] w-2.5 h-2.5 border-t-[1.5px] border-r-[1.5px] border-[#00E08A]/60 pointer-events-none" />
      <span className="absolute -bottom-[1px] -left-[1px] w-2.5 h-2.5 border-b-[1.5px] border-l-[1.5px] border-[#00E08A]/60 pointer-events-none" />
      <span className="absolute -bottom-[1px] -right-[1px] w-2.5 h-2.5 border-b-[1.5px] border-r-[1.5px] border-[#00E08A]/60 pointer-events-none" />

      {/* Terminal Header Bar */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/10 bg-[#12151A] shrink-0 select-none">
        <div className="flex items-center gap-2.5">
          <Terminal className="w-3.5 h-3.5 text-[#00E08A]" />
          <h2 className="font-mono text-[11px] font-bold text-[#E6E9ED] tracking-wider uppercase">
            BLACK-BOX FLIGHT RECORDER — LIVE TAIL
          </h2>
          <span className="text-[#565C66] font-mono text-[10px]">|</span>
          <span className="font-mono text-[10px] text-[#8A919C]">
            [BUFFER: {filteredLogs.length}/{logs.length} LINES]
          </span>
        </div>

        {/* Live Indicator — Subtle pulse tied to actual incoming-line events */}
        <div className="flex items-center gap-2 font-mono text-[10px]">
          <span
            className={`w-2 h-2 rounded-[1px] transition-all duration-150 ${
              isLivePulse
                ? "bg-[#00E08A] glow-nominal scale-125"
                : "bg-[#00E08A]/70"
            }`}
          />
          <span
            className={`font-bold tracking-wider ${
              isLivePulse ? "text-[#00E08A]" : "text-[#00E08A]/80"
            }`}
          >
            LIVE
          </span>
          <span className="text-[#343A46]">•</span>
          <span className="text-[9px] text-[#565C66]">TAILING 10Hz</span>
        </div>
      </div>

      {/* Main Terminal Output Area (Only internal overflow-y-auto on page) */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto px-3 py-2.5 font-mono text-[11.5px] leading-relaxed select-text space-y-0.5"
      >
        {filteredLogs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-[#565C66] font-mono text-xs">
            <Radio className="w-6 h-6 mb-2 text-[#343A46]" />
            <span>NO LOG ENTRIES MATCHING FILTER: {currentFilter}</span>
          </div>
        ) : (
          filteredLogs.map((log, index) => {
            const lineNum = String(index + 1).padStart(3, "0");
            const tagColor = getTagColor(log.level);
            const msgColor = getMessageColor(log.level);

            return (
              <div
                key={log.id}
                className="flex items-baseline gap-2 py-0.5 px-1 hover:bg-white/[0.02] rounded-[1px] transition-none group"
              >
                {/* Gutter Line Number */}
                <span className="w-7 text-[10px] text-[#343A46] shrink-0 text-right select-none group-hover:text-[#565C66]">
                  {lineNum}
                </span>

                {/* Log Line Timestamp */}
                <span className="text-[#565C66] shrink-0 select-none">
                  [{log.timestamp}]
                </span>

                {/* Log Tag: [ACCEPTED] / [REJECTED] / [ALERT] / [INFO] */}
                <span className={`font-bold shrink-0 ${tagColor}`}>
                  [{log.level}]
                </span>

                {/* Log Message Content */}
                <span className={`flex-1 break-all ${msgColor}`}>
                  {log.reason}
                </span>

                {/* Confidence metadata tag if available */}
                {log.confidence !== undefined && (
                  <span className="text-[10px] text-[#565C66] shrink-0 select-none hidden md:inline">
                    (p={log.confidence.toFixed(2)})
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Floating "N new lines — jump to live" Affordance */}
      {isUserScrolledUp && unseenCount > 0 && (
        <div className="absolute bottom-7 left-1/2 -translate-x-1/2 z-20 animate-in fade-in slide-in-from-bottom-2 duration-200">
          <button
            onClick={scrollToBottom}
            className="flex items-center gap-1.5 px-3 py-1 bg-[#171B21] border border-[#00E08A] text-[#00E08A] font-mono text-[10px] font-bold rounded-[2px] bezel-depth shadow-2xl hover:bg-[#00E08A]/15 transition-colors"
          >
            <ChevronDown className="w-3 h-3 text-[#00E08A] animate-bounce" />
            <span>
              {unseenCount} NEW {unseenCount === 1 ? "LINE" : "LINES"} — JUMP TO LIVE
            </span>
          </button>
        </div>
      )}

      {/* Terminal Footer Status Bar */}
      <div className="flex items-center justify-between px-3 py-1 border-t border-white/10 bg-[#12151A] font-mono text-[9px] text-[#565C66] shrink-0 select-none">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="text-[#8A919C]">STORAGE:</span>
            <span>CIRCULAR FIFO NVRAM</span>
          </span>
          <span className="text-[#343A46]">•</span>
          <span className="flex items-center gap-1">
            <span className="text-[#8A919C]">CAPACITY:</span>
            <span>500 LINES MAX</span>
          </span>
        </div>

        <div className="flex items-center gap-3 text-[#8A919C]">
          <span className="flex items-center gap-1">
            <span>TAILING STATUS:</span>
            <span
              className={`font-bold ${
                isUserScrolledUp ? "text-[#FFB020]" : "text-[#00E08A]"
              }`}
            >
              {isUserScrolledUp ? "PAUSED (USER SCROLLED)" : "ACTIVE"}
            </span>
          </span>
        </div>
      </div>
    </div>
  );
}
