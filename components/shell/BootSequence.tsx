"use client";

import React, { useState, useEffect } from "react";
import { BOOT_SEQUENCE_LINES } from "@/lib/constants";

interface BootSequenceProps {
  onComplete?: () => void;
  forceReplay?: boolean;
}

export function BootSequence({ onComplete, forceReplay = false }: BootSequenceProps) {
  const [visibleCount, setVisibleCount] = useState<number>(0);
  const [isFlashing, setIsFlashing] = useState<boolean>(false);
  const [isDismissed, setIsDismissed] = useState<boolean>(false);

  useEffect(() => {
    // Check sessionStorage unless forceReplay is specified
    if (!forceReplay) {
      const booted = sessionStorage.getItem("orbital_har_booted");
      if (booted === "true") {
        setIsDismissed(true);
        if (onComplete) onComplete();
        return;
      }
    }

    // Step through boot lines
    let currentIndex = 0;
    const interval = setInterval(() => {
      currentIndex += 1;
      setVisibleCount(currentIndex);

      if (currentIndex >= BOOT_SEQUENCE_LINES.length) {
        clearInterval(interval);
        // Trigger CRT flash after final line
        setTimeout(() => {
          setIsFlashing(true);
          setTimeout(() => {
            sessionStorage.setItem("orbital_har_booted", "true");
            setIsDismissed(true);
            if (onComplete) onComplete();
          }, 300);
        }, 400);
      }
    }, 180);

    return () => clearInterval(interval);
  }, [forceReplay, onComplete]);

  const handleSkip = () => {
    sessionStorage.setItem("orbital_har_booted", "true");
    setIsDismissed(true);
    if (onComplete) onComplete();
  };

  if (isDismissed) return null;

  return (
    <div
      className={`fixed inset-0 z-50 flex flex-col justify-between bg-[#0B0D10] p-8 md:p-12 font-mono transition-opacity duration-300 ${
        isFlashing ? "animate-crt-flash" : "opacity-100"
      }`}
    >
      <div className="scanline-overlay absolute inset-0 pointer-events-none opacity-40" />

      {/* Top Header */}
      <div className="flex items-center justify-between border-b border-white/10 pb-4 relative z-10">
        <div className="flex items-center gap-3">
          <span className="w-2.5 h-2.5 bg-[#00E08A] animate-ping" />
          <span className="text-xs uppercase tracking-widest text-[#E6E9ED] font-bold">
            ORBITAL-HAR — AVIONICS SYSTEM INITIALIZATION
          </span>
        </div>
        <button
          onClick={handleSkip}
          className="btn-bracket text-xs text-[#8A919C] hover:text-[#00E08A] transition-colors"
        >
          SKIP BOOT
        </button>
      </div>

      {/* Terminal Lines Container */}
      <div className="flex-1 my-8 space-y-2.5 overflow-y-auto max-w-3xl relative z-10">
        <div className="text-[11px] text-[#565C66] mb-4">
          BIO-ASTRONAUTICS RESEARCH PAYLOAD — FLIGHT FIRMWARE REV 4.2.1
          <br />
          MEMORY: 16384 MB ALLOCATED | COLD START VECTOR 0x8004
        </div>

        {BOOT_SEQUENCE_LINES.slice(0, visibleCount).map((line, idx) => {
          const isReady = line.tag === "[READY]";
          const isCalibrating = line.tag === "[CALIBRATING]";

          const tagColor = isReady
            ? "text-[#00E08A]"
            : isCalibrating
            ? "text-[#FFB020]"
            : "text-[#00E08A]";

          return (
            <div
              key={idx}
              className="flex items-start gap-3 text-xs tracking-wide transition-all duration-150 animate-fadeIn"
            >
              <span className={`font-bold shrink-0 ${tagColor}`}>{line.tag}</span>
              <span
                className={
                  isReady
                    ? "text-[#00E08A] font-semibold"
                    : "text-[#E6E9ED]"
                }
              >
                {line.text}
              </span>
            </div>
          );
        })}

        {visibleCount < BOOT_SEQUENCE_LINES.length && (
          <div className="flex items-center gap-2 text-xs text-[#565C66] pt-2">
            <span className="inline-block w-2 h-3.5 bg-[#00E08A] animate-pulse" />
            <span>CALIBRATING HARDWARE GATES...</span>
          </div>
        )}
      </div>

      {/* Bottom Status Bar */}
      <div className="flex items-center justify-between border-t border-white/10 pt-4 text-[10px] text-[#565C66] relative z-10">
        <span>SESSION: STN-ALPHA-EXP04</span>
        <span>INITIALIZING STABILIZER CORES [5 GATES]</span>
      </div>
    </div>
  );
}
