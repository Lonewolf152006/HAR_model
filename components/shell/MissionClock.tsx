"use client";

import React, { useState, useEffect } from "react";

export function MissionClock() {
  const [secondsElapsed, setSecondsElapsed] = useState<number>(142);

  useEffect(() => {
    const timer = setInterval(() => {
      setSecondsElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const hours = Math.floor(secondsElapsed / 3600);
  const minutes = Math.floor((secondsElapsed % 3600) / 60);
  const seconds = secondsElapsed % 60;

  const pad = (n: number) => String(n).padStart(2, "0");
  const timeString = `T+${pad(hours)}:${pad(minutes)}:${pad(seconds)}`;

  return (
    <div className="flex items-center gap-2.5 px-3 py-1 bg-[#12151A] bezel-depth-subtle border border-white/10 rounded-[2px]">
      <div className="flex flex-col">
        <span className="text-[9px] uppercase tracking-widest text-[#8A919C] font-mono leading-none">
          MET // CLOCK
        </span>
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#00E08A] glow-nominal animate-pulse" />
          <span className="font-mono text-sm tracking-wider font-semibold text-[#E6E9ED]">
            {timeString}
          </span>
        </div>
      </div>
    </div>
  );
}
