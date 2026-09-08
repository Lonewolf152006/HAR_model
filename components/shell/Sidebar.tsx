"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Video,
  ListOrdered,
  Terminal,
  Radio,
  Sliders,
  ShieldCheck,
} from "lucide-react";
import { APP_ROUTES } from "@/lib/constants";
import { useTelemetry } from "@/context/TelemetryContext";

const ICON_MAP = {
  "live-feed": Video,
  "sop-tracker": ListOrdered,
  logs: Terminal,
  stream: Radio,
  settings: Sliders,
};

export function Sidebar() {
  const pathname = usePathname();
  const { currentState, confidence, cyclesCompleted } = useTelemetry();

  return (
    <aside className="w-64 border-r border-white/10 bg-[#0B0D10] flex flex-col justify-between select-none shrink-0">
      {/* Brand / Codename Header */}
      <div>
        <div className="h-14 px-4 flex items-center gap-3 border-b border-white/10 bg-[#12151A] bezel-depth-subtle">
          <div className="w-6 h-6 rounded-[2px] bg-[#171B21] border border-[#00E08A]/50 flex items-center justify-center">
            <ShieldCheck className="w-4 h-4 text-[#00E08A]" />
          </div>
          <div className="flex flex-col">
            <span className="font-mono text-xs font-bold tracking-wider text-[#E6E9ED]">
              ORBITAL-HAR
            </span>
            <span className="font-mono text-[9px] tracking-widest text-[#565C66]">
              AVIONICS CONSOLE v4.2
            </span>
          </div>
        </div>

        {/* Navigation Breaker Switchboard */}
        <nav className="p-2 space-y-1">
          <div className="px-2 py-1.5 text-[9px] font-mono uppercase tracking-widest text-[#565C66]">
            NAVIGATION // CONSOLE BUS
          </div>

          {APP_ROUTES.map((route) => {
            const isActive = pathname.startsWith(route.path);
            const Icon = ICON_MAP[route.id as keyof typeof ICON_MAP] || Video;

            return (
              <Link
                key={route.id}
                href={route.path}
                className={`group relative flex items-center justify-between px-3 py-2.5 transition-all text-xs font-mono rounded-[2px] border ${isActive
                    ? "bg-[#171B21] bezel-depth-subtle border-[#00E08A]/50 text-[#E6E9ED] font-semibold"
                    : "bg-[#12151A] border-white/5 text-[#8A919C] hover:text-[#E6E9ED] hover:bg-[#171B21]"
                  }`}
              >
                {/* Breaker switch active bar indicator with tight glow */}
                {isActive && (
                  <span className="absolute left-0 top-0 bottom-0 w-[3px] bg-[#00E08A] glow-nominal" />
                )}

                <div className="flex items-center gap-2.5">
                  <Icon
                    className={`w-4 h-4 transition-colors ${isActive ? "text-[#00E08A]" : "text-[#565C66] group-hover:text-[#8A919C]"
                      }`}
                  />
                  <span className="tracking-wider">{route.label}</span>
                </div>

                <span
                  className={`text-[10px] font-mono ${isActive ? "text-[#00E08A]" : "text-[#565C66]"
                    }`}
                >
                  {route.index}
                </span>
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Persistent Bottom Status Footprint */}
      <div className="p-3 border-t border-white/10 bg-[#12151A] bezel-depth-subtle space-y-2">
        <div className="flex items-center justify-between font-mono text-[10px] text-[#8A919C]">
          <span>CURRENT STATE</span>
          <span className="text-[#00E08A] font-semibold flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00E08A] glow-nominal" />
            {currentState}
          </span>
        </div>

        <div className="flex items-center justify-between font-mono text-[10px] text-[#8A919C]">
          <span>CONFIDENCE</span>
          <span className="text-[#E6E9ED] font-bold">{(confidence * 100).toFixed(0)}%</span>
        </div>

        <div className="flex items-center justify-between font-mono text-[10px] text-[#8A919C] pt-1 border-t border-white/5">
          <span>CYCLES COMPLETED</span>
          <span className="text-[#4DA3FF] font-bold">#{cyclesCompleted}</span>
        </div>
      </div>
    </aside>
  );
}
