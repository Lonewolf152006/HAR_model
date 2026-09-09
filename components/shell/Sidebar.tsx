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
  ArrowLeft,
} from "lucide-react";
import { APP_ROUTES } from "@/lib/constants";

const ICON_MAP = {
  "live-feed": Video,
  "sop-tracker": ListOrdered,
  logs: Terminal,
  stream: Radio,
  settings: Sliders,
};

export function Sidebar() {
  const pathname = usePathname();

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
              ASTROFLOW AI
            </span>
          </div>
        </div>

        {/* Navigation Breaker Switchboard */}
        <nav className="p-2 space-y-1">
          <div className="px-2 py-1.5 text-[9px] font-mono uppercase tracking-widest text-[#565C66]">
            NAVIGATION: CONSOLE BUS
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

      {/* Persistent Bottom Utility Footer */}
      <div className="p-3 border-t border-white/10 bg-[#12151A] bezel-depth-subtle space-y-2.5">
        <div className="flex items-center justify-between font-mono text-[10px] text-[#565C66] pb-2 border-b border-white/5">
          <span>SESSION ID</span>
          <span className="text-[#8A919C] tracking-wider">EXP-2026-0924</span>
        </div>

        <Link
          href="/"
          className="group flex items-center gap-2 py-1 px-1 text-xs font-mono text-[#8A919C] hover:text-[#E6E9ED] transition-colors rounded-[2px]"
          title="Exit avionics console and return to mission briefing"
        >
          <ArrowLeft className="w-3.5 h-3.5 text-[#565C66] group-hover:text-[#E6E9ED] transition-transform group-hover:-translate-x-0.5" />
          <span className="tracking-wider text-[11px] font-medium">RETURN TO MISSION BRIEF</span>
        </Link>
      </div>
    </aside>
  );
}
