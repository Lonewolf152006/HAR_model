"use client";

import React from "react";

interface AvionicsPanelProps {
  children: React.ReactNode;
  title?: string;
  indexTag?: string;
  badge?: React.ReactNode;
  className?: string;
  bracketColor?: string;
  headerBorder?: boolean;
}

export function AvionicsPanel({
  children,
  title,
  indexTag,
  badge,
  className = "",
  bracketColor = "border-[#00E08A]/60",
  headerBorder = true,
}: AvionicsPanelProps) {
  return (
    <div
      className={`relative bg-[#12151A] bezel-depth border border-white/10 rounded-[2px] p-3 transition-colors ${className}`}
    >
      {/* 4 Precision HUD Corner Bracket Accents */}
      {/* Top-Left */}
      <span
        className={`absolute -top-[1px] -left-[1px] w-2 h-2 border-t-[1.5px] border-l-[1.5px] ${bracketColor} pointer-events-none`}
      />
      {/* Top-Right */}
      <span
        className={`absolute -top-[1px] -right-[1px] w-2 h-2 border-t-[1.5px] border-r-[1.5px] ${bracketColor} pointer-events-none`}
      />
      {/* Bottom-Left */}
      <span
        className={`absolute -bottom-[1px] -left-[1px] w-2 h-2 border-b-[1.5px] border-l-[1.5px] ${bracketColor} pointer-events-none`}
      />
      {/* Bottom-Right */}
      <span
        className={`absolute -bottom-[1px] -right-[1px] w-2 h-2 border-b-[1.5px] border-r-[1.5px] ${bracketColor} pointer-events-none`}
      />

      {/* Optional Panel Header */}
      {(title || indexTag || badge) && (
        <div
          className={`flex items-center justify-between pb-2 mb-2.5 ${
            headerBorder ? "border-b border-white/10" : ""
          }`}
        >
          <div className="flex items-center gap-2">
            {indexTag && (
              <span className="font-mono text-[9px] text-[#565C66] tracking-wider">
                {indexTag}
              </span>
            )}
            {title && (
              <h3 className="font-mono text-[11px] font-bold uppercase tracking-wider text-[#E6E9ED]">
                {title}
              </h3>
            )}
          </div>
          {badge && <div>{badge}</div>}
        </div>
      )}

      {children}
    </div>
  );
}
