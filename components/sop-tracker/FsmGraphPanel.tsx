"use client";

import React from "react";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { useTelemetry } from "@/context/TelemetryContext";
import { GitCommit, ShieldAlert } from "lucide-react";

interface GraphNode {
  id: string;
  label: string;
  x: number;
  y: number;
}

const GRAPH_NODES: GraphNode[] = [
  { id: "idle", label: "IDLE", x: 42, y: 28 },
  { id: "open_box", label: "OPEN", x: 126, y: 28 },
  { id: "pick_red", label: "P_RED", x: 210, y: 28 },
  { id: "place_red_out", label: "PL_RED", x: 294, y: 28 },
  { id: "pick_blue", label: "P_BLUE", x: 294, y: 84 },
  { id: "place_blue_in", label: "PL_BLUE", x: 210, y: 84 },
  { id: "close_box", label: "CLOSE", x: 126, y: 84 },
];

const VALID_EDGES = [
  { from: "idle", to: "open_box", d: "M 66 28 L 102 28" },
  { from: "open_box", to: "pick_red", d: "M 150 28 L 186 28" },
  { from: "pick_red", to: "place_red_out", d: "M 234 28 L 270 28" },
  { from: "place_red_out", to: "pick_blue", d: "M 294 40 L 294 72" },
  { from: "pick_blue", to: "place_blue_in", d: "M 270 84 L 234 84" },
  { from: "place_blue_in", to: "close_box", d: "M 186 84 L 150 84" },
  { from: "close_box", to: "idle", d: "M 102 84 Q 60 84 48 44" },
];

// Impossible / blocked edges explicitly constrained by PhysicalCausalLogic
const BLOCKED_EDGES = [
  {
    from: "idle",
    to: "pick_red",
    d: "M 64 34 Q 135 52 188 34",
    rule: "BOX_CLOSED_VIOLATION",
  },
  {
    from: "open_box",
    to: "pick_blue",
    d: "M 148 38 Q 210 65 272 78",
    rule: "RED_MUST_CLEAR_FIRST",
  },
  {
    from: "pick_red",
    to: "close_box",
    d: "M 200 40 Q 170 60 148 72",
    rule: "CANNOT_SEAL_WHILE_HELD",
  },
  {
    from: "place_red_out",
    to: "close_box",
    d: "M 280 40 Q 210 62 150 80",
    rule: "BLUE_UNSTOWED_VIOLATION",
  },
];

export function FsmGraphPanel() {
  const { currentState, expectedNext, fsm } = useTelemetry();

  const fsmItems = [
    { key: "box_open", label: "BOX_OPEN", val: fsm.box_open, rule: "LID >= 45°" },
    {
      key: "red_picked",
      label: "RED_PICKED",
      val: fsm.red_picked,
      rule: "GRIP RED",
    },
    {
      key: "red_placed_out",
      label: "RED_PLACED_OUT",
      val: fsm.red_placed_out,
      rule: "EXT BENCH",
    },
    {
      key: "blue_picked",
      label: "BLUE_PICKED",
      val: fsm.blue_picked,
      rule: "GRIP BLUE",
    },
    {
      key: "blue_placed_in",
      label: "BLUE_PLACED_IN",
      val: fsm.blue_placed_in,
      rule: "INT STOWED",
    },
  ];

  return (
    <AvionicsPanel
      title="CAUSAL FSM TOPOLOGY"
      indexTag="02 // DIRECTED GRAPH"
      badge={
        <div className="flex items-center gap-1 font-mono text-[9px] text-[#00E08A]">
          <GitCommit className="w-3 h-3 glow-nominal" />
          <span>CAUSAL ENFORCED</span>
        </div>
      }
      className="h-full flex flex-col justify-between overflow-hidden p-2.5"
    >
      {/* Mini Directed Graph Diagram */}
      <div className="relative w-full h-[115px] bg-[#0E1015] border border-white/5 rounded-[2px] bezel-depth-subtle flex items-center justify-center overflow-hidden shrink-0">
        <svg
          viewBox="0 0 336 112"
          className="w-full h-full select-none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            {/* Outbound Active Edge Arrow Marker */}
            <marker
              id="arrow-active"
              viewBox="0 0 6 6"
              refX="5"
              refY="3"
              markerWidth="4"
              markerHeight="4"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 6 3 L 0 6 z" fill="#00E08A" />
            </marker>

            {/* Standard Edge Arrow Marker */}
            <marker
              id="arrow-normal"
              viewBox="0 0 6 6"
              refX="5"
              refY="3"
              markerWidth="4"
              markerHeight="4"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 6 3 L 0 6 z" fill="#565C66" fillOpacity="0.6" />
            </marker>
          </defs>

          {/* Blocked / Causal Impossible Edges (Dashed and Dim) */}
          {BLOCKED_EDGES.map((edge, i) => (
            <path
              key={`blocked-${i}`}
              d={edge.d}
              fill="none"
              stroke="#565C66"
              strokeOpacity="0.25"
              strokeWidth="1"
              strokeDasharray="2.5 2.5"
            />
          ))}

          {/* Valid Sequential Edges */}
          {VALID_EDGES.map((edge, i) => {
            const isActiveOutbound =
              edge.from === currentState && edge.to === expectedNext;

            return (
              <path
                key={`valid-${i}`}
                d={edge.d}
                fill="none"
                stroke={isActiveOutbound ? "#00E08A" : "#565C66"}
                strokeOpacity={isActiveOutbound ? 1 : 0.45}
                strokeWidth={isActiveOutbound ? 2.2 : 1.2}
                markerEnd={
                  isActiveOutbound ? "url(#arrow-active)" : "url(#arrow-normal)"
                }
                className={isActiveOutbound ? "glow-nominal" : ""}
              />
            );
          })}

          {/* Graph Nodes */}
          {GRAPH_NODES.map((node) => {
            const isActive = node.id === currentState;
            return (
              <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
                <rect
                  x="-23"
                  y="-11"
                  width="46"
                  height="22"
                  rx="2"
                  fill={isActive ? "#171B21" : "#12151A"}
                  stroke={isActive ? "#00E08A" : "#343A46"}
                  strokeWidth={isActive ? 1.8 : 1}
                  className={isActive ? "glow-nominal" : ""}
                />
                <text
                  x="0"
                  y="3.5"
                  textAnchor="middle"
                  fill={isActive ? "#00E08A" : "#8A919C"}
                  fontSize="7.5"
                  fontFamily="monospace"
                  fontWeight={isActive ? "bold" : "normal"}
                  letterSpacing="0.5"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Legend Overlay */}
        <div className="absolute bottom-1 right-2 flex items-center gap-2 font-mono text-[7.5px] text-[#565C66]">
          <span className="flex items-center gap-1">
            <span className="w-2 h-[1px] bg-[#00E08A]" />
            <span>ACTIVE EDGE</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-[1px] border-b border-dashed border-[#565C66]" />
            <span>BLOCKED CAUSAL</span>
          </span>
        </div>
      </div>

      {/* Compact FSM Boolean Readout Table */}
      <div className="flex-1 flex flex-col justify-end mt-2 space-y-1">
        <div className="flex items-center justify-between pb-1 border-b border-white/5 font-mono text-[8px] text-[#565C66] tracking-wider">
          <span>FSM STATE REGISTER</span>
          <span>ASSERTION</span>
          <span>VALUE</span>
        </div>

        <div className="space-y-1 overflow-y-auto max-h-[125px] pr-0.5">
          {fsmItems.map((item) => (
            <div
              key={item.key}
              className="flex items-center justify-between px-2 py-0.5 bg-[#171B21] border border-white/5 rounded-[2px] bezel-depth-subtle font-mono text-[10px]"
            >
              <div className="flex items-center gap-1.5">
                <span
                  className={`w-1.5 h-1.5 rounded-[1px] ${
                    item.val ? "bg-[#00E08A] glow-nominal" : "bg-[#565C66]"
                  }`}
                />
                <span className="text-[#E6E9ED] font-semibold">{item.label}</span>
              </div>

              <span className="text-[9px] text-[#565C66] tracking-tight">
                {item.rule}
              </span>

              <span
                className={`font-bold text-[10px] ${
                  item.val ? "text-[#00E08A]" : "text-[#565C66]"
                }`}
              >
                {item.val ? "TRUE" : "FALSE"}
              </span>
            </div>
          ))}
        </div>
      </div>
    </AvionicsPanel>
  );
}
