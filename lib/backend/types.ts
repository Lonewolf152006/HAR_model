import type { HarState, GateResults, ContainmentState } from "../types";

// Standardized API envelope returned by every AstroFlow-AI endpoint.
// Keeps the wire contract stable regardless of internal representation.

export interface ApiResponse<T> {
  ok: boolean;
  data: T | null;
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  } | null;
  meta: {
    requestId: string;
    timestamp: string;
  };
}

// State mutation request accepted by the FSM transition endpoint.
export interface TransitionRequest {
  targetState: HarState;
  operatorOverride?: boolean;
  reason?: string;
}

// Snapshot returned by GET /state.
export interface StateSnapshot {
  currentState: HarState;
  expectedNext: HarState;
  gates: GateResults;
  containment: ContainmentState;
  capturedAt: string;
}

export type { HarState };