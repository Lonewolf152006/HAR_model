import type { ApiResponse, StateSnapshot } from "../../../../lib/backend/types";

// GET /api/v1/state
// Returns the current FSM snapshot. This is a stub that will be wired to the
// real telemetry stream in a follow-up; it exists so the API contract is
// documented and testable from day one.

export function GET(): Response {
  const snapshot: StateSnapshot = {
    currentState: "idle",
    expectedNext: "open_box",
    gates: {
      confidence: { id: "confidence", name: "Confidence", passed: true, value: 1, threshold: 0.52, reason: "ok" },
      stability: { id: "stability", name: "Stability", passed: true, value: 0, threshold: 5, reason: "ok" },
      cooldown: { id: "cooldown", name: "Cooldown", passed: true, value: 0, threshold: 0.85, reason: "ok" },
      motion: { id: "motion", name: "Motion", passed: true, value: 0, threshold: 0.12, reason: "ok" },
      causal_logic: { id: "causal_logic", name: "Causal Logic", passed: true, value: 0, threshold: 1, reason: "ok" },
    },
    containment: { red_box: "OUTSIDE", blue_box: "INSIDE", main_box: "OPEN" },
    capturedAt: new Date().toISOString(),
  };

  const body: ApiResponse<StateSnapshot> = {
    ok: true,
    data: snapshot,
    error: null,
    meta: {
      requestId: `req_${Date.now().toString(36)}`,
      timestamp: new Date().toISOString(),
    },
  };

  return Response.json(body, { status: 200 });
}