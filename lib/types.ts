export type HarState =
  | "idle"
  | "open_box"
  | "pick_red"
  | "place_red_out"
  | "pick_blue"
  | "place_blue_in"
  | "close_box";

export type GateId =
  | "confidence"
  | "stability"
  | "cooldown"
  | "motion"
  | "causal_logic";

export interface GateResult {
  id: GateId;
  name: string;
  passed: boolean;
  reason: string;
  value: number | string;
  threshold: number | string;
}

export type GateResults = Record<GateId, GateResult>;

export type ContainmentStatus = "INSIDE" | "OUTSIDE" | "HELD";

export interface ContainmentState {
  red_box: ContainmentStatus;
  blue_box: ContainmentStatus;
  main_box: "CLOSED" | "OPEN" | "AJAR";
}

export interface FsmBooleans {
  box_open: boolean;
  red_picked: boolean;
  red_placed_out: boolean;
  blue_picked: boolean;
  blue_placed_in: boolean;
}

export interface ActiveAlertState {
  active: boolean;
  reason: string;
  timestamp: string;
}

export interface TelemetryFrame {
  frame_id: number;
  timestamp: string;
  current_state: HarState;
  expected_next: HarState;
  confidence: number;
  stability_count: number;
  state_duration_ms: number;
  motion_energy: number;
  gates: GateResults;
  containment: ContainmentState;
  fsm: FsmBooleans;
  is_transition: boolean;
  transition_status: "nominal" | "rejected" | "alert";
  active_alert: ActiveAlertState | null;
  latency_ms: number;
  fps: number;
  confidence_history?: number[];
  cycles_completed?: number;
  boxes?: BoundingBox[];
  pose_points?: Array<{ x: number; y: number; v: number }>;
  hands_points?: Array<Array<{ x: number; y: number }>>;
  pose_locked?: boolean;
}

export type LogLevel = "ACCEPTED" | "REJECTED" | "ALERT" | "CONTAINMENT" | "INFO";

export interface LogEntry {
  id: string;
  timestamp: string;
  level: LogLevel;
  state: HarState;
  reason: string;
  message?: string;
  confidence: number;
}

export type SubsystemHealth = "nominal" | "degraded" | "fault";

export interface SubsystemStatusMap {
  camera: SubsystemHealth;
  tar_model: SubsystemHealth;
  yolo: SubsystemHealth;
  stream: SubsystemHealth;
}

export interface StateDefinition {
  id: HarState;
  name: string;
  index: number;
  description: string;
  expectedNext: HarState;
}

export type AnomalyType =
  | "out_of_sequence"
  | "confidence_drop"
  | "cooldown_violation"
  | "motion_stall";

export interface BoundingBox {
  id: string;
  label: string;
  confidence: number;
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  w: number; // percentage
  h: number; // percentage
  color: string;
  status: string;
}

export interface RejectionInfo {
  step: HarState;
  reason: string;
  timestamp: string;
}

export type StreamConnectionState = "CONNECTED" | "RECONNECTING" | "OFFLINE";

export interface StreamEvent {
  id: string;
  timestamp: string;
  type: "CONNECT" | "DISCONNECT" | "RECONNECT" | "CONFIG";
  message: string;
}

export interface StreamTargetConfig {
  ip: string;
  port: string;
  protocol: "WebRTC (RTP)" | "RTSP (H.264)" | "SRT (Low-Latency)";
}

export interface SignalQualityState {
  bars: number;
  maxBars: number;
  packetLossPct: number;
  jitterMs: number;
  snrDb: number;
}

export interface RecordingState {
  isRecording: boolean;
  elapsedMs: number;
  filePath: string;
  fileSizeBytes: number;
  remainingDiskBytes: number;
  totalDiskBytes: number;
}

export interface NetworkBandwidthState {
  bandwidthMbps: number;
  bandwidthTrend: "up" | "down" | "stable";
  latencyMs: number;
  latencyTrend: "up" | "down" | "stable";
}

export type CameraSourceId = "primary" | "secondary" | "file" | string;

export interface CameraSourceConfig {
  id: CameraSourceId;
  label: string;
  sensor: string;
  resolutionFps: string;
  bus: string;
}

export interface GateThresholds {
  confidence: number;      // default 0.52
  stabilityWindow: number; // default 5 frames
  cooldown: number;        // default 0.85 seconds
  motionFloor: number;     // default 0.12 energy
}
