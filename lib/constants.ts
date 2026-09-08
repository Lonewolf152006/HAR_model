import { HarState, StateDefinition, CameraSourceConfig, GateThresholds } from "./types";

export const HAR_STATES: StateDefinition[] = [
  {
    id: "idle",
    index: 0,
    name: "IDLE_STANDBY",
    description: "Astronaut stationary, baseline posture calibrating",
    expectedNext: "open_box",
  },
  {
    id: "open_box",
    index: 1,
    name: "OPEN_CONTAINER",
    description: "Release latch, open main experiment stowage container",
    expectedNext: "pick_red",
  },
  {
    id: "pick_red",
    index: 2,
    name: "PICK_RED_CUBE",
    description: "Grasp red sample cube from interior housing",
    expectedNext: "place_red_out",
  },
  {
    id: "place_red_out",
    index: 3,
    name: "PLACE_RED_EXTERIOR",
    description: "Deposit red sample cube onto exterior workbench bracket",
    expectedNext: "pick_blue",
  },
  {
    id: "pick_blue",
    index: 4,
    name: "PICK_BLUE_CUBE",
    description: "Grasp blue secondary sample cube from docking mount",
    expectedNext: "place_blue_in",
  },
  {
    id: "place_blue_in",
    index: 5,
    name: "PLACE_BLUE_INTERIOR",
    description: "Insert blue sample cube into main experiment container",
    expectedNext: "close_box",
  },
  {
    id: "close_box",
    index: 6,
    name: "CLOSE_CONTAINER",
    description: "Engage container lid, latch sealed, cycle complete",
    expectedNext: "idle",
  },
];

export const STATE_MAP: Record<HarState, StateDefinition> = HAR_STATES.reduce(
  (acc, state) => {
    acc[state.id] = state;
    return acc;
  },
  {} as Record<HarState, StateDefinition>
);

export const GATE_THRESHOLDS = {
  CONFIDENCE_MIN: 0.75,
  STABILITY_FRAMES_MIN: 14,
  COOLDOWN_MS: 1400,
  MOTION_ENERGY_MIN: 0.12,
};

export const BOOT_SEQUENCE_LINES = [
  { tag: "[OK]", text: "Camera sensor feed initialized (Sony IMX477 | 1080p60)", delay: 180 },
  { tag: "[OK]", text: "Pose & Hand tracker loaded (MediaPipe v0.10.14)", delay: 360 },
  { tag: "[OK]", text: "TAR model weights loaded (best_tar_bilstm_attn.pth)", delay: 540 },
  { tag: "[OK]", text: "YOLOv8 bounding detector online (yolo_boxes_v4.engine)", delay: 720 },
  { tag: "[CALIBRATING]", text: "Geometric containment engine (2.5D coordinate space)...", delay: 920 },
  { tag: "[OK]", text: "Containment engine calibrated (main_box / red / blue)", delay: 1120 },
  { tag: "[OK]", text: "DecisionStabilizer 5-gate pipeline armed", delay: 1300 },
  { tag: "[READY]", text: "Telemetry link synchronized. All systems nominal.", delay: 1520 },
];

export const APP_ROUTES = [
  { id: "live-feed", label: "LIVE FEED", path: "/console/live-feed", index: "01" },
  { id: "sop-tracker", label: "SOP TRACKER", path: "/console/sop-tracker", index: "02" },
  { id: "logs", label: "BLACK-BOX LOGS", path: "/console/logs", index: "03" },
  { id: "stream", label: "STREAM & REC", path: "/console/stream", index: "04" },
  { id: "settings", label: "SETTINGS", path: "/console/settings", index: "05" },
];

export const DEFAULT_GATE_THRESHOLDS: GateThresholds = {
  confidence: 0.52,
  stabilityWindow: 5,
  cooldown: 0.85,
  motionFloor: 0.12,
};

export const CAMERA_SOURCES: CameraSourceConfig[] = [
  {
    id: "primary",
    label: "PRIMARY (IMX477)",
    sensor: "SONY IMX477 MIPI-CSI2",
    resolutionFps: "1080p60",
    bus: "MIPI CSI-2 | 4-LANE D-PHY",
  },
  {
    id: "secondary",
    label: "SECONDARY (C920)",
    sensor: "LOGITECH C920 OPTICS",
    resolutionFps: "720p30",
    bus: "USB 3.1 GEN 1 | ISOCHRONOUS",
  },
  {
    id: "file",
    label: "FILE INPUT",
    sensor: "RAW H.265 DECODER",
    resolutionFps: "1080p60",
    bus: "NVRAM V4L2-LOOPBACK (/dev/video0)",
  },
];

export const TELEMETRY_SESSION_ID = "EXP-2026-0924";
