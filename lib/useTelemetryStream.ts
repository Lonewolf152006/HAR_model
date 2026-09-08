"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  HarState,
  TelemetryFrame,
  LogEntry,
  GateResults,
  ContainmentState,
  FsmBooleans,
  SubsystemStatusMap,
  AnomalyType,
  BoundingBox,
  StreamConnectionState,
  StreamEvent,
  StreamTargetConfig,
  SignalQualityState,
  RecordingState,
  NetworkBandwidthState,
  CameraSourceId,
  GateThresholds,
} from "./types";
import { HAR_STATES, GATE_THRESHOLDS, DEFAULT_GATE_THRESHOLDS } from "./constants";
import { avionicsAudio } from "./sound";

function formatTimestamp(date: Date): string {
  const pad = (n: number, z = 2) => String(n).padStart(z, "0");
  const h = pad(date.getHours());
  const m = pad(date.getMinutes());
  const s = pad(date.getSeconds());
  const ms = pad(date.getMilliseconds(), 3);
  return `${h}:${m}:${s}.${ms}`;
}

const STATE_DURATIONS: Record<HarState, number> = {
  idle: 4500,
  open_box: 5000,
  pick_red: 4500,
  place_red_out: 5200,
  pick_blue: 4800,
  place_blue_in: 5000,
  close_box: 4600,
};

const INITIAL_CONFIDENCE_HISTORY: number[] = Array.from({ length: 50 }, (_, i) => {
  return +(0.84 + Math.sin(i * 0.25) * 0.07 + Math.cos(i * 0.4) * 0.04).toFixed(2);
});

export function useTelemetryStream() {
  const [frameId, setFrameId] = useState<number>(1042);
  const [stateIndex, setStateIndex] = useState<number>(0);
  const [cyclesCompleted, setCyclesCompleted] = useState<number>(3);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [isReducedMotion, setIsReducedMotion] = useState<boolean>(false);
  const [confidenceHistory, setConfidenceHistory] = useState<number[]>(
    INITIAL_CONFIDENCE_HISTORY
  );

  const [subsystems, setSubsystems] = useState<SubsystemStatusMap>({
    camera: "nominal",
    tar_model: "nominal",
    yolo: "nominal",
    stream: "nominal",
  });

  // Stream & Recording Infrastructure State
  const [streamStatus, setStreamStatus] = useState<StreamConnectionState>("CONNECTED");
  const [streamConfig, setStreamConfig] = useState<StreamTargetConfig>({
    ip: "10.0.4.128",
    port: "8554",
    protocol: "WebRTC // RTP",
  });
  const [streamEvents, setStreamEvents] = useState<StreamEvent[]>([
    {
      id: "se-1",
      timestamp: "12:00:00.080",
      type: "CONFIG",
      message: "Target endpoint armed: 10.0.4.128:8554 (WebRTC // RTP)",
    },
    {
      id: "se-2",
      timestamp: "12:00:00.320",
      type: "CONNECT",
      message: "DTLS handshake & SRTP key exchange completed",
    },
    {
      id: "se-3",
      timestamp: "12:00:01.040",
      type: "CONNECT",
      message: "H.264 stream active // 1080p60 @ 14.8 Mbps // Sub-20ms",
    },
  ]);
  const [recordingState, setRecordingState] = useState<RecordingState>({
    isRecording: true,
    elapsedMs: 884000,
    filePath: "/data/nvram0/rec_columbus_har_20260924_01.mkv",
    fileSizeBytes: 1642000000,
    remainingDiskBytes: 118400000000,
    totalDiskBytes: 128000000000,
  });
  const [networkStats, setNetworkStats] = useState<NetworkBandwidthState>({
    bandwidthMbps: 14.8,
    bandwidthTrend: "stable",
    latencyMs: 18,
    latencyTrend: "stable",
  });
  const [signalQuality, setSignalQuality] = useState<SignalQualityState>({
    bars: 5,
    maxBars: 6,
    packetLossPct: 0.04,
    jitterMs: 2.1,
    snrDb: 28.4,
  });

  // Settings & Configuration State
  const [thresholds, setThresholds] = useState<GateThresholds>(DEFAULT_GATE_THRESHOLDS);
  const [cameraSource, setCameraSource] = useState<CameraSourceId>("primary");
  const [isReplayingBoot, setIsReplayingBoot] = useState<boolean>(false);

  const updateThresholds = useCallback((newThresholds: Partial<GateThresholds>) => {
    setThresholds((prev) => ({ ...prev, ...newThresholds }));
  }, []);

  const restoreDefaultThresholds = useCallback(() => {
    setThresholds(DEFAULT_GATE_THRESHOLDS);
  }, []);

  const replayBootSequence = useCallback(() => {
    if (typeof window !== "undefined") {
      sessionStorage.removeItem("orbital_har_booted");
    }
    setIsReplayingBoot(true);
  }, []);

  const handleBootComplete = useCallback(() => {
    setIsReplayingBoot(false);
  }, []);

  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: "log-init-1",
      timestamp: "12:00:00.120",
      level: "INFO",
      state: "idle",
      reason: "SYSTEM: Flight recorder memory buffer armed (NVRAM partition 0x4B)",
      message: "SYSTEM: Flight recorder memory buffer armed (NVRAM partition 0x4B)",
      confidence: 0.99,
    },
    {
      id: "log-init-2",
      timestamp: "12:00:00.340",
      level: "INFO",
      state: "idle",
      reason: "IMX477 Camera subsystem synchronized at 1080p60",
      message: "IMX477 Camera subsystem synchronized at 1080p60",
      confidence: 0.98,
    },
    {
      id: "log-init-3",
      timestamp: "12:00:00.580",
      level: "INFO",
      state: "idle",
      reason: "MediaPipe hand & body landmark detectors online",
      message: "MediaPipe hand & body landmark detectors online",
      confidence: 0.97,
    },
    {
      id: "log-init-4",
      timestamp: "12:00:00.890",
      level: "INFO",
      state: "idle",
      reason: "TAR BiLSTM attention model loaded (best_tar_bilstm_attn.pth)",
      message: "TAR BiLSTM attention model loaded (best_tar_bilstm_attn.pth)",
      confidence: 0.99,
    },
    {
      id: "log-init-5",
      timestamp: "12:00:01.042",
      level: "INFO",
      state: "idle",
      reason: "DecisionStabilizer armed (5-gate verification pipeline active)",
      message: "DecisionStabilizer armed (5-gate verification pipeline active)",
      confidence: 0.96,
    },
    {
      id: "log-init-6",
      timestamp: "12:00:01.400",
      level: "INFO",
      state: "idle",
      reason: "Geometric containment calibrated for 2.5D coordinate space",
      message: "Geometric containment calibrated for 2.5D coordinate space",
      confidence: 0.98,
    },
    {
      id: "log-init-7",
      timestamp: "12:00:02.110",
      level: "ACCEPTED",
      state: "open_box",
      reason: "STATE → open_box (conf 0.89) // All 5 DecisionStabilizer gates satisfied",
      message: "STATE → open_box (conf 0.89) // All 5 DecisionStabilizer gates satisfied",
      confidence: 0.89,
    },
    {
      id: "log-init-8",
      timestamp: "12:00:02.115",
      level: "INFO",
      state: "open_box",
      reason: "CONTAINMENT: Main Stowage Box latch released [OPEN]",
      message: "CONTAINMENT: Main Stowage Box latch released [OPEN]",
      confidence: 0.99,
    },
    {
      id: "log-init-9",
      timestamp: "12:00:06.240",
      level: "ACCEPTED",
      state: "pick_red",
      reason: "STATE → pick_red (conf 0.94) // All 5 DecisionStabilizer gates satisfied",
      message: "STATE → pick_red (conf 0.94) // All 5 DecisionStabilizer gates satisfied",
      confidence: 0.94,
    },
    {
      id: "log-init-10",
      timestamp: "12:00:06.245",
      level: "INFO",
      state: "pick_red",
      reason: "CONTAINMENT: Red Cube [Sample-A] grasp confirmed [HELD]",
      message: "CONTAINMENT: Red Cube [Sample-A] grasp confirmed [HELD]",
      confidence: 0.95,
    },
  ]);

  const [activeAlert, setActiveAlert] = useState<string | null>(null);
  const [activeAnomaly, setActiveAnomaly] = useState<AnomalyType | null>(null);
  const [activeRejection, setActiveRejection] = useState<{
    step: HarState;
    reason: string;
    timestamp: string;
  } | null>(null);
  const [lastAcceptedState, setLastAcceptedState] = useState<HarState | null>(null);

  // Internal mutable simulation refs
  const stateStartRef = useRef<number>(Date.now());
  const lastTransitionRef = useRef<number>(Date.now() - 3000);
  const stabilityCounterRef = useRef<number>(18);
  const frameCounterRef = useRef<number>(1042);
  const stateIndexRef = useRef<number>(0);
  const anomalyTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const currentStateDef = HAR_STATES[stateIndex];
  const currentState = currentStateDef.id;
  const expectedNext = currentStateDef.expectedNext;

  // Append a log entry to the running chronological buffer (capped at 500 lines)
  const appendLog = useCallback(
    (entry: Omit<LogEntry, "id"> & { id?: string }) => {
      const newEntry: LogEntry = {
        id: entry.id || `log-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
        message: entry.reason,
        ...entry,
      };
      setLogs((prev) => {
        if (prev.length >= 500) {
          return [...prev.slice(prev.length - 499), newEntry];
        }
        return [...prev, newEntry];
      });
    },
    []
  );

  // Sync mute state with sound system
  useEffect(() => {
    avionicsAudio.setMuted(isMuted);
  }, [isMuted]);

  // Handle manual anomaly injection
  const injectAnomaly = useCallback((type: AnomalyType) => {
    setActiveAnomaly(type);
    const now = new Date();
    const timeStr = formatTimestamp(now);

    if (type === "out_of_sequence") {
      setActiveAlert("OUT-OF-SEQUENCE: Illegal state skipped (FSM transition blocked)");
      setActiveRejection({
        step: currentState,
        reason: "Blocked: Red cube must be placed out before picking blue",
        timestamp: timeStr,
      });
      avionicsAudio.playAlertTone();
      avionicsAudio.speakVoiceAlert("Warning: Out of sequence activity detected.");
      appendLog({
        timestamp: timeStr,
        level: "ALERT",
        state: currentState,
        reason: "OUT-OF-SEQUENCE: Illegal state skipped (FSM transition blocked)",
        confidence: 0.54,
      });
      appendLog({
        timestamp: timeStr,
        level: "REJECTED",
        state: currentState,
        reason: "Blocked: Red cube must be placed out before picking blue",
        confidence: 0.54,
      });
    } else if (type === "confidence_drop") {
      setActiveAlert("GATE REJECTION: Confidence dropped below threshold (0.58 < 0.75)");
      setActiveRejection({
        step: currentState,
        reason: "Low conf (0.58 < 0.75)",
        timestamp: timeStr,
      });
      avionicsAudio.playRejectedTone();
      appendLog({
        timestamp: timeStr,
        level: "REJECTED",
        state: currentState,
        reason: "GATE REJECTION: Low conf (0.58 < 0.75 threshold)",
        confidence: 0.58,
      });
    } else if (type === "cooldown_violation") {
      setActiveAlert("GATE REJECTION: Transition attempted within 1.4s cooldown window");
      setActiveRejection({
        step: currentState,
        reason: "Cooldown active (0.6s < 1.4s)",
        timestamp: timeStr,
      });
      avionicsAudio.playRejectedTone();
      appendLog({
        timestamp: timeStr,
        level: "REJECTED",
        state: currentState,
        reason: "GATE REJECTION: Cooldown active (0.6s elapsed < 1.4s required)",
        confidence: 0.86,
      });
    } else if (type === "motion_stall") {
      setActiveAlert("GATE REJECTION: Insufficient kinematic velocity (static posture hallucination)");
      setActiveRejection({
        step: currentState,
        reason: "Kinematic velocity stalled (<0.12 J)",
        timestamp: timeStr,
      });
      avionicsAudio.playRejectedTone();
      appendLog({
        timestamp: timeStr,
        level: "REJECTED",
        state: currentState,
        reason: "GATE REJECTION: Motion gate failed (kinematic energy 0.04 < 0.12 min)",
        confidence: 0.81,
      });
    }

    if (anomalyTimeoutRef.current) clearTimeout(anomalyTimeoutRef.current);
    anomalyTimeoutRef.current = setTimeout(() => {
      setActiveAnomaly(null);
      setActiveAlert(null);
    }, 3200);
  }, [currentState, appendLog]);

  // Main 10Hz simulation clock loop
  useEffect(() => {
    if (isPaused) return;

    const interval = setInterval(() => {
      frameCounterRef.current += 1;
      const curFrame = frameCounterRef.current;
      setFrameId(curFrame);

      const now = Date.now();
      const elapsedInState = now - stateStartRef.current;
      const targetDuration = STATE_DURATIONS[HAR_STATES[stateIndexRef.current].id];

      stabilityCounterRef.current += 1;

      // Calculate confidence for this tick
      let tickConf = +(0.86 + Math.sin(curFrame * 0.15) * 0.07 + Math.cos(curFrame * 0.08) * 0.03).toFixed(2);
      if (activeAnomaly === "confidence_drop") {
        tickConf = +(0.58 + Math.sin(curFrame * 0.3) * 0.04).toFixed(2);
      }

      setConfidenceHistory((prev) => [...prev.slice(1), tickConf]);

      // Advance local recording metrics if recording
      setRecordingState((prev) => {
        if (!prev.isRecording) return prev;
        const addBytes = 28000;
        return {
          ...prev,
          elapsedMs: prev.elapsedMs + 100,
          fileSizeBytes: prev.fileSizeBytes + addBytes,
          remainingDiskBytes: Math.max(0, prev.remainingDiskBytes - addBytes),
        };
      });

      // Update network bandwidth & latency metrics if connected
      setNetworkStats((prev) => {
        if (streamStatus !== "CONNECTED") {
          return {
            bandwidthMbps: 0,
            bandwidthTrend: "stable",
            latencyMs: 0,
            latencyTrend: "stable",
          };
        }
        const bDelta = Math.sin(curFrame * 0.1) * 0.4 + (Math.random() - 0.5) * 0.2;
        const nextBw = +(14.8 + bDelta).toFixed(1);
        const lDelta = Math.cos(curFrame * 0.15) * 1.5 + (Math.random() - 0.5) * 1;
        const nextLat = Math.round(18 + lDelta);
        return {
          bandwidthMbps: nextBw,
          bandwidthTrend: nextBw > prev.bandwidthMbps ? "up" : nextBw < prev.bandwidthMbps ? "down" : "stable",
          latencyMs: nextLat,
          latencyTrend: nextLat > prev.latencyMs ? "up" : nextLat < prev.latencyMs ? "down" : "stable",
        };
      });

      // Check if state is ready to advance normally
      if (elapsedInState >= targetDuration && !activeAnomaly) {
        // Transition to next state
        const nextIdx = (stateIndexRef.current + 1) % HAR_STATES.length;
        const acceptedState = HAR_STATES[nextIdx].id;
        stateIndexRef.current = nextIdx;
        setStateIndex(nextIdx);
        stateStartRef.current = now;
        lastTransitionRef.current = now;
        stabilityCounterRef.current = 1;

        const timeStr = formatTimestamp(new Date());

        // Audio & Log for accepted step
        avionicsAudio.playAcceptedTone();
        setLastAcceptedState(acceptedState);
        setActiveRejection(null);

        const stepConf = +(0.88 + Math.random() * 0.08).toFixed(2);
        appendLog({
          timestamp: timeStr,
          level: "ACCEPTED",
          state: acceptedState,
          reason: `STATE → ${acceptedState} (conf ${stepConf}) // All 5 DecisionStabilizer gates satisfied`,
          confidence: stepConf,
        });

        // Specific containment log for state transition
        if (acceptedState === "open_box") {
          appendLog({ timestamp: timeStr, level: "INFO", state: acceptedState, reason: "CONTAINMENT: Main Stowage Box latch released [OPEN]", confidence: 0.99 });
        } else if (acceptedState === "pick_red") {
          appendLog({ timestamp: timeStr, level: "INFO", state: acceptedState, reason: "CONTAINMENT: Red Cube [Sample-A] grasp confirmed [HELD]", confidence: 0.95 });
        } else if (acceptedState === "place_red_out") {
          appendLog({ timestamp: timeStr, level: "INFO", state: acceptedState, reason: "CONTAINMENT: Red Cube [Sample-A] deposited on exterior workbench [OUTSIDE]", confidence: 0.96 });
        } else if (acceptedState === "pick_blue") {
          appendLog({ timestamp: timeStr, level: "INFO", state: acceptedState, reason: "CONTAINMENT: Blue Cube [Sample-B] grasp confirmed [HELD]", confidence: 0.93 });
        } else if (acceptedState === "place_blue_in") {
          appendLog({ timestamp: timeStr, level: "INFO", state: acceptedState, reason: "CONTAINMENT: Blue Cube [Sample-B] inserted into stowage interior [INSIDE]", confidence: 0.97 });
        } else if (acceptedState === "close_box") {
          appendLog({ timestamp: timeStr, level: "INFO", state: acceptedState, reason: "CONTAINMENT: Main Stowage Box lid engaged and sealed [CLOSED]", confidence: 0.99 });
        }

        // If cycle completed
        if (nextIdx === 0) {
          setCyclesCompleted((c) => {
            const nextCycle = c + 1;
            appendLog({
              timestamp: timeStr,
              level: "INFO",
              state: "idle",
              reason: `SOP Full Cycle Completed #${nextCycle} — All containment restowed`,
              confidence: 0.98,
            });
            return nextCycle;
          });
          avionicsAudio.playCycleCompleteChime();
        }
      }
    }, 100);

    return () => clearInterval(interval);
  }, [isPaused, activeAnomaly]);

  // Derive realistic frame telemetry
  const timeStr = formatTimestamp(new Date());
  const currentElapsed = Date.now() - stateStartRef.current;
  const timeSinceTransSec = Math.max(0.1, (Date.now() - lastTransitionRef.current) / 1000);

  // Current confidence from latest history sample
  const confidence =
    confidenceHistory.length > 0
      ? confidenceHistory[confidenceHistory.length - 1]
      : 0.88;

  let motionEnergy = +(0.26 + Math.cos(frameId * 0.2) * 0.12).toFixed(2);
  if (currentState === "idle") {
    motionEnergy = 0.08;
  }
  if (activeAnomaly === "motion_stall") {
    motionEnergy = 0.04;
  }

  // Derive Containment status from state
  const containment: ContainmentState = {
    red_box:
      currentState === "pick_red"
        ? "HELD"
        : currentState === "place_red_out" ||
          currentState === "pick_blue" ||
          currentState === "place_blue_in"
          ? "OUTSIDE"
          : "INSIDE",
    blue_box:
      currentState === "pick_blue"
        ? "HELD"
        : currentState === "place_blue_in" || currentState === "close_box"
          ? "INSIDE"
          : "INSIDE",
    main_box:
      currentState === "idle"
        ? "CLOSED"
        : currentState === "close_box"
          ? "CLOSED"
          : "OPEN",
  };

  // Derive FSM Booleans
  const fsm: FsmBooleans = {
    box_open: currentState !== "idle" && currentState !== "close_box",
    red_picked: stateIndex >= 2 && stateIndex !== 0,
    red_placed_out: stateIndex >= 3 && stateIndex !== 0,
    blue_picked: stateIndex >= 4 && stateIndex !== 0,
    blue_placed_in: stateIndex >= 5 && stateIndex !== 0,
  };

  // Dynamic gate evaluations using mutable operator thresholds
  const isConfidencePass = confidence >= thresholds.confidence;
  const isStabilityPass = stabilityCounterRef.current >= thresholds.stabilityWindow;
  const isCooldownPass =
    activeAnomaly === "cooldown_violation"
      ? false
      : timeSinceTransSec >= thresholds.cooldown;
  const isMotionPass =
    currentState === "idle" ? true : motionEnergy >= thresholds.motionFloor;
  const isCausalPass = activeAnomaly !== "out_of_sequence";

  const gates: GateResults = {
    confidence: {
      id: "confidence",
      name: "Confidence Gate",
      passed: isConfidencePass,
      reason: isConfidencePass
        ? `Sufficient probability (${confidence} ≥ ${thresholds.confidence})`
        : `Under-confidence (${confidence} < ${thresholds.confidence})`,
      value: confidence,
      threshold: thresholds.confidence,
    },
    stability: {
      id: "stability",
      name: "Stability Window",
      passed: isStabilityPass,
      reason: isStabilityPass
        ? `Held for ${stabilityCounterRef.current} frames (min ${thresholds.stabilityWindow})`
        : `Buffer filling (${stabilityCounterRef.current}/${thresholds.stabilityWindow})`,
      value: stabilityCounterRef.current,
      threshold: thresholds.stabilityWindow,
    },
    cooldown: {
      id: "cooldown",
      name: "Transition Cooldown",
      passed: isCooldownPass,
      reason: isCooldownPass
        ? `Cooldown satisfied (+${timeSinceTransSec.toFixed(1)}s)`
        : `Cooldown active (${timeSinceTransSec.toFixed(1)}s < ${thresholds.cooldown.toFixed(2)}s)`,
      value: `${timeSinceTransSec.toFixed(1)}s`,
      threshold: `${thresholds.cooldown.toFixed(2)}s`,
    },
    motion: {
      id: "motion",
      name: "Kinematic Motion",
      passed: isMotionPass,
      reason: isMotionPass
        ? `Motion confirmed (${motionEnergy})`
        : `Velocity stalled (${motionEnergy} < ${thresholds.motionFloor})`,
      value: motionEnergy,
      threshold: thresholds.motionFloor,
    },
    causal_logic: {
      id: "causal_logic",
      name: "Causal / FSM Logic",
      passed: isCausalPass,
      reason: isCausalPass
        ? "FSM sequence consistent"
        : "Illegal sequence transition blocked by causal graph",
      value: isCausalPass ? "VALID" : "VIOLATION",
      threshold: "STRICT_ORDER",
    },
  };

  // Dynamic 2.5D bounding boxes tracking physical experiment state
  const boundingBoxes: BoundingBox[] = useMemo(() => {
    let redX = 26;
    let redY = 52;
    let redStatus = "INSIDE";

    let blueX = 39;
    let blueY = 52;
    let blueStatus = "INSIDE";

    let mainBoxStatus = "OPEN";

    if (currentState === "idle") {
      mainBoxStatus = "CLOSED";
      redX = 27;
      redY = 54;
      blueX = 40;
      blueY = 54;
    } else if (currentState === "open_box") {
      mainBoxStatus = "LATCH_OPEN";
      redX = 27;
      redY = 52;
      blueX = 40;
      blueY = 52;
    } else if (currentState === "pick_red") {
      mainBoxStatus = "OPEN";
      redX = 46;
      redY = 38;
      redStatus = "HELD";
    } else if (currentState === "place_red_out") {
      mainBoxStatus = "OPEN";
      redX = 72;
      redY = 50;
      redStatus = "OUTSIDE";
    } else if (currentState === "pick_blue") {
      mainBoxStatus = "OPEN";
      redX = 72;
      redStatus = "OUTSIDE";
      blueX = 46;
      blueY = 38;
      blueStatus = "HELD";
    } else if (currentState === "place_blue_in") {
      mainBoxStatus = "OPEN";
      redX = 72;
      redStatus = "OUTSIDE";
      blueX = 28;
      blueY = 52;
      blueStatus = "INSIDE";
    } else if (currentState === "close_box") {
      mainBoxStatus = "CLOSED";
      redX = 72;
      redStatus = "OUTSIDE";
      blueX = 28;
      blueStatus = "INSIDE";
    }

    return [
      {
        id: "main_box",
        label: "MAIN CONTAINER",
        confidence: 0.94,
        x: 20,
        y: 34,
        w: 38,
        h: 44,
        color: "#4DA3FF",
        status: mainBoxStatus,
      },
      {
        id: "red_box",
        label: "RED CUBE [SAMPLE-A]",
        confidence: +(0.88 + (frameId % 7) * 0.01).toFixed(2),
        x: redX,
        y: redY,
        w: 12,
        h: 15,
        color: "#FF4D4F",
        status: redStatus,
      },
      {
        id: "blue_box",
        label: "BLUE CUBE [SAMPLE-B]",
        confidence: +(0.91 + (frameId % 5) * 0.01).toFixed(2),
        x: blueX,
        y: blueY,
        w: 12,
        h: 15,
        color: "#00E08A",
        status: blueStatus,
      },
    ];
  }, [currentState, frameId]);

  const frame: TelemetryFrame = {
    frame_id: frameId,
    timestamp: timeStr,
    current_state: currentState,
    expected_next: expectedNext,
    confidence,
    stability_count: stabilityCounterRef.current,
    state_duration_ms: currentElapsed,
    motion_energy: motionEnergy,
    gates,
    containment,
    fsm,
    is_transition: currentElapsed < 300,
    transition_status: activeAlert
      ? "alert"
      : !isConfidencePass || !isCooldownPass || !isMotionPass
        ? "rejected"
        : "nominal",
    active_alert: activeAlert,
    fps: 59.8,
  };

  const resetSimulation = useCallback(() => {
    stateIndexRef.current = 0;
    setStateIndex(0);
    stateStartRef.current = Date.now();
    lastTransitionRef.current = Date.now() - 3000;
    stabilityCounterRef.current = 15;
    setActiveAlert(null);
    setActiveAnomaly(null);
  }, []);

  const toggleStreamConnect = useCallback(() => {
    const timeStr = formatTimestamp(new Date());
    if (streamStatus === "CONNECTED") {
      setStreamStatus("OFFLINE");
      setSignalQuality((prev) => ({ ...prev, bars: 0, packetLossPct: 100, jitterMs: 0 }));
      setStreamEvents((prev) => [
        {
          id: `se-${Date.now()}`,
          timestamp: timeStr,
          type: "DISCONNECT",
          message: "Stream link severed by operator // Carrier dropped",
        },
        ...prev.slice(0, 3),
      ]);
      avionicsAudio.playRejectedTone();
    } else {
      setStreamStatus("RECONNECTING");
      setStreamEvents((prev) => [
        {
          id: `se-${Date.now()}`,
          timestamp: timeStr,
          type: "RECONNECT",
          message: `Re-negotiating WebRTC SDP with ${streamConfig.ip}:${streamConfig.port}...`,
        },
        ...prev.slice(0, 3),
      ]);
      setTimeout(() => {
        setStreamStatus("CONNECTED");
        setSignalQuality({
          bars: 5,
          maxBars: 6,
          packetLossPct: 0.04,
          jitterMs: 2.1,
          snrDb: 28.4,
        });
        setStreamEvents((prev) => [
          {
            id: `se-${Date.now()}`,
            timestamp: formatTimestamp(new Date()),
            type: "CONNECT",
            message: `Stream locked: ${streamConfig.protocol} @ 14.8 Mbps // Link nominal`,
          },
          ...prev.slice(0, 3),
        ]);
        avionicsAudio.playAcceptedTone();
      }, 700);
    }
  }, [streamStatus, streamConfig]);

  const updateStreamConfig = useCallback((newConfig: Partial<StreamTargetConfig>) => {
    setStreamConfig((prev) => {
      const updated = { ...prev, ...newConfig };
      setStreamEvents((events) => [
        {
          id: `se-${Date.now()}`,
          timestamp: formatTimestamp(new Date()),
          type: "CONFIG",
          message: `Endpoint target updated: ${updated.ip}:${updated.port} (${updated.protocol})`,
        },
        ...events.slice(0, 3),
      ]);
      return updated;
    });
  }, []);

  const toggleRecording = useCallback(() => {
    setRecordingState((prev) => {
      const nextIsRecording = !prev.isRecording;
      if (nextIsRecording) {
        avionicsAudio.playAcceptedTone();
      }
      return {
        ...prev,
        isRecording: nextIsRecording,
      };
    });
  }, []);

  return {
    frame,
    currentState,
    expectedNext,
    confidence,
    confidenceHistory,
    boundingBoxes,
    gates,
    containment,
    fsm,
    logs,
    cyclesCompleted,
    subsystems,
    isPaused,
    setIsPaused,
    isMuted,
    setIsMuted,
    isReducedMotion,
    setIsReducedMotion,
    injectAnomaly,
    resetSimulation,
    activeAlert,
    activeRejection,
    lastAcceptedState,
    streamStatus,
    streamConfig,
    streamEvents,
    signalQuality,
    recordingState,
    networkStats,
    toggleStreamConnect,
    updateStreamConfig,
    toggleRecording,
    thresholds,
    updateThresholds,
    restoreDefaultThresholds,
    cameraSource,
    setCameraSource,
    isReplayingBoot,
    replayBootSequence,
    handleBootComplete,
    isConnected: true,
  };
}
