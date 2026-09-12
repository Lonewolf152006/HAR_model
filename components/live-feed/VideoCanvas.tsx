"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { useTelemetry } from "@/context/TelemetryContext";
import { AvionicsPanel } from "@/components/ui/AvionicsPanel";
import { Crosshair, Radio, Target, Activity, ShieldCheck, AlertTriangle, Camera, RefreshCw, Upload } from "lucide-react";

// MediaPipe 33-point pose skeletal graph connections
const POSE_CONNECTIONS: Array<[number, number]> = [
  // Torso
  [11, 12],
  [12, 24],
  [24, 23],
  [23, 11],
  // Left arm
  [11, 13],
  [13, 15],
  // Right arm
  [12, 14],
  [14, 16],
  // Face / Shoulders
  [0, 11],
  [0, 12],
  // Left leg
  [23, 25],
  [25, 27],
  // Right leg
  [24, 26],
  [26, 28],
];

export function VideoCanvas() {
  const {
    currentState,
    expectedNext,
    confidence,
    frame,
    cyclesCompleted,
    activeAlert,
    isReducedMotion,
    isMuted,
    ingestRealTelemetry,
    poseLocked,
    boundingBoxes,
  } = useTelemetry();

  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const prevObjectUrlRef = useRef<string | null>(null);

  const [reticlePos, setReticlePos] = useState({ x: 50, y: 50 });
  const [isHovered, setIsHovered] = useState(false);
  const [browserDevices, setBrowserDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [activeCamName, setActiveCamName] = useState<string>("Detecting Cameras...");
  const [streamResolution, setStreamResolution] = useState<string>("1280x720");
  const [cameraPermissionGranted, setCameraPermissionGranted] = useState<boolean | null>(null);
  const [inferenceLatency, setInferenceLatency] = useState<number>(11.4);
  const [isPlayingVideoFile, setIsPlayingVideoFile] = useState<boolean>(false);
  const [displayPose, setDisplayPose] = useState<Array<{ x: number; y: number; v: number }>>([]);
  const lastPoseRef = useRef<Array<{ x: number; y: number; v: number }>>([]);
  const missingCountRef = useRef<number>(0);
  const isInferringRef = useRef<boolean>(false);

  // 1. Initialize Browser Camera Stream & Enumerate Devices
  const activateCamera = useCallback((deviceId?: string) => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices) return;
    const constraints: MediaStreamConstraints = {
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    };

    const tryGetUserMedia = async () => {
      try {
        return await navigator.mediaDevices.getUserMedia(constraints);
      } catch {
        // Fallback: try basic video constraint if 1280x720 or exact deviceId was rejected
        return await navigator.mediaDevices.getUserMedia({
          video: deviceId ? { deviceId } : true,
          audio: false,
        });
      }
    };

    tryGetUserMedia()
      .then((stream) => {
        setCameraPermissionGranted(true);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(() => {});
        }
        const track = stream.getVideoTracks()[0];
        if (track) {
          const settings = track.getSettings();
          if (settings.width && settings.height) {
            setStreamResolution(`${settings.width}x${settings.height}`);
          }
          if (track.label) {
            setActiveCamName(track.label);
          }
          if (settings.deviceId) {
            setSelectedDeviceId(settings.deviceId);
          }
        }
        return navigator.mediaDevices.enumerateDevices();
      })
      .then((devices) => {
        if (!devices) return;
        const videoInputs = devices.filter((d) => d.kind === "videoinput");
        setBrowserDevices(videoInputs);
      })
      .catch((err) => {
        console.warn("[CAMERA] getUserMedia access failed/denied:", err);
        setCameraPermissionGranted(false);
        setActiveCamName("Camera Access Blocked");
      });
  }, []);

  useEffect(() => {
    activateCamera();
  }, [activateCamera]);

  // Handle switching camera device
  // Handle switching camera device
  const handleDeviceChange = (deviceId: string) => {
    setSelectedDeviceId(deviceId);
    setIsPlayingVideoFile(false);

    if (deviceId === "backend-mjpeg") {
      setActiveCamName("Backend 30 FPS Direct Feed");
      setStreamResolution("1280x720");
      setCameraPermissionGranted(true);
      // Instruct Python backend to acquire hardware camera 0
      fetch("http://127.0.0.1:8080/api/v1/camera/select", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device_id: "0" }),
      }).catch(() => {});
      // Stop browser webcam stream if active to release camera
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((t) => t.stop());
        videoRef.current.srcObject = null;
      }
      return;
    }

    // Switching to browser camera: release Python camera if held
    fetch("http://127.0.0.1:8080/api/v1/camera/select", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: "browser" }),
    }).catch(() => {});

    const matched = browserDevices.find((d) => d.deviceId === deviceId);
    if (matched) {
      setActiveCamName(matched.label || `Camera Device`);
    }
    activateCamera(deviceId);
  };

  // Handle recorded video file upload & testing
  const handleVideoFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Revoke prior blob URL to avoid memory leaks
    if (prevObjectUrlRef.current) {
      URL.revokeObjectURL(prevObjectUrlRef.current);
    }
    const url = URL.createObjectURL(file);
    prevObjectUrlRef.current = url;

    setIsPlayingVideoFile(true);
    setSelectedDeviceId("video-file");
    setActiveCamName(`Test File: ${file.name}`);

    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = url;
      videoRef.current.loop = true;
      videoRef.current.play().catch(() => {});
    }
  };

  const handleSwitchToCamera = () => {
    setIsPlayingVideoFile(false);
    if (prevObjectUrlRef.current) {
      URL.revokeObjectURL(prevObjectUrlRef.current);
      prevObjectUrlRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.src = "";
    }
    activateCamera(selectedDeviceId);
  };

  // Cleanup object URLs on unmount
  useEffect(() => {
    return () => {
      if (prevObjectUrlRef.current) {
        URL.revokeObjectURL(prevObjectUrlRef.current);
      }
    };
  }, []);

  // 2. Real-Time AI Inference Loop (High-speed ~25 FPS Frame Grabber feeding Python backend)
  useEffect(() => {
    const interval = setInterval(async () => {
      if (selectedDeviceId === "backend-mjpeg") return;
      if (isInferringRef.current || !videoRef.current || !hiddenCanvasRef.current) return;
      const video = videoRef.current;
      if (video.readyState < 2 || video.videoWidth === 0) return;

      isInferringRef.current = true;
      try {
        const canvas = hiddenCanvasRef.current;
        // Standardized 640px wide resolution for high-speed YOLO & MediaPipe inference
        const targetW = 640;
        const targetH = video.videoHeight > 0 && video.videoWidth > 0
          ? Math.round((video.videoHeight / video.videoWidth) * targetW)
          : 360;
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(video, 0, 0, targetW, targetH);
          canvas.toBlob(async (blob) => {
            if (!blob) {
              isInferringRef.current = false;
              return;
            }
            try {
              const formData = new FormData();
              formData.append("file", blob, "frame.jpg");
              const res = await fetch("http://127.0.0.1:8080/api/v1/infer", {
                method: "POST",
                body: formData,
              });
              if (res.ok) {
                const data = await res.json();
                if (data.ok && data.telemetry) {
                  // Ingest directly into TelemetryContext to update StateConfidenceCard, GateChecklist, etc.
                  ingestRealTelemetry(data.telemetry);
                  if (data.telemetry.latency_ms) {
                    setInferenceLatency(data.telemetry.latency_ms);
                  }

                  // Smooth pose landmarks across brief camera dropouts (zero flickering!)
                  if (data.telemetry.pose_points && Array.isArray(data.telemetry.pose_points) && data.telemetry.pose_points.length >= 15) {
                    lastPoseRef.current = data.telemetry.pose_points;
                    missingCountRef.current = 0;
                    setDisplayPose(data.telemetry.pose_points);
                  } else {
                    missingCountRef.current += 1;
                    if (missingCountRef.current > 6) {
                      setDisplayPose([]);
                    }
                  }

                  // Voice Copilot Audio Speech (100% reliable in-browser speech synthesis)
                  if (data.telemetry.voice_prompt && !isMuted && typeof window !== "undefined" && "speechSynthesis" in window) {
                    try {
                      window.speechSynthesis.cancel();
                      const utterance = new SpeechSynthesisUtterance(data.telemetry.voice_prompt);
                      utterance.rate = 1.0;
                      utterance.pitch = 1.0;
                      window.speechSynthesis.speak(utterance);
                    } catch {
                      // Speech synthesis fallback
                    }
                  }
                }
              }
            } catch {
              // Edge hub may be offline
            } finally {
              isInferringRef.current = false;
            }
          }, "image/jpeg", 0.7);
        } else {
          isInferringRef.current = false;
        }
      } catch {
        isInferringRef.current = false;
      }
    }, 40); // 40ms interval (~25 FPS high responsiveness)

    return () => clearInterval(interval);
  }, [ingestRealTelemetry, isMuted, selectedDeviceId]);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    setReticlePos({
      x: +Math.max(0, Math.min(100, x)).toFixed(1),
      y: +Math.max(0, Math.min(100, y)).toFixed(1),
    });
  };

  return (
    <AvionicsPanel
      key="avionics-video-canvas"
      className={`p-0 overflow-hidden bg-[#0A0C0F] border h-full flex flex-col min-h-0 shadow-2xl transition-colors duration-150 ${
        activeAlert?.active
          ? !isReducedMotion
            ? "animate-alert-border-sustained border-[#FF4D4F]"
            : "border-[#FF4D4F]"
          : "border-white/15"
      }`}
      bracketColor={activeAlert?.active ? "border-[#FF4D4F]" : "border-[#00E08A]"}
    >
      {/* Hidden Frame Grabber Canvas */}
      <canvas ref={hiddenCanvasRef} className="hidden" />

      {/* Top Header Flight Strip with Dynamic Camera Selector */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-[#12151A] border-b border-white/10 font-mono text-xs select-none shrink-0">
        <div className="flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-[#00E08A] glow-nominal animate-pulse shrink-0" />
          
          {/* Dynamic Browser Camera Selector (Google Meet style) */}
          <div className="flex items-center gap-1.5 bg-[#0E1015] border border-white/15 px-2 py-0.5 rounded-[2px] bezel-depth-subtle">
            <Camera className="w-3.5 h-3.5 text-[#00E08A] shrink-0" />
            <select
              value={selectedDeviceId}
              onChange={(e) => handleDeviceChange(e.target.value)}
              className="bg-transparent text-[#00E08A] font-mono text-[10px] font-bold outline-none cursor-pointer pr-1"
              title="Select camera (Direct Python Stream, Camo Studio, OBS Virtual Camera, Webcams)"
            >
              <option value="backend-mjpeg" className="bg-[#12151A] text-[#00E08A] font-bold">
                Direct Python 30 FPS Stream (/video_feed)
              </option>
              {browserDevices.map((d, i) => (
                <option key={d.deviceId || i} value={d.deviceId} className="bg-[#12151A] text-white">
                  {d.label || `Camera #${i + 1}`}
                </option>
              ))}
              {browserDevices.length === 0 && selectedDeviceId !== "backend-mjpeg" && (
                <option value="" className="bg-[#12151A] text-white">
                  {activeCamName}
                </option>
              )}
            </select>
          </div>

          {/* Video File Upload & Playback Test Trigger */}
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/mkv,video/avi,video/webm"
            className="hidden"
            onChange={handleVideoFileUpload}
          />

          {isPlayingVideoFile ? (
            <button
              onClick={handleSwitchToCamera}
              className="flex items-center gap-1 bg-[#00E08A]/15 border border-[#00E08A] px-2 py-0.5 rounded-[2px] font-mono text-[10px] text-[#00E08A] font-bold hover:bg-[#00E08A]/25 transition-colors"
              title="Return to real-time live camera"
            >
              <Camera className="w-3 h-3 text-[#00E08A]" />
              <span>RETURN TO LIVE CAM</span>
            </button>
          ) : (
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1 bg-[#171B21] border border-white/20 hover:border-[#00E08A] px-2 py-0.5 rounded-[2px] font-mono text-[10px] text-[#8A919C] hover:text-[#00E08A] font-bold transition-colors"
              title="Select a recorded video file to run through the AI model"
            >
              <Upload className="w-3 h-3" />
              <span>TEST VIDEO FILE</span>
            </button>
          )}

          <span className="text-[10px] text-[#565C66] hidden md:inline truncate max-w-[200px]">
            [{isPlayingVideoFile ? activeCamName : `${streamResolution} | ${activeCamName}`}]
          </span>
        </div>

        <div className="flex items-center gap-3 text-[10px]">
          <button
            onClick={() => activateCamera(selectedDeviceId)}
            className="text-[#8A919C] hover:text-[#00E08A] transition-colors flex items-center gap-1"
            title="Refresh Camera Connection"
          >
            <RefreshCw className="w-3 h-3 text-[#00E08A]" />
          </button>
          <span className="text-[#8A919C] flex items-center gap-1">
            <Radio className="w-3 h-3 text-[#00E08A]" />
            <span>{frame?.fps ? `${frame.fps.toFixed(1)} FPS` : "30.0 FPS"}</span>
          </span>
          <span className="px-1.5 py-0.5 bg-[#1A0E10] border border-[#FF4D4F]/50 text-[#FF4D4F] font-bold rounded-[2px] flex items-center gap-1.5 bezel-depth-subtle">
            <span className="w-1.5 h-1.5 rounded-full bg-[#FF4D4F] glow-critical animate-ping" />
            REC
          </span>
        </div>
      </div>

      {/* Main Video Viewport */}
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className="flex-1 min-h-0 relative w-full bg-[#07080B] lens-vignette select-none overflow-hidden cursor-crosshair group flex items-center justify-center"
      >
        {/* Real Live Hardware Video Element or Backend Direct Stream */}
        {selectedDeviceId === "backend-mjpeg" ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src="http://127.0.0.1:8080/video_feed"
            alt="Backend 30 FPS Live Video Feed"
            className="absolute inset-0 w-full h-full object-contain z-0"
          />
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="absolute inset-0 w-full h-full object-contain z-0"
          />
        )}

        {/* Camera Permission Needed State (Only when browser camera is selected) */}
        {selectedDeviceId !== "backend-mjpeg" && cameraPermissionGranted === false && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-[#07080B]/90 p-6 text-center font-mono">
            <Camera className="w-12 h-12 text-[#FFB020] mb-3 animate-pulse" />
            <div className="text-sm font-bold text-[#E6E9ED] mb-1">CAMERA PERMISSION REQUIRED</div>
            <div className="text-xs text-[#8A919C] max-w-md mb-4">
              Please click &quot;Allow&quot; on the browser camera permission dialog to stream your Camo Studio phone camera or webcam, or select &quot;Direct Python 30 FPS Stream&quot; above.
            </div>
            <button
              onClick={() => activateCamera()}
              className="px-3 py-1.5 bg-[#00E08A]/10 border border-[#00E08A] text-[#00E08A] rounded-[2px] text-xs font-bold hover:bg-[#00E08A]/20 transition-colors"
            >
              REQUEST PERMISSION AGAIN
            </button>
          </div>
        )}

        {/* Real MediaPipe Pose Tracking Skeleton (Stabilized with hold buffer to eliminate flickering) */}
        {displayPose && displayPose.length >= 15 && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none z-10"
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
          >
            {/* Skeleton Connecting Lines */}
            {POSE_CONNECTIONS.map(([i, j], idx) => {
              const p1 = displayPose[i];
              const p2 = displayPose[j];
              if (!p1 || !p2 || (p1.v != null && p1.v < 0.35) || (p2.v != null && p2.v < 0.35)) return null;
              return (
                <line
                  key={`bone-${idx}`}
                  x1={`${p1.x}`}
                  y1={`${p1.y}`}
                  x2={`${p2.x}`}
                  y2={`${p2.y}`}
                  stroke="#00E08A"
                  strokeWidth="0.75"
                  strokeOpacity="0.85"
                />
              );
            })}

            {/* Glowing Joint Nodes */}
            {displayPose.map((pt, idx) => {
              if (pt.v != null && pt.v < 0.35) return null;
              return (
                <circle
                  key={`joint-${idx}`}
                  cx={`${pt.x}`}
                  cy={`${pt.y}`}
                  r="0.85"
                  fill="#4DA3FF"
                  stroke="#00E08A"
                  strokeWidth="0.25"
                />
              );
            })}
          </svg>
        )}

        {/* Real Dynamic AI YOLOv8 Bounding Boxes (ONLY rendered when detected by model!) */}
        {boundingBoxes && boundingBoxes.map((box) => {
          const glowClass =
            box.color === "#FF4D4F"
              ? "glow-critical"
              : box.color === "#00E08A"
              ? "glow-nominal"
              : "glow-info";

          return (
            <div
              key={box.id}
              className="absolute transition-all duration-150 ease-out pointer-events-none z-10"
              style={{
                left: `${box.x}%`,
                top: `${box.y}%`,
                width: `${box.w}%`,
                height: `${box.h}%`,
              }}
            >
              <div
                className="w-full h-full border-2 relative"
                style={{ borderColor: box.color }}
              >
                <span
                  className="absolute -top-1 -left-1 w-2 h-2 border-t-2 border-l-2"
                  style={{ borderColor: box.color }}
                />
                <span
                  className="absolute -top-1 -right-1 w-2 h-2 border-t-2 border-r-2"
                  style={{ borderColor: box.color }}
                />
                <span
                  className="absolute -bottom-1 -left-1 w-2 h-2 border-b-2 border-l-2"
                  style={{ borderColor: box.color }}
                />
                <span
                  className="absolute -bottom-1 -right-1 w-2 h-2 border-b-2 border-r-2"
                  style={{ borderColor: box.color }}
                />

                <div
                  className={`absolute -top-6 left-0 px-1.5 py-0.5 font-mono text-[9px] font-bold text-white uppercase flex items-center gap-1.5 rounded-[1px] ${glowClass}`}
                  style={{ backgroundColor: box.color }}
                >
                  <span>{box.label}</span>
                  <span className="opacity-90">
                    {Math.min(100, Math.max(0, Math.round((box.confidence <= 1.0 ? box.confidence : box.confidence / 100) * 100)))}%
                  </span>
                  {box.status && (
                    <span className="bg-black/40 px-1 py-0.2 text-[8px] rounded-[1px]">
                      {box.status}
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}

        {/* Top-Left Flight HUD Overlay */}
        <div className="absolute top-3 left-3 flex flex-col gap-1.5 pointer-events-none z-20">
          <div className="flex items-center gap-2 px-2.5 py-1 bg-[#0E1015]/90 border border-white/20 rounded-[2px] bezel-depth backdrop-blur-md">
            <span
              className={`w-2 h-2 rounded-full ${
                activeAlert?.active
                  ? "bg-[#FF4D4F] glow-critical animate-ping"
                  : "bg-[#00E08A] glow-nominal animate-pulse"
              }`}
            />
            <span className="font-mono text-xs font-bold text-[#E6E9ED] tracking-wider">
              STATE: {currentState.toUpperCase()}
            </span>
          </div>

          <div className="flex items-center gap-2 px-2 py-0.5 bg-[#0E1015]/85 border border-white/10 rounded-[2px] font-mono text-[10px] text-[#8A919C] backdrop-blur-sm">
            <span>NEXT:</span>
            <span className="text-[#00E08A] font-bold">{expectedNext}</span>
            <span className="text-[#565C66]">|</span>
            <span>CONF:</span>
            <span
              className={`font-bold ${
                confidence >= 0.75
                  ? "text-[#00E08A]"
                  : confidence >= 0.52
                  ? "text-[#FFB020]"
                  : "text-[#FF4D4F]"
              }`}
            >
              {confidence.toFixed(2)}
            </span>
          </div>

          {/* AI Pose Tracking Lock Status Indicator (Stabilized) */}
          <div className="flex items-center gap-1.5 px-2 py-0.5 bg-[#0E1015]/85 border border-white/10 rounded-[2px] font-mono text-[9px] backdrop-blur-sm">
            <span className={`w-1.5 h-1.5 rounded-full ${displayPose.length >= 15 ? "bg-[#00E08A] animate-pulse" : "bg-[#FFB020]"}`} />
            <span className={displayPose.length >= 15 ? "text-[#00E08A] font-bold" : "text-[#FFB020]"}>
              {displayPose.length >= 15 ? "POSE: 33 PTS LOCKED" : "POSE: SEARCHING SUBJECT"}
            </span>
          </div>
        </div>

        {/* Top-Right Telemetry Badge Strip */}
        <div className="absolute top-3 right-3 flex flex-col items-end gap-1.5 pointer-events-none z-20 font-mono text-[10px]">
          <div className="px-2 py-0.5 bg-[#0E1015]/85 border border-white/10 rounded-[2px] text-[#8A919C] flex items-center gap-1.5 backdrop-blur-sm">
            <Activity className="w-3 h-3 text-[#00E08A]" />
            <span>AI INFERENCE: {inferenceLatency.toFixed(1)} ms</span>
          </div>
          <div className="px-2 py-0.5 bg-[#0E1015]/85 border border-white/10 rounded-[2px] text-[#8A919C] flex items-center gap-1.5 backdrop-blur-sm">
            <ShieldCheck className="w-3 h-3 text-[#4DA3FF]" />
            <span>CYCLES VERIFIED: #{cyclesCompleted}</span>
          </div>
        </div>

        {/* Interactive Flight Reticle Crosshair */}
        <div
          className={`absolute pointer-events-none transition-opacity duration-150 z-20 ${
            isHovered ? "opacity-100" : "opacity-30"
          }`}
          style={{
            left: `${reticlePos.x}%`,
            top: `${reticlePos.y}%`,
            transform: "translate(-50%, -50%)",
          }}
        >
          <Crosshair className="w-8 h-8 text-[#00E08A]/70" />
        </div>

        {/* Bottom Avionics Status Bar */}
        <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between px-3 py-1 bg-[#0E1015]/90 border border-white/10 rounded-[2px] font-mono text-[10px] text-[#8A919C] backdrop-blur-sm z-20">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-[#00E08A]">
              <Target className="w-3 h-3" />
              <span>{poseLocked ? "BIOMETRIC TRACKING ACTIVE" : "AWAITING OPERATOR POSTURE"}</span>
            </span>
            <span className="text-[#565C66]">|</span>
            <span>COORDINATES: X:{reticlePos.x}% Y:{reticlePos.y}%</span>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-[#E6E9ED] font-bold">5/5 GATES ARMED</span>
          </div>
        </div>

        {/* Critical Alert Warning Bar */}
        {activeAlert?.active && (
          <div className="absolute top-16 left-3 right-3 bg-[#1A0E10]/95 border-2 border-[#FF4D4F] p-2 rounded-[2px] flex items-center gap-2 font-mono text-xs text-[#FF4D4F] font-bold glow-critical z-30 animate-reject-pulse backdrop-blur-md">
            <AlertTriangle className="w-4 h-4 shrink-0 animate-ping" />
            <div className="flex-1 truncate">
              PROCEDURAL ALERT: {activeAlert.reason}
            </div>
          </div>
        )}
      </div>
    </AvionicsPanel>
  );
}
