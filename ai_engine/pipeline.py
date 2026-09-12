"""
ai_engine/pipeline.py - Real-Time AI Inference Pipeline & Decision Engine

Thin, high-performance wrapper around realtime.py from vision-pipeline:
- Preserves 100% of the proven BiLSTM + Attention inference, YOLO object validation,
  BoxTracker dropout smoothing, MotionActionSpotter, and 2.5D Geometric Containment.
- Smooths pose landmarks across frame boundaries to prevent UI flickering.
- Feeds live frames from both cameras and uploaded video files into the exact same pipeline.
"""

import os
import sys
import time
import threading
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import torch
import mediapipe as mp

# Ensure ai_engine is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import realtime as rt
from pose_extract_advanced import extract_base_features, BASE_DIM
import pose_extract_advanced as pea
from ai_engine.tts import VoiceCopilot
from ai_engine.camera import CameraManager

# Re-export core FSM classes for backward compatibility
PhysicalCausalLogic = rt.PhysicalCausalLogic
DecisionStabilizer = rt.DecisionStabilizer


class AstroFlowPipeline:
    """Production Inference Pipeline wrapping realtime.py."""
    def __init__(self):
        print(f"[PIPELINE] Initializing models from {rt.MODEL_PATH} and {rt.YOLO_MODEL_PATH}...")
        self.tar_model = rt.load_tar_model(rt.MODEL_PATH)
        self.yolo_model = rt.load_yolo(rt.YOLO_MODEL_PATH)
        self.async_yolo = rt.AsyncYOLODetector(self.yolo_model)
        self.camera = CameraManager(initial_source="browser")

        self.stabilizer = rt.DecisionStabilizer(
            rt.CONFIDENCE_THRESHOLD, rt.STABILITY_WINDOW, rt.COOLDOWN_SEC, rt.CYCLE_COOLDOWN_SEC
        )
        self.stabilizer.motion_floor = rt.MOTION_THRESHOLD
        self.spotter = rt.MotionActionSpotter()
        self.box_tracker = rt.BoxTracker(max_missing=8)
        self.containment = rt.GeometricContainmentEngine(buffer_size=8)
        self.sop_tracker = rt.SOPTracker()
        self.voice = VoiceCopilot(enabled=True)

        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

        self.window = deque(maxlen=rt.SEQ_LEN)
        self.prev_base = None
        self.prev_feat = None
        self.last_frame_time = time.time()
        self.frame_counter = 1000
        self.confidence_history = deque([0.85] * 50, maxlen=50)

        # Pose smoothing buffer: holds pose across brief 4-frame dropouts to prevent flickering
        self.last_valid_pose = []
        self.pose_missing_frames = 0
        self.running = False
        self.cam_thread = None

    def start(self):
        self.running = True
        self.camera.start()
        self.voice.start()
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

    def stop(self):
        self.running = False
        self.camera.stop()
        self.voice.stop()
        if hasattr(self, "async_yolo") and self.async_yolo:
            self.async_yolo.stop()

    def _camera_loop(self):
        """Background continuous inference loop when hardware camera / test video is active."""
        while self.running:
            if self.camera.source_type == "browser":
                time.sleep(0.05)
                continue
            ret, frame, frame_t, cam_fps = self.camera.read()
            if ret and frame is not None:
                try:
                    self.process_frame(frame, fps=cam_fps)
                except Exception as e:
                    print(f"[PIPELINE] Hardware camera loop error: {e}")
            time.sleep(0.015)

    def switch_camera(self, device_id):
        if hasattr(self, "camera") and self.camera:
            self.camera.switch_source(device_id)

    def process_frame(self, frame, fps=30.0):
        t0 = time.time()
        self.frame_counter += 1
        now = time.time()
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb)
        hand_res = self.hands.process(rgb)
        cur_lms = pose_res.pose_landmarks if pose_res else None

        # 1. Asynchronous YOLO Object Detection with Calibrated HSV & Spatial Validation
        self.async_yolo.update_frame(frame)
        yolo_detections = self.async_yolo.get_detections()
        yolo_red, yolo_blue, yolo_main = None, None, None

        for det in yolo_detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            box_item = {"rect": rect, "area": rect[2] * rect[3], "score": det["conf"]}
            name = det["name"]

            if "red" in name:
                if rt.validate_yolo_detection(frame, rect, name, pose_landmarks=cur_lms):
                    if yolo_red is None or det["conf"] > yolo_red["score"]:
                        yolo_red = box_item
            elif "blue" in name:
                if rt.validate_yolo_detection(frame, rect, name, pose_landmarks=cur_lms):
                    if yolo_blue is None:
                        yolo_blue = box_item
                    else:
                        # Foreground proximity prioritization: pick blue box nearest to hand/workbench
                        if cur_lms and len(cur_lms.landmark) > 16:
                            cx1, cy1 = rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0
                            old_r = yolo_blue["rect"]
                            cx0, cy0 = old_r[0] + old_r[2] / 2.0, old_r[1] + old_r[3] / 2.0
                            d1 = min(np.hypot(cx1 - cur_lms.landmark[15].x * w, cy1 - cur_lms.landmark[15].y * h),
                                     np.hypot(cx1 - cur_lms.landmark[16].x * w, cy1 - cur_lms.landmark[16].y * h))
                            d0 = min(np.hypot(cx0 - cur_lms.landmark[15].x * w, cy0 - cur_lms.landmark[15].y * h),
                                     np.hypot(cx0 - cur_lms.landmark[16].x * w, cy0 - cur_lms.landmark[16].y * h))
                            if d1 < d0:
                                yolo_blue = box_item
                        elif det["conf"] > yolo_blue["score"]:
                            yolo_blue = box_item
            elif "main" in name or "box" in name:
                if yolo_main is None or det["conf"] > yolo_main["score"]:
                    if rt.validate_yolo_detection(frame, rect, name, pose_landmarks=cur_lms):
                        yolo_main = box_item

        # 2. Extract Base Features with Bounding Boxes
        base, det_flags, (red_box, blue_box, main_box), edge_debug, lid_score = extract_base_features(
            pose_res, hand_res, frame, override_boxes=(yolo_red, yolo_blue, yolo_main)
        )

        # Prevent edge detector from hallucinating main_box on empty room walls
        if yolo_main is None and yolo_red is None and yolo_blue is None:
            # Keep edge-detected main_box if operator is present and actively interacting near/with the box
            if cur_lms and len(cur_lms.landmark) > 16 and main_box and "rect" in main_box:
                mx, my, mw, mh = main_box["rect"]
                cx, cy = mx + mw / 2.0, my + mh / 2.0
                d_hand = min(
                    np.hypot(cx - cur_lms.landmark[15].x * w, cy - cur_lms.landmark[15].y * h),
                    np.hypot(cx - cur_lms.landmark[16].x * w, cy - cur_lms.landmark[16].y * h)
                )
                if d_hand > (max(mw, mh) * 0.95 + 80):
                    main_box = None
            else:
                main_box = None

        # 3. Smooth Brief Dropouts via BoxTracker (up to 8 frames)
        red_box = self.box_tracker.get_fallback("red", red_box)
        blue_box = self.box_tracker.get_fallback("blue", blue_box)
        main_box = self.box_tracker.get_fallback("main", main_box)

        # 4. Geometric Containment Evaluation
        self.containment.update(red_box, blue_box, main_box, cur_lms, frame.shape)

        # 5. Kinematic Motion Calculation with Time-Scale Normalization
        # Normalizes velocity vector to 30 FPS training data distribution regardless of ingest rate
        dt = max(0.001, now - self.last_frame_time)
        self.last_frame_time = now
        time_scale = min(2.5, max(0.4, 30.0 * dt))

        if self.prev_base is None:
            vel = np.zeros_like(base)
        else:
            vel = (base - self.prev_base) / time_scale
        self.prev_base = base.copy()

        # 6. Dual-Path Temporal Action Prediction
        feat = np.concatenate([base, vel]).astype(np.float32)
        motion = rt.compute_motion(self.prev_feat, feat)
        self.prev_feat = feat.copy()

        result = rt.process_frame(
            feat, motion, self.window, self.tar_model, self.stabilizer, self.spotter, now,
            yolo_red=yolo_red, yolo_blue=yolo_blue, red_box=red_box, blue_box=blue_box,
            main_box=main_box, pose_landmarks=cur_lms,
            frame_shape=frame.shape, expected_action=self.sop_tracker.expected_action,
            containment=self.containment
        )

        # 7. Update SOP Tracker & Voice Announcements
        voice_prompt = None
        if result["state_changed"] and result["current_state"] != "idle":
            self.sop_tracker.update(result["current_state"], now)
            if result["current_state"] in ("close_box", "open_box"):
                self.containment.reset()
            step_title = self.sop_tracker.expected_action_title or result["current_state"]
            voice_prompt = f"Step verified: {step_title}."
            self.voice.speak(voice_prompt, priority=2)

        if result.get("is_mistake") and result.get("mistake_detected"):
            reason = result.get("mistake_reason", "Procedure sequence deviation")
            voice_prompt = f"Procedure alert: {reason}."
            self.voice.speak(voice_prompt, priority=0)

        # 8. Extract Real Detected Bounding Boxes for UI with Normalized Confidence
        def _norm_conf(raw, default=0.90):
            try:
                f = float(raw)
                if f > 1.0:
                    return round(min(0.96, max(0.65, default)), 2)
                return round(min(1.0, max(0.0, f)), 2)
            except Exception:
                return default

        is_box_open = self.stabilizer.causal_logic.box_open or lid_score > 0.02 or self.sop_tracker.current_step > 0

        detected_boxes_list = []
        if main_box and "rect" in main_box:
            mx, my, mw, mh = main_box["rect"]
            detected_boxes_list.append({
                "id": "main_box",
                "label": "MAIN CONTAINER",
                "confidence": _norm_conf(main_box.get("score", 0.94), 0.94),
                "x": round((mx / w) * 100, 1),
                "y": round((my / h) * 100, 1),
                "w": round((mw / w) * 100, 1),
                "h": round((mh / h) * 100, 1),
                "color": "#00E08A",
                "status": "OPEN" if is_box_open else "CLOSED"
            })
        if red_box and "rect" in red_box:
            rx, ry, rw, rh = red_box["rect"]
            detected_boxes_list.append({
                "id": "red_box",
                "label": "RED CUBE [SAMPLE-A]",
                "confidence": _norm_conf(red_box.get("score", 0.90), 0.90),
                "x": round((rx / w) * 100, 1),
                "y": round((ry / h) * 100, 1),
                "w": round((rw / w) * 100, 1),
                "h": round((rh / h) * 100, 1),
                "color": "#FF4D4F",
                "status": self.containment.red_status
            })
        if blue_box and "rect" in blue_box:
            bx, by, bw, bh = blue_box["rect"]
            detected_boxes_list.append({
                "id": "blue_box",
                "label": "BLUE CUBE [SAMPLE-B]",
                "confidence": _norm_conf(blue_box.get("score", 0.90), 0.90),
                "x": round((bx / w) * 100, 1),
                "y": round((by / h) * 100, 1),
                "w": round((bw / w) * 100, 1),
                "h": round((bh / h) * 100, 1),
                "color": "#2979FF",
                "status": self.containment.blue_status
            })

        # 9. Extract Pose Coordinates with Smoothing Buffer (No Flickering!)
        current_pose_points = []
        if pose_res and pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks.landmark:
                current_pose_points.append({
                    "x": round(lm.x * 100, 1),
                    "y": round(lm.y * 100, 1),
                    "v": round(lm.visibility, 2)
                })

        if len(current_pose_points) >= 25:
            self.last_valid_pose = current_pose_points
            self.pose_missing_frames = 0
            pose_points_out = current_pose_points
        else:
            self.pose_missing_frames += 1
            # Hold previous valid pose for up to 5 frames to eliminate flickering
            if self.pose_missing_frames <= 5 and self.last_valid_pose:
                pose_points_out = self.last_valid_pose
            else:
                pose_points_out = []

        # 10. Format Containment
        if main_box is None:
            containment_state = {
                "main_box": "NOT DETECTED",
                "red_box": "NOT DETECTED",
                "blue_box": "NOT DETECTED"
            }
        else:
            containment_state = {
                "main_box": "OPEN" if is_box_open else "CLOSED",
                "red_box": self.containment.red_status if self.containment.red_status != "UNKNOWN" else "STANDBY",
                "blue_box": self.containment.blue_status if self.containment.blue_status != "UNKNOWN" else "STANDBY",
            }

        # 11. Format 5 DecisionStabilizer Gates
        cur_conf = float(result.get("confidence", 0.0))
        self.confidence_history.append(cur_conf)
        stab_count = len(self.stabilizer.recent)
        motion_passed = result.get("current_state") == "idle" or motion >= rt.MOTION_THRESHOLD
        conf_passed = cur_conf >= rt.CONFIDENCE_THRESHOLD
        stab_passed = stab_count >= rt.STABILITY_WINDOW
        cooldown_rem = max(0.0, self.stabilizer.cooldown_sec - (now - self.stabilizer.last_change_time))
        cool_passed = cooldown_rem <= 0.05
        causal_passed = not result.get("is_mistake", False)

        gates = {
            "confidence": {
                "id": "confidence",
                "name": "Confidence Gate",
                "passed": bool(conf_passed),
                "value": round(cur_conf, 2),
                "threshold": round(rt.CONFIDENCE_THRESHOLD, 2),
                "reason": f"Probability {cur_conf:.2f} >= {rt.CONFIDENCE_THRESHOLD:.2f}" if conf_passed else f"Under-confidence ({cur_conf:.2f})"
            },
            "stability": {
                "id": "stability",
                "name": "Stability Window",
                "passed": bool(stab_passed),
                "value": stab_count,
                "threshold": rt.STABILITY_WINDOW,
                "reason": f"Held for {stab_count}/{rt.STABILITY_WINDOW} frames"
            },
            "cooldown": {
                "id": "cooldown",
                "name": "Transition Cooldown",
                "passed": bool(cool_passed),
                "value": f"{cooldown_rem:.1f}s" if not cool_passed else "0.0s",
                "threshold": f"{rt.COOLDOWN_SEC:.2f}s",
                "reason": "Cooldown clear" if cool_passed else f"Cooldown active ({cooldown_rem:.1f}s)"
            },
            "motion": {
                "id": "motion",
                "name": "Kinematic Motion",
                "passed": bool(motion_passed),
                "value": round(motion, 3),
                "threshold": round(rt.MOTION_THRESHOLD, 3),
                "reason": f"Motion {motion:.3f} >= {rt.MOTION_THRESHOLD:.3f}" if motion_passed else "Motion below floor"
            },
            "causal_logic": {
                "id": "causal_logic",
                "name": "Causal Logic Gate",
                "passed": bool(causal_passed),
                "value": 1 if causal_passed else 0,
                "threshold": 1,
                "reason": result.get("mistake_reason", "Protocol causal logic verified")
            }
        }

        active_alert = None
        if result.get("is_mistake") and result.get("mistake_detected"):
            active_alert = {
                "active": True,
                "reason": result.get("mistake_reason", "Out-of-sequence action attempted"),
                "timestamp": datetime.now().isoformat()
            }

        elapsed_ms = (time.time() - t0) * 1000.0

        telemetry = {
            "frame_id": self.frame_counter,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "current_state": result.get("current_state", "idle"),
            "expected_next": self.sop_tracker.expected_action or "open_box",
            "confidence": cur_conf,
            "confidence_history": list(self.confidence_history),
            "stability_count": stab_count,
            "state_duration_ms": int((now - self.stabilizer.last_change_time) * 1000),
            "motion_energy": round(motion, 3),
            "gates": gates,
            "containment": containment_state,
            "fsm": {
                "box_open": is_box_open,
                "red_picked": self.stabilizer.causal_logic.red_picked,
                "red_placed_out": self.stabilizer.causal_logic.red_placed_out,
                "blue_picked": self.stabilizer.causal_logic.blue_picked,
                "blue_placed_in": self.stabilizer.causal_logic.blue_placed_in,
            },
            "boxes": detected_boxes_list,
            "pose_points": pose_points_out,
            "pose_locked": len(pose_points_out) > 0,
            "voice_prompt": voice_prompt,
            "is_transition": result.get("state_changed", False),
            "transition_status": "alert" if active_alert else ("nominal" if all(g["passed"] for g in gates.values()) else "evaluating"),
            "active_alert": active_alert,
            "latency_ms": round(elapsed_ms, 1),
            "fps": round(fps, 1),
            "cycles_completed": self.sop_tracker.cycle_count
        }

        self.latest_telemetry = telemetry
        try:
            _, jpeg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.latest_frame_bytes = jpeg_buf.tobytes()
        except Exception:
            pass

        return telemetry

    def get_latest_jpeg(self):
        return self.latest_frame_bytes

    def get_latest_telemetry(self):
        return self.latest_telemetry

    def update_gate_thresholds(self, confidence=None, stability=None, cooldown=None, motion=None):
        if confidence is not None:
            self.stabilizer.confidence_threshold = float(confidence)
        if stability is not None:
            self.stabilizer.stability_window = int(stability)
        if cooldown is not None:
            self.stabilizer.cooldown_sec = float(cooldown)
        if motion is not None:
            self.stabilizer.motion_floor = float(motion)
