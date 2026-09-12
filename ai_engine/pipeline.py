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


class AstroFlowPipeline:
    """Production Inference Pipeline wrapping realtime.py."""
    def __init__(self):
        print(f"[PIPELINE] Initializing models from {rt.MODEL_PATH} and {rt.YOLO_MODEL_PATH}...")
        self.tar_model = rt.load_tar_model(rt.MODEL_PATH)
        self.yolo_model = rt.load_yolo(rt.YOLO_MODEL_PATH)

        self.stabilizer = rt.DecisionStabilizer(
            rt.CONFIDENCE_THRESHOLD, rt.STABILITY_WINDOW, rt.COOLDOWN_SEC, rt.CYCLE_COOLDOWN_SEC
        )
        self.spotter = rt.MotionActionSpotter()
        self.box_tracker = rt.BoxTracker(max_missing=8)
        self.containment = rt.GeometricContainmentEngine(buffer_size=8)
        self.sop_tracker = rt.SOPTracker()
        self.voice = VoiceCopilot(enabled=True)

        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

        self.window = deque(maxlen=rt.SEQ_LEN)
        self.prev_base = None
        self.frame_counter = 1000
        self.confidence_history = deque([0.85] * 50, maxlen=50)

        # Pose smoothing buffer: holds pose across brief 4-frame dropouts to prevent flickering
        self.last_valid_pose = []
        self.pose_missing_frames = 0

    def start(self):
        self.voice.start()

    def stop(self):
        self.voice.stop()

    def process_frame(self, frame, fps=30.0):
        t0 = time.time()
        self.frame_counter += 1
        now = time.time()
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = self.pose.process(rgb)
        hand_res = self.hands.process(rgb)

        # 1. YOLO Object Detection (Direct proven implementation from test_video_tar.py)
        yolo_detections = pea.run_yolo(self.yolo_model, frame, conf_threshold=0.20)
        yolo_red, yolo_blue, yolo_main = None, None, None
        for det in yolo_detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            box_item = {"rect": rect, "area": rect[2] * rect[3], "score": det["conf"]}
            name = det["name"]
            if "red" in name and (yolo_red is None or det["conf"] > yolo_red["score"]):
                yolo_red = box_item
            elif "blue" in name and (yolo_blue is None or det["conf"] > yolo_blue["score"]):
                yolo_blue = box_item
            elif "main" in name or "box" in name:
                if yolo_main is None or det["conf"] > yolo_main["score"]:
                    yolo_main = box_item

        # 2. Extract Base Features with Bounding Boxes
        base, det_flags, (red_box, blue_box, main_box), edge_debug, lid_score = extract_base_features(
            pose_res, hand_res, frame, override_boxes=(yolo_red, yolo_blue, yolo_main)
        )

        # Prevent edge detector from hallucinating main_box on empty room desks
        if yolo_main is None and yolo_red is None and yolo_blue is None:
            main_box = None

        # 3. Smooth Brief Dropouts via BoxTracker (up to 8 frames)
        red_box = self.box_tracker.get_fallback("red", red_box)
        blue_box = self.box_tracker.get_fallback("blue", blue_box)
        main_box = self.box_tracker.get_fallback("main", main_box)

        # 4. Geometric Containment Evaluation
        self.containment.update(red_box, blue_box, main_box, pose_res.pose_landmarks if pose_res else None, frame.shape)

        # 5. Kinematic Motion Calculation
        if self.prev_base is None:
            motion = 0.0
            vel = np.zeros_like(base)
        else:
            motion = float(np.mean(np.abs(base - self.prev_base)))
            vel = base - self.prev_base
        self.prev_base = base.copy()

        # 6. Dual-Path Temporal Action Prediction
        feat = np.concatenate([base, vel]).astype(np.float32)
        result = rt.process_frame(
            feat, motion, self.window, self.tar_model, self.stabilizer, self.spotter, now,
            yolo_red=yolo_red, yolo_blue=yolo_blue, red_box=red_box, blue_box=blue_box,
            main_box=main_box, pose_landmarks=pose_res.pose_landmarks if pose_res else None,
            frame_shape=frame.shape, expected_action=self.sop_tracker.expected_action,
            containment=self.containment
        )

        # 7. Update SOP Tracker & Voice Announcements
        voice_prompt = None
        if result["state_changed"] and result["current_state"] != "idle":
            self.sop_tracker.update(result["current_state"], now)
            step_title = self.sop_tracker.expected_action_title or result["current_state"]
            voice_prompt = f"Step verified: {step_title}."
            self.voice.speak(voice_prompt, priority=2)

        if result.get("is_mistake") and result.get("mistake_detected"):
            reason = result.get("mistake_reason", "Procedure sequence deviation")
            voice_prompt = f"Procedure alert: {reason}."
            self.voice.speak(voice_prompt, priority=0)

        # 8. Extract Real Detected Bounding Boxes for UI
        detected_boxes_list = []
        if main_box and "rect" in main_box:
            mx, my, mw, mh = main_box["rect"]
            detected_boxes_list.append({
                "id": "main_box",
                "label": "MAIN CONTAINER",
                "confidence": round(main_box.get("score", 0.94), 2),
                "x": round((mx / w) * 100, 1),
                "y": round((my / h) * 100, 1),
                "w": round((mw / w) * 100, 1),
                "h": round((mh / h) * 100, 1),
                "color": "#00E08A",
                "status": "OPEN" if self.stabilizer.causal_logic.box_open else "CLOSED"
            })
        if red_box and "rect" in red_box:
            rx, ry, rw, rh = red_box["rect"]
            detected_boxes_list.append({
                "id": "red_box",
                "label": "RED CUBE [SAMPLE-A]",
                "confidence": round(red_box.get("score", 0.90), 2),
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
                "confidence": round(blue_box.get("score", 0.90), 2),
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
                "main_box": "OPEN" if self.stabilizer.causal_logic.box_open else "CLOSED",
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
                "box_open": self.stabilizer.causal_logic.box_open,
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
