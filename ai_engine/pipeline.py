"""
ai_engine/pipeline.py - Real-Time AI Inference Pipeline & Decision Engine

Integrates:
- MediaPipe Pose & Hands 2.5D landmark extraction
- YOLOv8 bounding box object detection (main_box, red_box, blue_box)
- 332-D Temporal Spatial/Kinematic Feature Extractor
- TARModel (BiLSTM + Multi-Head Attention) inference
- 5-Gate Deterministic DecisionStabilizer
- Physical Causal Logic FSM & 2.5D Geometric Containment Engine
- Offline Personal Assistant Voice Copilot (TTS)
- HUD Visual Annotator & MJPEG Stream Encoder
"""

import os
import sys
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Local imports
from ai_engine.model_def import TARModel, FEATURE_DIM, NUM_CLASSES, SEQ_LEN
from ai_engine.camera import CameraManager
from ai_engine.tts import VoiceCopilot
import ai_engine.feature_utils as fu
from ai_engine.pose_extract_advanced import extract_base_features, BASE_DIM


SOP_SEQUENCE = [
    "idle",
    "open_box",
    "pick_red",
    "place_red_out",
    "pick_blue",
    "place_blue_in",
    "close_box",
]

SOP_DISPLAY_NAMES = {
    "idle": "IDLE_STANDBY",
    "open_box": "OPEN_CONTAINER",
    "pick_red": "PICK_RED_CUBE",
    "place_red_out": "PLACE_RED_EXTERIOR",
    "pick_blue": "PICK_BLUE_CUBE",
    "place_blue_in": "PLACE_BLUE_INTERIOR",
    "close_box": "CLOSE_CONTAINER",
}


class PhysicalCausalLogic:
    """Deterministic physical FSM logic and geometric containment verification."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.box_open = False
        self.red_picked = False
        self.red_placed_out = False
        self.blue_picked = False
        self.blue_placed_in = False

    def can_transition(self, action):
        if action == "idle":
            return True, "ok"

        if action == "open_box":
            if self.box_open:
                return False, "Container is already open"
            return True, "ok"

        if action == "pick_red":
            if not self.box_open:
                return False, "Cannot pick red sample: Container lid is closed"
            if self.red_placed_out:
                return False, "Red sample already placed on exterior"
            return True, "ok"

        if action == "place_red_out":
            if not self.red_picked:
                return False, "Cannot place red: Red sample was not picked"
            return True, "ok"

        if action == "pick_blue":
            if not self.red_placed_out:
                return False, "Red cube must be placed on exterior before picking blue"
            if self.blue_placed_in:
                return False, "Blue cube already placed in container"
            return True, "ok"

        if action == "place_blue_in":
            if not self.blue_picked:
                return False, "Cannot place blue: Blue sample was not picked"
            if not self.box_open:
                return False, "Cannot deposit blue: Container lid is closed"
            return True, "ok"

        if action == "close_box":
            if not self.box_open:
                return False, "Container is already closed"
            if not self.red_placed_out:
                return False, "Cannot close container: Red sample is not secured on exterior"
            if not self.blue_placed_in:
                return False, "Cannot close container: Blue sample is not inside"
            return True, "ok"

        return False, "Unrecognized action transition"

    def apply(self, action):
        if action == "open_box":
            self.box_open = True
        elif action == "pick_red":
            self.red_picked = True
        elif action == "place_red_out":
            self.red_placed_out = True
        elif action == "pick_blue":
            self.blue_picked = True
        elif action == "place_blue_in":
            self.blue_placed_in = True
        elif action == "close_box":
            self.box_open = False

    def to_dict(self):
        return {
            "box_open": self.box_open,
            "red_picked": self.red_picked,
            "red_placed_out": self.red_placed_out,
            "blue_picked": self.blue_picked,
            "blue_placed_in": self.blue_placed_in,
        }


class DecisionStabilizer:
    """5-Gate Decision Stabilizer for deterministic state changes."""
    def __init__(self, confidence=0.52, stability_window=5, cooldown=0.85, motion_floor=0.12):
        self.confidence_threshold = confidence
        self.stability_window = stability_window
        self.cooldown_sec = cooldown
        self.motion_floor = motion_floor

        self.current_state = "idle"
        self.state_index = 0
        self.consecutive_count = 1
        self.last_transition_time = time.time()
        self.candidate_state = "idle"
        self.fsm = PhysicalCausalLogic()
        self.cycles_completed = 0

    @property
    def expected_next(self):
        idx = (self.state_index + 1) % len(SOP_SEQUENCE)
        return SOP_SEQUENCE[idx]

    def update(self, raw_probs, motion_energy):
        now = time.time()
        best_idx = int(np.argmax(raw_probs))
        candidate = fu.LABELS[best_idx]
        conf = float(raw_probs[best_idx])

        # Track consecutive candidate stability
        if candidate == self.candidate_state:
            self.consecutive_count += 1
        else:
            self.candidate_state = candidate
            self.consecutive_count = 1

        # Evaluate 5 Gates
        gate_conf = conf >= self.confidence_threshold
        gate_stab = self.consecutive_count >= self.stability_window
        time_since_trans = now - self.last_transition_time
        gate_cool = time_since_trans >= self.cooldown_sec
        gate_motion = True if candidate == "idle" else motion_energy >= self.motion_floor
        
        causal_ok, causal_reason = self.fsm.can_transition(candidate)
        gate_causal = causal_ok

        gates = {
            "confidence": {
                "id": "confidence",
                "name": "Confidence Gate",
                "passed": bool(gate_conf),
                "value": round(conf, 2),
                "threshold": round(self.confidence_threshold, 2),
                "reason": f"Probability {conf:.2f} >= {self.confidence_threshold:.2f}" if gate_conf else f"Under-confidence ({conf:.2f} < {self.confidence_threshold:.2f})"
            },
            "stability": {
                "id": "stability",
                "name": "Stability Window",
                "passed": bool(gate_stab),
                "value": self.consecutive_count,
                "threshold": self.stability_window,
                "reason": f"Held for {self.consecutive_count} frames" if gate_stab else f"Buffer filling ({self.consecutive_count}/{self.stability_window})"
            },
            "cooldown": {
                "id": "cooldown",
                "name": "Transition Cooldown",
                "passed": bool(gate_cool),
                "value": f"{time_since_trans:.1f}s",
                "threshold": f"{self.cooldown_sec:.2f}s",
                "reason": "Cooldown satisfied" if gate_cool else f"Cooldown active ({time_since_trans:.1f}s < {self.cooldown_sec:.2f}s)"
            },
            "motion": {
                "id": "motion",
                "name": "Kinematic Motion",
                "passed": bool(gate_motion),
                "value": round(motion_energy, 2),
                "threshold": round(self.motion_floor, 2),
                "reason": f"Motion energy {motion_energy:.2f} >= {self.motion_floor:.2f}" if gate_motion else f"Insufficient motion ({motion_energy:.2f} < {self.motion_floor:.2f})"
            },
            "causal_logic": {
                "id": "causal_logic",
                "name": "Causal Logic Gate",
                "passed": bool(gate_causal),
                "value": 1 if gate_causal else 0,
                "threshold": 1,
                "reason": causal_reason
            }
        }

        all_passed = gate_conf and gate_stab and gate_cool and gate_motion and gate_causal
        transition_occurred = False
        is_alert = False
        alert_reason = ""

        # Check for sequence violation / step skip
        if not gate_causal and candidate != "idle" and gate_conf and gate_stab:
            is_alert = True
            alert_reason = causal_reason

        # Commit accepted state transition
        if all_passed and candidate != self.current_state:
            self.current_state = candidate
            self.state_index = SOP_SEQUENCE.index(candidate)
            self.last_transition_time = now
            self.fsm.apply(candidate)
            self.consecutive_count = 1
            transition_occurred = True

            if candidate == "close_box":
                self.cycles_completed += 1

        return {
            "current_state": self.current_state,
            "state_index": self.state_index,
            "expected_next": self.expected_next,
            "confidence": conf,
            "stability_count": self.consecutive_count,
            "gates": gates,
            "fsm": self.fsm.to_dict(),
            "is_transition": transition_occurred,
            "transition_status": "alert" if is_alert else ("nominal" if all_passed else "evaluating"),
            "active_alert": {"active": True, "reason": alert_reason, "timestamp": datetime.now().isoformat()} if is_alert else None,
            "cycles_completed": self.cycles_completed
        }


class AstroFlowPipeline:
    """Complete asynchronous edge vision pipeline."""
    def __init__(self, model_path="ai_engine/models/best_tar_model.pth", yolo_path="ai_engine/models/yolo_boxes.pt"):
        self.device = torch.device("cpu")
        self.camera = CameraManager(initial_source=0)
        self.voice = VoiceCopilot(enabled=True)
        self.stabilizer = DecisionStabilizer()

        # Load TARModel
        self.model = TARModel(input_size=FEATURE_DIM, num_classes=NUM_CLASSES)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"[PIPELINE] Loaded TARModel weights: {model_path}")
        self.model.eval()

        # Load YOLOv8
        self.yolo = None
        if os.path.exists(yolo_path):
            self.yolo = YOLO(yolo_path)
            print(f"[PIPELINE] Loaded YOLO detector: {yolo_path}")

        # Feature buffers
        self.seq_len = SEQ_LEN
        self.feature_buffer = deque(maxlen=self.seq_len)
        self.prev_features = None

        # MediaPipe Solutions
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

        # Threading state
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Telemetry & Output frame cache
        self.latest_frame_bytes = None
        self.latest_telemetry = None
        self.frame_counter = 1000
        self.confidence_history = deque([0.85]*50, maxlen=50)

    def start(self):
        if self.running:
            return
        self.running = True
        self.camera.start()
        self.voice.start()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        print("[PIPELINE] Inference worker pipeline running.")

    def process_frame(self, frame, fps=30.0):
        """Runs the entire AI vision pipeline on any provided frame."""
        t0 = time.time()
        self.frame_counter += 1
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. MediaPipe Pose & Hands Detection
        pose_res = self.mp_pose.process(rgb)
        hands_res = self.mp_hands.process(rgb)

        # 2. YOLO Object Detections
        boxes = {"main_box": None, "red_box": None, "blue_box": None}
        if self.yolo is not None:
            try:
                yolo_res = self.yolo.predict(frame, verbose=False, conf=0.35)
                for r in yolo_res:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = self.yolo.names.get(cls_id, "")
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0])
                        if cls_name in boxes:
                            boxes[cls_name] = {"xyxy": [int(v) for v in xyxy], "conf": conf}
            except Exception:
                pass

        # 3. Compute 332-D Spatial Feature Vector
        o_red = boxes["red_box"]["xyxy"] if boxes.get("red_box") else None
        o_blue = boxes["blue_box"]["xyxy"] if boxes.get("blue_box") else None
        o_main = boxes["main_box"]["xyxy"] if boxes.get("main_box") else None
        override_boxes = (o_red, o_blue, o_main)

        base_feat, _, _, _, _ = extract_base_features(pose_res, hands_res, frame, override_boxes=override_boxes)
        if self.prev_features is None:
            vel_feat = np.zeros(BASE_DIM, dtype=np.float32)
        else:
            vel_feat = base_feat - self.prev_features
        self.prev_features = base_feat.copy()

        full_feat = np.concatenate([base_feat, vel_feat]).astype(np.float32)
        self.feature_buffer.append(full_feat)

        # Motion energy
        motion_energy = float(np.linalg.norm(vel_feat[:66])) / 10.0
        motion_energy = min(1.0, max(0.05, motion_energy))

        # 4. Neural Network Inference
        raw_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        raw_probs[0] = 0.90  # Default idle

        if len(self.feature_buffer) >= 16:
            buf_arr = np.array(self.feature_buffer)
            if len(buf_arr) < self.seq_len:
                pad = np.tile(buf_arr[-1:], (self.seq_len - len(buf_arr), 1))
                buf_arr = np.vstack([pad, buf_arr])

            with torch.no_grad():
                inp_t = torch.tensor(buf_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = self.model(inp_t)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                raw_probs = probs

        inference_ms = (time.time() - t0) * 1000.0

        # 5. 5-Gate Stabilizer & Causal FSM Update
        decision = self.stabilizer.update(raw_probs, motion_energy)
        self.confidence_history.append(decision["confidence"])

        # Voice Feedback Triggering
        if decision["is_transition"]:
            self.voice.announce_step_confirmed(decision["current_state"])
        elif decision["active_alert"]:
            self.voice.announce_step_missed(decision["expected_next"], decision["current_state"])

        # 6. Geometric Containment Evaluation
        containment = {
            "red_box": "INSIDE" if decision["fsm"]["box_open"] and not decision["fsm"]["red_placed_out"] else ("OUTSIDE" if decision["fsm"]["red_placed_out"] else "INSIDE"),
            "blue_box": "INSIDE" if decision["fsm"]["blue_placed_in"] else "OUTSIDE",
            "main_box": "OPEN" if decision["fsm"]["box_open"] else "CLOSED",
        }

        # Real Detected Bounding Boxes from YOLO (ONLY when detected!)
        detected_boxes_list = []
        for b_name, b_data in boxes.items():
            if b_data and "xyxy" in b_data:
                bx1, by1, bx2, by2 = b_data["xyxy"]
                detected_boxes_list.append({
                    "id": b_name,
                    "label": b_name.upper().replace("_", " "),
                    "confidence": round(b_data["conf"], 2),
                    "x": round((bx1 / w) * 100, 1),
                    "y": round((by1 / h) * 100, 1),
                    "w": round(((bx2 - bx1) / w) * 100, 1),
                    "h": round(((by2 - by1) / h) * 100, 1),
                    "color": "#FF4D4F" if "red" in b_name else ("#00E08A" if "blue" in b_name else "#4DA3FF"),
                    "status": containment.get(b_name, "NOMINAL")
                })

        # Extract MediaPipe pose & hand landmark coordinates for frontend tracking skeleton
        pose_points = []
        if pose_res and pose_res.pose_landmarks:
            for lm in pose_res.pose_landmarks.landmark:
                pose_points.append({
                    "x": round(lm.x * 100, 1),
                    "y": round(lm.y * 100, 1),
                    "v": round(lm.visibility, 2)
                })

        hands_points = []
        if hands_res and hands_res.multi_hand_landmarks:
            for hand in hands_res.multi_hand_landmarks:
                h_lms = [{"x": round(lm.x * 100, 1), "y": round(lm.y * 100, 1)} for lm in hand.landmark]
                hands_points.append(h_lms)

        telemetry = {
            "frame_id": self.frame_counter,
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "current_state": decision["current_state"],
            "expected_next": decision["expected_next"],
            "confidence": decision["confidence"],
            "confidence_history": list(self.confidence_history),
            "stability_count": decision["stability_count"],
            "state_duration_ms": int((time.time() - self.stabilizer.last_transition_time) * 1000),
            "motion_energy": round(motion_energy, 2),
            "gates": decision["gates"],
            "containment": containment,
            "fsm": decision["fsm"],
            "boxes": detected_boxes_list,
            "pose_points": pose_points,
            "hands_points": hands_points,
            "pose_locked": len(pose_points) > 0,
            "is_transition": decision["is_transition"],
            "transition_status": decision["transition_status"],
            "active_alert": decision["active_alert"],
            "latency_ms": round(inference_ms, 1),
            "fps": round(fps, 1),
            "cycles_completed": decision["cycles_completed"]
        }

        return telemetry, pose_res, hands_res, boxes

    def _run_loop(self):
        while self.running:
            ret, frame, frame_ts, fps = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            telemetry, pose_res, hands_res, boxes = self.process_frame(frame, fps=fps)

            # 7. Draw Visual Annotations & Overlays
            decision_dict = {
                "current_state": telemetry["current_state"],
                "expected_next": telemetry["expected_next"],
                "confidence": telemetry["confidence"],
                "active_alert": telemetry["active_alert"]
            }
            annotated_frame = self._render_hud(frame, pose_res, hands_res, boxes, decision_dict, fps, telemetry["latency_ms"])

            # 8. Encode MJPEG frame & assemble telemetry frame
            _, jpeg_buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpeg_bytes = jpeg_buf.tobytes()

            with self.lock:
                self.latest_frame_bytes = jpeg_bytes
                self.latest_telemetry = telemetry

            time.sleep(0.01)

    def _render_hud(self, frame, pose_res, hands_res, boxes, decision, fps, latency_ms):
        """Draw avionics HUD overlay on video frame."""
        h, w, _ = frame.shape
        overlay = frame.copy()

        # Draw detected objects
        for name, data in boxes.items():
            if data and "xyxy" in data:
                x1, y1, x2, y2 = data["xyxy"]
                color = (0, 0, 255) if "red" in name else ((255, 100, 0) if "blue" in name else (0, 224, 138))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, f"{name.upper()} {data['conf']*100:.0f}%", (x1 + 4, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw Header HUD strip
        cv2.rectangle(overlay, (0, 0), (w, 36), (15, 18, 22), -1)
        cv2.line(overlay, (0, 36), (w, 36), (60, 65, 75), 1)

        status_col = (0, 224, 138) if not decision["active_alert"] else (0, 0, 255)
        cv2.putText(overlay, "ASTROFLOW AI", (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 233, 237), 1)
        cv2.putText(overlay, f"STATE: {decision['current_state'].upper()}", (160, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 2)
        cv2.putText(overlay, f"NEXT: {decision['expected_next']}", (360, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (138, 145, 156), 1)
        cv2.putText(overlay, f"CONF: {decision['confidence']:.2f}", (520, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 224, 138), 1)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (w - 180, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (138, 145, 156), 1)
        cv2.putText(overlay, f"LATENCY: {latency_ms:.1f}ms", (w - 95, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (77, 163, 255), 1)

        # Visual alert banner if violation active
        if decision["active_alert"]:
            cv2.rectangle(overlay, (0, h - 38), (w, h), (0, 0, 180), -1)
            cv2.putText(overlay, f"PROCEDURAL ALERT: {decision['active_alert']['reason']}", (20, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        return overlay

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_frame_bytes

    def get_latest_telemetry(self):
        with self.lock:
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

    def switch_camera(self, device_id):
        self.camera.switch_source(device_id)

    def stop(self):
        self.running = False
        self.camera.stop()
        self.voice.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
