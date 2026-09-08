"""
Live inference: camera -> pose/hands -> TAR model (state classification)
                        -> YOLO (auxiliary object detection)
                        -> stabilized state decision

WHY "STATE" ISN'T JUST "WHATEVER THE MODEL SAID THIS FRAME":
A single frame's prediction, even a high-confidence one, is not trusted on
its own - that's how you get hallucinated state flips from one noisy frame.
Three gates have to all pass before a new state is accepted:

  1. CONFIDENCE  - the model's top-class probability must clear
                    CONFIDENCE_THRESHOLD.
  2. STABILITY    - the SAME class must be the (confidence-passing)
                    prediction for STABILITY_WINDOW consecutive frames in
                    a row - a single flickered frame resets this count.
  3. COOLDOWN     - even after 1 and 2 pass, a new state won't be accepted
                    until COOLDOWN_SEC has elapsed since the last accepted
                    change - prevents rapid back-and-forth flipping.
  + MOTION GATE   - a non-idle prediction is rejected outright if there's
                    essentially no movement between frames. A model saying
                    "pick_red" while nothing is moving is a classic
                    hallucination signature, not a real action.

This logic lives in DecisionStabilizer, tested independently of the camera
loop - see the accompanying tests.

ON YOLO'S CURRENT ROLE:
yolov8n.pt is COCO-pretrained - it has no concept of your red_box/blue_box/
main_box specifically, so today it can only report generic categories. It's
wired in, displayed on screen, and logged, but is NOT a hard requirement
for state changes - it can't reliably confirm your actual objects yet.
Point YOLO_MODEL_PATH at a fine-tuned custom model later and this same
code starts getting real object-identity evidence with no changes needed.

Usage:
    python realtime.py
"""

import os
import time
import csv
import threading
from collections import deque

import cv2
import numpy as np
import torch

from model_def import TARModel, NUM_CLASSES, FEATURE_DIM, SEQ_LEN
import feature_utils as fu
from pose_extract_advanced import extract_base_features, BASE_DIM

# Optional non-blocking sound feedback (Windows)
try:
    import winsound
    def play_sound(freq=1200, dur=70):
        threading.Thread(target=lambda: winsound.Beep(freq, dur), daemon=True).start()
except Exception:
    def play_sound(freq=1200, dur=70):
        pass

MODEL_PATH = "best_tar_model1.pth" if os.path.exists("best_tar_model1.pth") else "best_tar_model.pth"
YOLO_MODEL_PATH = "yolo_boxes.pt"
CAMERA_ID = 0

CONFIDENCE_THRESHOLD = 0.52          # Responsive threshold for valid actions
STABILITY_WINDOW = 5                 # 5 consecutive frames (~0.17s) ensures intentional action
COOLDOWN_SEC = 0.85                  # 0.85s cooldown prevents hand retraction from triggering next step
MOTION_THRESHOLD = 0.010             # Require minimal movement for action (hands resting = idle)
EMA_ALPHA = 0.65                     # Responsive probability smoothing
DWELL_TIME_SEC = 1.0                 # Visual dwell: hold confirmed action on screen for 1.0s
USE_FSM = True                       # Physical Causal Logic gate
IDLE_LABEL = fu.LABELS[0]            # "idle" - exempt from motion gate
FEEDBACK_DIR = "feedback_data"


# ===================================================================== #
# Level 1: Geometric 2.5D Box Containment Engine
# ===================================================================== #
class GeometricContainmentEngine:
    """
    Level 1: 2.5D Geometric Box Containment Engine.
    Uses real-time 2D bounding boxes and depth/area relationships to compute:
      - Is the Red Box physically INSIDE or OUTSIDE the Main Container?
      - Is the Blue Box physically INSIDE or OUTSIDE the Main Container?
      - Are the boxes being actively HELD by the operator's hands?
    """
    def __init__(self, buffer_size=8):
        self.buffer_size = buffer_size
        self.red_history = deque(maxlen=buffer_size)
        self.blue_history = deque(maxlen=buffer_size)

        self.red_is_inside = False
        self.blue_is_inside = False
        self.red_is_held = False
        self.blue_is_held = False

        self.red_status = "UNKNOWN"   # "INSIDE", "OUTSIDE", "HELD", "UNKNOWN"
        self.blue_status = "UNKNOWN"

    def reset(self):
        self.red_history.clear()
        self.blue_history.clear()
        self.red_is_inside = False
        self.blue_is_inside = False
        self.red_is_held = False
        self.blue_is_held = False
        self.red_status = "UNKNOWN"
        self.blue_status = "UNKNOWN"

    def update(self, red_box, blue_box, main_box, pose_landmarks, frame_shape):
        if main_box is None:
            self.red_status = "UNKNOWN"
            self.blue_status = "UNKNOWN"
            return

        mx, my, mw, mh = main_box["rect"]
        h, w = frame_shape[:2]

        # Container interior bounds with small perspective margin
        margin_x = int(mw * 0.08)
        margin_y = int(mh * 0.08)
        mx0, mx1 = mx - margin_x, mx + mw + margin_x
        my0, my1 = my - margin_y, my + mh + margin_y

        # Hands wrist coordinates
        lms = pose_landmarks.landmark if pose_landmarks else None
        wrist_pts = []
        if lms and len(lms) > 16:
            for wi in (15, 16):
                wrist_pts.append((lms[wi].x * w, lms[wi].y * h))

        # 1. Evaluate Red Box
        if red_box is not None:
            rx, ry, rw, rh = red_box["rect"]
            rcx, rcy = rx + rw / 2.0, ry + rh / 2.0
            self.red_is_held = any(np.hypot(rcx - wx, rcy - wy) < (max(rw, rh) * 1.5 + 40) for wx, wy in wrist_pts)
            inside = (mx0 <= rcx <= mx1) and (my0 <= rcy <= my1)
            self.red_history.append(inside)
            self.red_is_inside = (sum(self.red_history) > len(self.red_history) / 2)
            if self.red_is_held:
                self.red_status = "HELD"
            else:
                self.red_status = "INSIDE" if self.red_is_inside else "OUTSIDE"
        else:
            self.red_status = "UNKNOWN"

        # 2. Evaluate Blue Box
        if blue_box is not None:
            bx, by, bw, bh = blue_box["rect"]
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            self.blue_is_held = any(np.hypot(bcx - wx, bcy - wy) < (max(bw, bh) * 1.5 + 40) for wx, wy in wrist_pts)
            inside = (mx0 <= bcx <= mx1) and (my0 <= bcy <= my1)
            self.blue_history.append(inside)
            self.blue_is_inside = (sum(self.blue_history) > len(self.blue_history) / 2)
            if self.blue_is_held:
                self.blue_status = "HELD"
            else:
                self.blue_status = "INSIDE" if self.blue_is_inside else "OUTSIDE"
        else:
            self.blue_status = "UNKNOWN"

    def get_summary(self):
        r_icon = "[v]" if self.red_status != "UNKNOWN" else "[?]"
        b_icon = "[v]" if self.blue_status != "UNKNOWN" else "[?]"
        return f"Physics 2.5D: Red: {self.red_status} {r_icon} | Blue: {self.blue_status} {b_icon}"

    def draw(self, frame, main_box):
        """Draws 2.5D container volume wireframe over main container."""
        if main_box is None:
            return
        mx, my, mw, mh = main_box["rect"]
        cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 120), 2)
        # Perspective depth inset
        inset = min(16, max(6, min(mw, mh) // 8))
        cv2.rectangle(frame, (mx + inset, my + inset), (mx + mw - inset, my + mh - inset), (0, 200, 100), 1)
        cv2.line(frame, (mx, my), (mx + inset, my + inset), (0, 200, 100), 1)
        cv2.line(frame, (mx + mw, my), (mx + mw - inset, my + inset), (0, 200, 100), 1)
        cv2.line(frame, (mx, my + mh), (mx + inset, my + mh - inset), (0, 200, 100), 1)
        cv2.line(frame, (mx + mw, my + mh), (mx + mw - inset, my + mh - inset), (0, 200, 100), 1)
        cv2.putText(frame, "3D CONTAINER VOLUME", (mx + 8, max(18, my - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1)


# ===================================================================== #
# Human Physical / Causal Logic + Geometric Verification
# ===================================================================== #
class PhysicalCausalLogic:
    """
    Strict human physical and causal reality combined with 2.5D geometric validation:
      1. Main box cannot be closed without being open.
      2. Red item can only be picked when the box is open, and not already placed out.
      3. Red item must be placed out before Blue item can be picked up.
      4. Blue item can only be picked after red item is placed out.
      5. Blue item must be placed inside the box before closing.
      6. Box cannot be closed while holding either red or blue.
      7. Box cannot be closed before red is out and blue is in.
      8. Idle is always permitted at any moment without changing physical state.
      + Physical 2.5D Containment Check:
         - place_red_out is blocked if red box is still inside main box volume.
         - place_blue_in is blocked if blue box is still outside main box volume.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.box_open = False
        self.red_picked = False
        self.red_placed_out = False
        self.blue_picked = False
        self.blue_placed_in = False

    def can_transition(self, action, containment=None):
        if action == "idle":
            return True, "ok"

        if action == "open_box":
            if self.box_open:
                return False, "box already open"
            return True, "ok"

        if action == "pick_red":
            if not self.box_open:
                return False, "box must be open before picking red"
            if self.red_placed_out:
                return False, "red already placed out"
            if self.red_picked:
                return False, "red already in hand"
            if containment and containment.red_status == "OUTSIDE":
                return False, "physics: red box is already outside"
            return True, "ok"

        if action == "place_red_out":
            if not self.box_open:
                return False, "box must be open"
            if self.red_placed_out:
                return False, "red already placed out"
            if not self.red_picked:
                return False, "red must be picked up first"
            if containment and containment.red_status == "INSIDE":
                return False, "physics: red is still inside container"
            return True, "ok"

        if action == "pick_blue":
            if not self.box_open:
                return False, "box must be open before handling items"
            if not self.red_placed_out:
                return False, "red item must be placed out before picking blue"
            if self.blue_placed_in:
                return False, "blue already placed inside"
            if self.blue_picked:
                return False, "blue already in hand"
            if containment and containment.blue_status == "INSIDE":
                return False, "physics: blue is already inside container"
            return True, "ok"

        if action == "place_blue_in":
            if not self.box_open:
                return False, "box must be open to place blue inside"
            if self.blue_placed_in:
                return False, "blue already placed inside"
            if not self.blue_picked:
                return False, "blue must be picked up first"
            if containment and containment.blue_status == "OUTSIDE":
                return False, "physics: blue is still outside on table"
            return True, "ok"

        if action == "close_box":
            if not self.box_open:
                return False, "box cannot close without being open"
            if self.red_picked or self.blue_picked:
                return False, "cannot close box while holding an item"
            if not self.red_placed_out:
                return False, "red must be placed out before closing"
            if not self.blue_placed_in:
                return False, "blue must be placed inside before closing"
            if containment and containment.red_status == "INSIDE":
                return False, "physics: red cannot be inside when closing"
            return True, "ok"

        return True, "ok"

    def apply_transition(self, action):
        """Updates internal physical state once an action is confirmed."""
        if action == "open_box":
            self.box_open = True
        elif action == "pick_red":
            self.red_picked = True
        elif action == "place_red_out":
            self.red_picked = False
            self.red_placed_out = True
        elif action == "pick_blue":
            self.blue_picked = True
        elif action == "place_blue_in":
            self.blue_picked = False
            self.blue_placed_in = True
        elif action == "close_box":
            # Cycle complete -> reset for next box
            self.reset()

    def get_status_summary(self, dominant_color="NONE"):
        box_str = "OPEN" if self.box_open else "CLOSED"
        if self.red_picked:
            red_str = "HELD"
        elif self.red_placed_out:
            red_str = "OUT"
        else:
            red_str = "IN"

        if self.blue_picked:
            blue_str = "HELD"
        elif self.blue_placed_in:
            blue_str = "IN"
        else:
            blue_str = "OUT"

        yolo_str = f" | Active: {dominant_color}" if dominant_color != "NONE" else ""
        return f"Box: {box_str} | Red: {red_str} | Blue: {blue_str}{yolo_str}"


# ===================================================================== #
# Anti-hallucination decision logic (Human Causal Logic + SOP Guided)
# ===================================================================== #
class DecisionStabilizer:
    """
    Confidence + EMA smoothing + fast temporal latch + Human Causal Logic + Visual Dwell.
    Guarantees visible feedback and effortless return to idle when movement pauses.
    """
    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD,
                 stability_window=STABILITY_WINDOW,
                 cooldown_sec=COOLDOWN_SEC,
                 use_fsm=USE_FSM,
                 ema_alpha=EMA_ALPHA,
                 dwell_sec=DWELL_TIME_SEC):
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        self.cooldown_sec = cooldown_sec
        self.use_fsm = use_fsm
        self.ema_alpha = ema_alpha
        self.dwell_sec = dwell_sec

        self.smoothed_probs = None
        self.recent = deque(maxlen=stability_window)
        self.current_state = "idle"
        self.last_change_time = 0.0
        self.dwell_until = 0.0
        self.last_rejected_reason = ""
        self.dominant_color = "NONE"
        self.causal_logic = PhysicalCausalLogic()

    def update(self, raw_probs, motion, now, expected_action=None, dominant_color="NONE", containment=None):
        if raw_probs is None:
            return False, self.current_state, None, 0.0, "no input"

        self.dominant_color = dominant_color
        self.containment = containment

        # 1. EMA Probability Smoothing
        if self.smoothed_probs is None:
            self.smoothed_probs = raw_probs.copy()
        else:
            self.smoothed_probs = (1.0 - self.ema_alpha) * self.smoothed_probs + self.ema_alpha * raw_probs

        pred_idx = int(np.argmax(self.smoothed_probs))
        confidence = float(self.smoothed_probs[pred_idx])
        predicted_class = fu.LABELS[pred_idx]

        # 2. Visual Dwell Lock: Keep the confirmed state visible for dwell_sec
        if now < self.dwell_until:
            return False, self.current_state, predicted_class, confidence, "dwelling"

        # 3. Auto-Return to Idle: When hands stop moving after an action, settle cleanly to idle
        if self.current_state != "idle":
            if motion < MOTION_THRESHOLD or (predicted_class == "idle" and confidence > 0.35):
                self.current_state = "idle"
                self.recent.clear()
                self.last_rejected_reason = "settled to idle"
                return True, self.current_state, "idle", 1.0, "settled to idle"

        # 4. Motion Gate: non-idle actions require actual physical movement
        if predicted_class != IDLE_LABEL and motion < MOTION_THRESHOLD:
            self.last_rejected_reason = "low motion"
            return False, self.current_state, predicted_class, confidence, self.last_rejected_reason

        # 5. Cooldown Gate: enforce 0.85s pause between actions so hand retraction cannot trigger
        if self.last_change_time > 0 and (now - self.last_change_time) < self.cooldown_sec:
            rem = self.cooldown_sec - (now - self.last_change_time)
            self.last_rejected_reason = f"cooldown ({rem:.1f}s)"
            return False, self.current_state, predicted_class, confidence, self.last_rejected_reason

        # 6. Physical Causal Reality Gate: block actions that are physically impossible
        if self.use_fsm:
            allowed, reason = self.causal_logic.can_transition(predicted_class, containment=containment)
            if not allowed:
                self.last_rejected_reason = f"blocked: {reason}"
                return False, self.current_state, predicted_class, confidence, self.last_rejected_reason

        # 7. Confidence Gate: 0.50 for expected SOP step, 0.65 for unprompted action
        req_thresh = 0.50 if (expected_action and predicted_class == expected_action) else self.confidence_threshold
        if confidence < req_thresh:
            self.last_rejected_reason = f"low conf ({confidence:.2f}<{req_thresh:.2f})"
            return False, self.current_state, predicted_class, confidence, self.last_rejected_reason

        # 8. Stability Latch: require STABILITY_WINDOW (5 frames ~0.17s) of consistent prediction
        self.recent.append(predicted_class)
        if len(self.recent) < self.stability_window or any(x != predicted_class for x in self.recent):
            match_cnt = self.recent.count(predicted_class)
            self.last_rejected_reason = f"latching ({match_cnt}/{self.stability_window})"
            return False, self.current_state, predicted_class, confidence, self.last_rejected_reason

        candidate = predicted_class
        if candidate == self.current_state:
            self.last_rejected_reason = "active"
            return False, self.current_state, candidate, confidence, self.last_rejected_reason

        # All gates passed -> State transition accepted!
        self.current_state = candidate
        self.last_change_time = now
        if candidate != "idle":
            self.dwell_until = now + self.dwell_sec
        self.recent.clear()
        self.causal_logic.apply_transition(candidate)
        self.last_rejected_reason = "accepted"
        return True, self.current_state, candidate, confidence, self.last_rejected_reason

    def confirm_spotted_action(self, grounded_probs, now, expected_action=None, containment=None):
        """
        Directly accepts a spotted gesture burst from MotionActionSpotter
        when a physical movement completes and settles.
        """
        pred_idx = int(np.argmax(grounded_probs))
        conf = float(grounded_probs[pred_idx])
        cand = fu.LABELS[pred_idx]

        if cand == "idle":
            return False, self.current_state, "idle", conf, "idle gesture"

        # 1. Cooldown Gate: Enforce full cooldown between separate actions
        if self.last_change_time > 0 and (now - self.last_change_time) < self.cooldown_sec:
            rem = self.cooldown_sec - (now - self.last_change_time)
            self.last_rejected_reason = f"cooldown ({rem:.1f}s)"
            return False, self.current_state, cand, conf, self.last_rejected_reason

        # 2. Strict Confidence Gate: 0.50 for expected step, 0.65 for unprompted
        min_conf = 0.50 if (expected_action and cand == expected_action) else 0.65
        if conf < min_conf:
            self.last_rejected_reason = f"low spotted conf ({conf:.2f} < {min_conf:.2f})"
            return False, self.current_state, cand, conf, self.last_rejected_reason

        # 3. Strict Human Physical Causal Logic Check
        if self.use_fsm:
            allowed, reason = self.causal_logic.can_transition(cand, containment=containment or self.containment)
            if not allowed:
                self.last_rejected_reason = f"blocked: {reason}"
                return False, self.current_state, cand, conf, self.last_rejected_reason

        # All gates passed -> State transition accepted!
        self.current_state = cand
        self.last_change_time = now
        self.dwell_until = now + self.dwell_sec
        self.recent.clear()
        self.causal_logic.apply_transition(cand)
        self.last_rejected_reason = "accepted (spotted)"
        return True, self.current_state, cand, conf, "accepted (spotted)"


# ===================================================================== #
# Model / feature helpers
# ===================================================================== #
def load_tar_model(path):
    model = TARModel(input_size=FEATURE_DIM, num_classes=NUM_CLASSES)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def normalize_window(window):
    """
    MUST match train_tar.py's TARDataset normalization exactly (single
    scalar mean/std over the whole window), or the model sees a different
    input distribution than it was trained on - a silent, easy-to-miss way
    to sabotage inference without any error being raised.
    """
    arr = np.array(window, dtype=np.float32)
    mean = np.mean(arr)
    std = np.std(arr) + 1e-6
    arr = (arr - mean) / std
    arr = np.clip(arr, -5.0, 5.0)
    return arr


def compute_motion(prev_feat, curr_feat):
    if prev_feat is None:
        return 0.0
    return float(np.mean(np.abs(curr_feat - prev_feat)))


def predict_from_window(tar_model, window):
    """Runs the TAR model on a full window, returns (label, confidence, probs)."""
    normed = normalize_window(window)
    with torch.no_grad():
        x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)
        logits = tar_model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    label = fu.LABELS[pred_idx]
    return label, confidence, probs


def resample_sequence(data, target_len=SEQ_LEN):
    """Linearly resamples a sequence of shape (T, D) to (target_len, D)."""
    T, D = data.shape
    if T == target_len:
        return data
    orig_idx = np.linspace(0, T - 1, num=target_len)
    resampled = np.zeros((target_len, D), dtype=np.float32)
    t_steps = np.arange(T)
    for d in range(D):
        resampled[:, d] = np.interp(orig_idx, t_steps, data[:, d])
    return resampled


class MotionActionSpotter:
    """
    Motion-Triggered Action Spotter (Peak Scoring Architecture):
    Monitors hand velocity to detect gesture boundaries:
      1. Motion onset: starts buffering gesture trajectory.
      2. Motion peaks: user reaches, grasps, moves.
      3. Motion settles: hand rests or reaches destination for ~5 frames.
    When motion settles, the exact movement burst is resampled to SEQ_LEN (48 frames)
    and classified ONCE with maximum confidence, completely eliminating rolling-window jitter.
    """
    STATE_IDLE = "IDLE"          # Hands resting / waiting for motion
    STATE_TRACKING = "TRACKING"  # Active movement stroke being captured
    STATE_COOLDOWN = "COOLDOWN"  # Brief pause after confirming an action

    def __init__(self,
                 start_motion_thresh=0.016,   # Motion to trigger gesture tracking
                 settle_motion_thresh=0.014,  # Realistic settle motion threshold accounting for landmark jitter
                 min_action_frames=12,        # Minimum duration of a valid gesture (~0.40s)
                 max_action_frames=65,        # Maximum duration before auto-triggering (~2.1s)
                 settle_required_frames=4,    # Number of calm frames needed to confirm motion end (~0.13s)
                 cooldown_sec=0.85):          # 0.85s cooldown prevents hand retraction from triggering next step!
        self.start_motion_thresh = start_motion_thresh
        self.settle_motion_thresh = settle_motion_thresh
        self.min_action_frames = min_action_frames
        self.max_action_frames = max_action_frames
        self.settle_required_frames = settle_required_frames
        self.cooldown_sec = cooldown_sec

        self.state = self.STATE_IDLE
        self.buffer = deque(maxlen=90)
        self.motion_frames = 0
        self.settle_counter = 0
        self.smooth_motion = 0.0
        self.peak_motion = 0.0
        self.last_spot_time = 0.0
        self.last_spotted_action = None
        self.last_spotted_conf = 0.0

    def update(self, feat, motion, now):
        self.buffer.append(feat)
        self.smooth_motion = 0.65 * self.smooth_motion + 0.35 * motion

        triggered = False
        action_clip = None

        if self.state == self.STATE_COOLDOWN:
            if (now - self.last_spot_time) >= self.cooldown_sec:
                self.state = self.STATE_IDLE
                self.motion_frames = 0
                self.settle_counter = 0
                self.peak_motion = 0.0

        if self.state == self.STATE_IDLE:
            if self.smooth_motion >= self.start_motion_thresh:
                self.state = self.STATE_TRACKING
                self.motion_frames = 1
                self.settle_counter = 0
                self.peak_motion = self.smooth_motion

        elif self.state == self.STATE_TRACKING:
            self.motion_frames += 1
            if self.smooth_motion > self.peak_motion:
                self.peak_motion = self.smooth_motion

            if self.smooth_motion < self.settle_motion_thresh:
                self.settle_counter += 1
            else:
                self.settle_counter = 0

            # Condition A: Hand has settled after a meaningful movement
            settle_ready = (self.settle_counter >= self.settle_required_frames and 
                            self.motion_frames >= self.min_action_frames)

            # Condition B: Max duration safeguard
            max_dur_ready = (self.motion_frames >= self.max_action_frames)

            if settle_ready or max_dur_ready:
                clip_len = min(len(self.buffer), self.motion_frames + 6)
                clip_feats = list(self.buffer)[-clip_len:]
                arr = np.array(clip_feats, dtype=np.float32)
                action_clip = resample_sequence(arr, target_len=SEQ_LEN)
                triggered = True

                self.state = self.STATE_COOLDOWN
                self.last_spot_time = now

        status = {
            "spotter_state": self.state,
            "motion_frames": self.motion_frames,
            "smooth_motion": self.smooth_motion,
            "peak_motion": self.peak_motion,
            "last_action": self.last_spotted_action,
            "last_conf": self.last_spotted_conf
        }
        return triggered, action_clip, status


# ===================================================================== #
# YOLO (auxiliary, non-blocking)
# ===================================================================== #
def load_yolo(path):
    if not os.path.exists(path):
        print(f"[YOLO] Model file '{path}' not found - continuing without YOLO overlay.")
        return None
    try:
        from ultralytics import YOLO
        return YOLO(path)
    except Exception as e:
        print(f"[YOLO] Could not load '{path}': {e}. Continuing without YOLO overlay.")
        return None


# Only allow box-like classes from YOLO (filters out person, chair, tv, etc.)
ALLOWED_YOLO_CLASSES = {
    "box", "cardboard box", "package", "container", "crate", "suitcase",
    "red_box", "blue_box", "main_box"
}


def run_yolo(yolo_model, frame, conf_threshold=0.20, allowed_classes=ALLOWED_YOLO_CLASSES):
    if yolo_model is None:
        return []
    try:
        result = yolo_model(frame, verbose=False)[0]
        detections = []
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, str(cls_id)).lower()
                # Ignore persons, chairs, tvs, and other clutter
                if allowed_classes and not any(target in cls_name for target in allowed_classes):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append({"name": cls_name, "conf": conf, "box": (x1, y1, x2, y2)})
        return detections
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        return []


# ===================================================================== #
# YOLO Object-Action Grounding & Per-frame processing
# ===================================================================== #
def get_box_wrist_distance(box, pose_landmarks, frame_shape):
    """Computes minimum normalized distance from either wrist to box center."""
    if box is None or pose_landmarks is None:
        return 999.0
    h, w = frame_shape[:2]
    bx = box["rect"][0] + box["rect"][2] / 2.0
    by = box["rect"][1] + box["rect"][3] / 2.0
    lms = pose_landmarks.landmark
    min_dist = 999.0
    for wi in (15, 16):
        if wi < len(lms):
            wx, wy = lms[wi].x * w, lms[wi].y * h
            dist = np.hypot(bx - wx, by - wy) / max(w, h)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def ground_probs_with_yolo(raw_probs, yolo_red, yolo_blue, red_box, blue_box,
                           pose_landmarks, frame_shape, causal_logic, expected_action=None):
    """
    Gentle Prior Guidance:
    - Eliminates destructive zeroing of probabilities from static background boxes.
    - If user's wrist is actively near a detected box, provides a gentle soft boost (1.25x).
    - If an expected SOP action is known, provides a gentle alignment boost (1.20x).
    - NEVER crushes other classes and renormalizes (which artificially inflates low-confidence noise).
      Disallowed actions are strictly blocked at decision time by PhysicalCausalLogic.can_transition().
    """
    if raw_probs is None:
        return raw_probs, "NONE"

    probs = raw_probs.copy()
    idx_pick_red = fu.LABELS.index("pick_red")
    idx_place_red = fu.LABELS.index("place_red_out")
    idx_pick_blue = fu.LABELS.index("pick_blue")
    idx_place_blue = fu.LABELS.index("place_blue_in")

    # 1. Proximity bonus if wrist is physically near a box
    dist_red = get_box_wrist_distance(red_box or yolo_red, pose_landmarks, frame_shape)
    dist_blue = get_box_wrist_distance(blue_box or yolo_blue, pose_landmarks, frame_shape)

    dominant_color = "NONE"
    if dist_red < 0.22 and dist_red < dist_blue:
        dominant_color = "RED"
        probs[idx_pick_red] *= 1.25
        probs[idx_place_red] *= 1.25
    elif dist_blue < 0.22 and dist_blue < dist_red:
        dominant_color = "BLUE"
        probs[idx_pick_blue] *= 1.25
        probs[idx_place_blue] *= 1.25

    # 2. SOP Expected Action Prior Boost (gentle, non-distorting)
    if expected_action and expected_action in fu.LABELS:
        exp_idx = fu.LABELS.index(expected_action)
        probs[exp_idx] *= 1.20

    # DO NOT multiply disallowed classes by 0.05 and renormalize!
    # Renormalizing after zeroing/attenuating classes artificially inflates background noise.
    # PhysicalCausalLogic.can_transition() handles blocking invalid transitions directly.
    total = float(np.sum(probs))
    if total > 1e-6:
        probs = probs / total

    return probs, dominant_color


def process_frame(feat, motion, window, tar_model, stabilizer, spotter, now,
                  yolo_red=None, yolo_blue=None, red_box=None, blue_box=None,
                  pose_landmarks=None, frame_shape=None, expected_action=None,
                  containment=None):
    """
    Dual-Path Inference with 2.5D Physical Containment:
      1. Action Spotter: Monitors physical movement boundaries (start -> peak -> settle).
         When an action gesture completes, the exact motion burst is resampled to SEQ_LEN
         and classified ONCE with highest confidence.
      2. Rolling Window: Maintains smooth real-time telemetry, 5-frame stability latching,
         and auto-idle return.
    """
    window.append(feat)

    result = {
        "predicted_label": None,
        "confidence": 0.0,
        "state_changed": False,
        "current_state": stabilizer.current_state,
        "motion": motion,
        "status": "",
        "dominant_color": "NONE",
        "spotter_status": None,
        "probs": None,
    }

    # Path 1: Motion Action Spotter (Boundary-Aware Peak Scoring)
    triggered, action_clip, spotter_status = spotter.update(feat, motion, now)
    result["spotter_status"] = spotter_status

    if triggered and action_clip is not None:
        sp_label, sp_conf, sp_probs = predict_from_window(tar_model, action_clip)
        grounded_probs, dominant_color = ground_probs_with_yolo(
            sp_probs, yolo_red, yolo_blue, red_box, blue_box,
            pose_landmarks, frame_shape, stabilizer.causal_logic,
            expected_action=expected_action
        )
        changed, current_state, pred_class, conf, status = stabilizer.confirm_spotted_action(
            grounded_probs, now, expected_action=expected_action, containment=containment
        )
        spotter.last_spotted_action = pred_class
        spotter.last_spotted_conf = conf

        result.update({
            "predicted_label": pred_class,
            "confidence": conf,
            "state_changed": changed,
            "current_state": current_state,
            "status": status,
            "dominant_color": dominant_color,
            "probs": grounded_probs,
        })
        return result

    # Path 2: Continuous Rolling Window (Telemetry + Auto-Idle Settling + Stable Latching)
    if len(window) == SEQ_LEN:
        raw_label, raw_conf, raw_probs = predict_from_window(tar_model, list(window))

        # Ground probabilities with gentle physical & SOP guidance
        grounded_probs, dominant_color = ground_probs_with_yolo(
            raw_probs, yolo_red, yolo_blue, red_box, blue_box,
            pose_landmarks, frame_shape, stabilizer.causal_logic,
            expected_action=expected_action
        )

        changed, current_state, pred_class, conf, status = stabilizer.update(
            grounded_probs, motion, now,
            expected_action=expected_action,
            dominant_color=dominant_color,
            containment=containment
        )

        result.update({
            "predicted_label": pred_class,
            "confidence": conf,
            "state_changed": changed,
            "current_state": current_state,
            "status": status,
            "dominant_color": dominant_color,
            "probs": grounded_probs,
        })

    return result


class BoxTracker:
    """Maintains last-known box coordinates across brief occlusions without infinite feedback."""
    def __init__(self, max_missing=8):
        self.max_missing = max_missing
        self.last_boxes = {"red": None, "blue": None, "main": None}
        self.missing = {"red": 999, "blue": 999, "main": 999}

    def get_fallback(self, name, direct_box):
        if direct_box is not None:
            self.last_boxes[name] = direct_box
            self.missing[name] = 0
            return direct_box

        self.missing[name] += 1
        if self.missing[name] <= self.max_missing and self.last_boxes[name] is not None:
            return self.last_boxes[name]

        self.last_boxes[name] = None
        return None


# ===================================================================== #
# Feedback (Active Learning) & SOP Sequence Tracker
# ===================================================================== #
def save_feedback(window, feedback_type, label_id, label_name, notes=""):
    """
    Saves the current 48-frame window as training / review sample.
    """
    if len(window) < SEQ_LEN:
        return False, "Buffer warming up (wait 2s)"

    arr = np.array(list(window), dtype=np.float32)
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    ts = int(time.time())

    if feedback_type == "CORRECT":
        sub_dir = os.path.join(FEEDBACK_DIR, "correct")
        os.makedirs(sub_dir, exist_ok=True)
        fname = f"correct_{label_name}_{ts}.npy"
        np.save(os.path.join(sub_dir, fname), arr)
        play_sound(1200, 70)
        msg = f"FEEDBACK: SAVED AS CORRECT ({label_name})"

    elif feedback_type == "CORRECTION":
        sub_dir = os.path.join(FEEDBACK_DIR, "corrections")
        os.makedirs(sub_dir, exist_ok=True)
        fname = f"corrected_{label_name}_{ts}.npy"
        np.save(os.path.join(sub_dir, fname), arr)

        # Direct dataset augmentation so retrain_tar.py immediately benefits
        ds_dir = "dataset_advanced"
        if os.path.exists(ds_dir):
            import re
            pattern = re.compile(rf"^label_{label_id}_(\d+)\.npy$")
            existing = [int(pattern.match(f).group(1)) for f in os.listdir(ds_dir) if pattern.match(f)]
            next_idx = max(existing, default=-1) + 1
            ds_path = os.path.join(ds_dir, f"label_{label_id}_{next_idx}.npy")
            np.save(ds_path, arr)

        play_sound(1500, 90)
        msg = f"CORRECTED TO: {label_name} (Added to training data!)"

    elif feedback_type == "MISTAKE":
        sub_dir = os.path.join(FEEDBACK_DIR, "mistakes")
        os.makedirs(sub_dir, exist_ok=True)
        fname = f"mistake_{label_name}_{ts}.npy"
        np.save(os.path.join(sub_dir, fname), arr)
        play_sound(550, 110)
        msg = f"FLAGGED AS MISTAKE: {label_name}"
    else:
        msg = "Feedback saved"

    # Append to feedback_log.csv
    log_file = os.path.join(FEEDBACK_DIR, "feedback_log.csv")
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "feedback_type", "label_id", "label_name", "notes"])
        writer.writerow([ts, feedback_type, label_id, label_name, notes])

    return True, msg


SOP_STEPS = [
    ("open_box", "1. Open Box"),
    ("pick_red", "2. Pick Red"),
    ("place_red_out", "3. Place Red Out"),
    ("pick_blue", "4. Pick Blue"),
    ("place_blue_in", "5. Place Blue In"),
    ("close_box", "6. Close Box"),
]


class SOPTracker:
    """Tracks SOP execution through the 6 core packaging steps with fast-forward tolerance."""
    def __init__(self):
        self.current_step = 0
        self.completed = [False] * len(SOP_STEPS)
        self.cycle_count = 0
        self.cycle_complete_time = 0.0

    @property
    def expected_action(self):
        if self.current_step < len(SOP_STEPS):
            return SOP_STEPS[self.current_step][0]
        return None

    @property
    def expected_action_title(self):
        if self.current_step < len(SOP_STEPS):
            return SOP_STEPS[self.current_step][1]
        return "Cycle Complete"

    def update(self, state, now):
        if state is None or state == "idle":
            return

        expected = self.expected_action

        # Standard sequential advance
        if state == expected:
            self.completed[self.current_step] = True
            play_sound(1100, 60)
            if self.current_step < len(SOP_STEPS) - 1:
                self.current_step += 1
            else:
                self.cycle_count += 1
                self.cycle_complete_time = now
                play_sound(1400, 90)
                self.reset()
            return

        # Automatic new cycle start on re-opening box after completing previous cycle
        if state == "open_box" and self.current_step > 0 and (now - self.cycle_complete_time) > 1.5:
            self.reset()
            self.completed[0] = True
            self.current_step = 1
            play_sound(1100, 60)

    def reset(self):
        self.current_step = 0
        self.completed = [False] * len(SOP_STEPS)

    def draw(self, frame, now):
        h, w = frame.shape[:2]
        box_w, box_h = 240, 180
        x0, y0 = w - box_w - 10, 10

        # Semi-transparent backdrop
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (80, 80, 80), 1)

        # Header
        cv2.putText(frame, f"SOP FLOW  (Cycles: {self.cycle_count})", (x0 + 12, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

        # Step checklist
        for i, (action, title) in enumerate(SOP_STEPS):
            sy = y0 + 46 + i * 22
            if self.completed[i]:
                icon = "[v]"
                color = (0, 255, 0)
            elif i == self.current_step:
                icon = "[>]"
                color = (0, 255, 255)
            else:
                icon = "[ ]"
                color = (130, 130, 130)

            text = f"{icon} {title}"
            cv2.putText(frame, text, (x0 + 10, sy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # Big Cycle Complete Banner
        if now - self.cycle_complete_time < 2.2:
            cx, cy = w // 2, h // 2
            cv2.rectangle(frame, (cx - 200, cy - 40), (cx + 200, cy + 40), (0, 180, 0), -1)
            cv2.putText(frame, "CYCLE COMPLETE! (+1)", (cx - 165, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


# ===================================================================== #
# Main camera loop
# ===================================================================== #
def main():
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
    except AttributeError:
        raise RuntimeError(
            "This mediapipe install doesn't expose the legacy `mp.solutions` API. "
            "Pin an older version: pip install mediapipe==0.10.9"
        )

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

    tar_model = load_tar_model(MODEL_PATH)
    yolo_model = load_yolo(YOLO_MODEL_PATH)
    stabilizer = DecisionStabilizer(CONFIDENCE_THRESHOLD, STABILITY_WINDOW, COOLDOWN_SEC)
    spotter = MotionActionSpotter()
    box_tracker = BoxTracker(max_missing=8)
    containment = GeometricContainmentEngine(buffer_size=8)
    sop_tracker = SOPTracker()

    window = deque(maxlen=SEQ_LEN)
    prev_feat = None
    prev_base = None

    feedback_banner = ""
    feedback_banner_color = (0, 255, 0)
    feedback_banner_time = 0.0

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    print(f"[Realtime] confidence>={CONFIDENCE_THRESHOLD}  stability={STABILITY_WINDOW} frames  "
          f"cooldown={COOLDOWN_SEC}s  motion_threshold={MOTION_THRESHOLD}")
    print("[Controls] [C]=Mark Correct  |  [0-6]=Correct Label  |  [X]=Flag Wrong  |  [Q]=Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from camera (ret=False). Exiting.")
            break

        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        hand_res = hands.process(rgb)

        # --- YOLO detections (deep learning box detector) ---
        yolo_detections = run_yolo(yolo_model, frame)
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

        # 332-dim features: 166 base (pose+hands+objects+distances+lid) + 166 velocity
        base, det_flags, (red_box, blue_box, main_box), edge_debug, lid_score = extract_base_features(
            pose_res, hand_res, frame, override_boxes=(yolo_red, yolo_blue, yolo_main)
        )

        # Smooth brief detection dropouts (up to 8 frames) without infinite latching
        red_box = box_tracker.get_fallback("red", red_box)
        blue_box = box_tracker.get_fallback("blue", blue_box)
        main_box = box_tracker.get_fallback("main", main_box)

        # Update 2.5D Geometric Containment Engine
        containment.update(
            red_box, blue_box, main_box,
            pose_res.pose_landmarks if pose_res else None,
            frame.shape
        )

        velocity = base - prev_base if prev_base is not None else np.zeros(BASE_DIM, dtype=np.float32)
        prev_base = base.copy()
        feat = np.concatenate([base, velocity])

        motion = compute_motion(prev_feat, feat)
        prev_feat = feat

        # --- Draw bounding boxes ---
        if red_box is not None:
            rx, ry, rw, rh = red_box["rect"]
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

        if blue_box is not None:
            bx, by, bw, bh = blue_box["rect"]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

        if main_box is not None:
            mx, my, mw, mh = main_box["rect"]
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)

        # Draw 2.5D container volume wireframe
        containment.draw(frame, main_box)

        # Draw YOLO detections
        for det in yolo_detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # --- Inference with Motion Spotter + gentle grounding & SOP expected step ---
        result = process_frame(
            feat, motion, window, tar_model, stabilizer, spotter, now,
            yolo_red=yolo_red, yolo_blue=yolo_blue,
            red_box=red_box, blue_box=blue_box,
            pose_landmarks=pose_res.pose_landmarks if pose_res else None,
            frame_shape=frame.shape,
            expected_action=sop_tracker.expected_action,
            containment=containment
        )
        if result["state_changed"] and result['current_state'] != "idle":
            print(f"[STATE] -> {result['current_state']}  (confidence {result['confidence']:.2f})")
            sop_tracker.update(result['current_state'], now)

        # --- Minimal HUD: Left Side (State + Diagnostic Telemetry) ---
        state_text = result['current_state'] or "idle"
        is_idle = (state_text == "idle")
        state_color = (180, 180, 180) if is_idle else (0, 255, 0)
        cv2.putText(frame, f"State: {state_text}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, state_color, 2)

        exp_title = sop_tracker.expected_action_title or "Complete"
        cv2.putText(frame, f"Next: {exp_title}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        conf_pct = int(result['confidence'] * 100)
        stab = len(stabilizer.recent)
        pred_label = result.get('predicted_label') or "..."
        status_text = result.get('status', '')
        debug_line = f"Pred: {pred_label} ({conf_pct}%) | Latch: {stab}/{STABILITY_WINDOW} | {status_text}"
        cv2.putText(frame, debug_line, (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

        # Physical causal state readout with active object focus
        causal_line = stabilizer.causal_logic.get_status_summary(stabilizer.dominant_color)
        r_ok = "YES" if red_box is not None else "NO"
        b_ok = "YES" if blue_box is not None else "NO"
        m_ok = "YES" if main_box is not None else "NO"
        det_str = f"[R: {r_ok}] [B: {b_ok}] [M: {m_ok}]"
        cv2.putText(frame, f"{causal_line} | {det_str}", (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 255), 1)

        # 2.5D Geometric containment status readout
        cv2.putText(frame, containment.get_summary(), (10, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 100), 1)

        # Motion Spotter Telemetry Widget (Boundary-Aware Tracking)
        spot_st = result.get("spotter_status")
        if spot_st:
            sp_mode = spot_st["spotter_state"]
            sp_frames = spot_st["motion_frames"]
            sp_mot = spot_st["smooth_motion"]

            if sp_mode == "TRACKING":
                txt = f"[ACTIVE GESTURE] Frames: {sp_frames} | Velocity: {sp_mot:.3f}"
                cv2.putText(frame, txt, (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 165, 255), 1)
                bar_w = min(150, int((sp_mot / 0.04) * 150))
                cv2.rectangle(frame, (10, 155), (10 + bar_w, 161), (0, 165, 255), -1)
                cv2.rectangle(frame, (10, 155), (160, 161), (80, 80, 80), 1)
            elif sp_mode == "COOLDOWN":
                txt = f"[SPOTTER] Action Scored! Settling..."
                cv2.putText(frame, txt, (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1)
            else:
                txt = "[SPOTTER] Ready (Hands Calm)"
                cv2.putText(frame, txt, (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1)

        # Key hints at bottom
        h_frame = frame.shape[0]
        hints = "[C] Correct  |  [0-6] Fix Label  |  [R] Reset Cycle  |  [X] Flag Mistake  |  [Q] Quit"
        cv2.putText(frame, hints, (10, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        # --- Draw SOP Step Tracker (Top Right) ---
        sop_tracker.draw(frame, now)

        # --- Draw Feedback Alert Banner ---
        if now - feedback_banner_time < 2.5:
            fh, fw = frame.shape[:2]
            banner_w = min(560, fw - 40)
            bx0 = (fw - banner_w) // 2
            by0 = fh - 65
            cv2.rectangle(frame, (bx0, by0), (bx0 + banner_w, by0 + 36), (20, 20, 20), -1)
            cv2.rectangle(frame, (bx0, by0), (bx0 + banner_w, by0 + 36), feedback_banner_color, 2)
            cv2.putText(frame, feedback_banner, (bx0 + 15, by0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, feedback_banner_color, 1)

        cv2.imshow("Realtime HAR", frame)

        # --- Keyboard Interactions ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('r'), ord('R')):
            # Reset cycle and physical state to start
            stabilizer.causal_logic.reset()
            containment.reset()
            stabilizer.current_state = "idle"
            spotter.state = spotter.STATE_IDLE
            spotter.buffer.clear()
            sop_tracker.reset()
            feedback_banner = "RESET: Cycle & physical logic reset to start"
            feedback_banner_color = (0, 255, 255)
            feedback_banner_time = time.time()
            print("[RESET] Physical causal state, 2.5D containment, and SOP reset.")

        elif key in (ord('c'), ord('C')):
            # User marks current sequence as CORRECT
            curr_label = result.get('predicted_label') or stabilizer.current_state or "idle"
            label_id = fu.LABELS.index(curr_label) if curr_label in fu.LABELS else 0
            ok, msg = save_feedback(window, "CORRECT", label_id, curr_label)
            feedback_banner = msg
            feedback_banner_color = (0, 255, 0)
            feedback_banner_time = time.time()
            print(f"[FEEDBACK] {msg}")

        elif key in (ord('x'), ord('X')):
            # User marks current sequence as WRONG / MISTAKE
            curr_label = result.get('predicted_label') or "unknown"
            label_id = fu.LABELS.index(curr_label) if curr_label in fu.LABELS else 0
            ok, msg = save_feedback(window, "MISTAKE", label_id, curr_label)
            feedback_banner = msg
            feedback_banner_color = (0, 0, 255)
            feedback_banner_time = time.time()
            print(f"[FEEDBACK] {msg}")

        elif ord('0') <= key <= ord('6'):
            # User presses a number (0-6) to tell the exact right label
            target_id = key - ord('0')
            target_name = fu.LABELS[target_id]
            stabilizer.current_state = target_name
            stabilizer.causal_logic.apply_transition(target_name)
            sop_tracker.update(target_name, now)
            ok, msg = save_feedback(window, "CORRECTION", target_id, target_name,
                                    notes=f"model_said_{result.get('predicted_label')}")
            feedback_banner = msg
            feedback_banner_color = (0, 255, 255)
            feedback_banner_time = time.time()
            print(f"[FEEDBACK] {msg}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()