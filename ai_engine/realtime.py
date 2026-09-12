"""
Live inference: camera -> pose/hands -> TAR model (state classification)
                        -> YOLO (auxiliary object detection)
                        -> stabilized state decision
qq
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
import sys
import time
import csv
import threading
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

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

MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "best_tar_model.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "models/best_tar_model.pth" if os.path.exists("models/best_tar_model.pth") else "best_tar_model.pth"

YOLO_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "yolo_boxes.pt")
if not os.path.exists(YOLO_MODEL_PATH):
    YOLO_MODEL_PATH = "models/yolo_boxes.pt" if os.path.exists("models/yolo_boxes.pt") else "yolo_boxes.pt"

CAMERA_ID = 0
VIDEO_OUTPUT_DIR = "videos"

CONFIDENCE_THRESHOLD = 0.45          # Fast, responsive threshold for valid actions
STABILITY_WINDOW = 3                 # Ultra-fast 3-frame (~100ms) intentional action confirmation
COOLDOWN_SEC = 0.40                  # Fast 0.40s cooldown between distinct steps
CYCLE_COOLDOWN_SEC = 1.0             # 1.0s cooldown after closing box
MOTION_THRESHOLD = 0.009             # Sensitive movement trigger
EMA_ALPHA = 0.78                     # Highly responsive probability updating (instant reaction)
DWELL_TIME_SEC = 0.60                # 0.60s visual dwell so state doesn't freeze the screen
USE_FSM = True                       # Physical Causal Logic gate
IDLE_LABEL = fu.LABELS[0]            # "idle" - exempt from motion gate
FEEDBACK_DIR = "feedback_data"

# Per-class confidence thresholds calibrated for instant recognition:
CLASS_CONF_THRESHOLDS = {
    "idle": 0.35,
    "open_box": 0.46,
    "pick_red": 0.48,
    "place_red_out": 0.46,
    "pick_blue": 0.46,
    "place_blue_in": 0.45,
    "close_box": 0.45,

}


def ts_now_str():
    """Returns current local time as HH:MM:SS.mmm string."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def ts_print(msg):
    """Prints message to stdout with millisecond-precision timestamp prefix."""
    print(f"[{ts_now_str()}] {msg}")


class SessionVideoRecorder:
    """
    Records clean raw camera frames into an MP4 file for dataset training,
    while concurrently writing a frame-by-frame timeline CSV and auto-generating
    action annotations compatible with VIDEO.PY.
    """
    def __init__(self, output_dir=VIDEO_OUTPUT_DIR, fps=25.0):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.fps = fps
        self.writer = None
        self.video_path = None
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time = 0.0
        self.segments = []
        self.current_segment_label = "idle"
        self.current_segment_start = 0

    def start(self, frame_w, frame_h, fps=None):
        if fps and fps > 0:
            self.fps = fps
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.output_dir, f"session_{ts_tag}.mp4")
        self.csv_path = os.path.join(self.output_dir, f"session_{ts_tag}_timeline.csv")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (frame_w, frame_h))
        if not self.writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_path = os.path.join(self.output_dir, f"session_{ts_tag}.avi")
            self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (frame_w, frame_h))

        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame_idx", "timestamp", "elapsed_sec",
            "fsm_state", "predicted_label", "confidence", "fsm_status"
        ])

        self.is_recording = True
        self.frame_count = 0
        self.start_time = time.time()
        self.segments = []
        self.current_segment_label = "idle"
        self.current_segment_start = 0
        ts_print(f"[REC] Recording session video to: {self.video_path} ({self.fps:.1f} FPS)")
        ts_print(f"[REC] Timeline CSV metadata: {self.csv_path}")

    def write_frame(self, raw_clean_frame, fsm_state, pred_label, confidence, status):
        if not self.is_recording:
            return

        # Write clean camera frame for ML dataset training
        if self.writer is not None:
            self.writer.write(raw_clean_frame)

        self.frame_count += 1

        elapsed = time.time() - self.start_time
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if self.csv_writer:
            self.csv_writer.writerow([
                self.frame_count, ts_str, f"{elapsed:.3f}",
                fsm_state or "idle", pred_label or "idle", f"{confidence:.3f}", status or ""
            ])

        # Track timeline segments for auto-generating annotations for VIDEO.PY
        active_label = fsm_state if (fsm_state and fsm_state != "idle") else "idle"
        if active_label != self.current_segment_label:
            if self.frame_count > self.current_segment_start:
                self.segments.append((self.current_segment_start, self.frame_count, self.current_segment_label))
            self.current_segment_label = active_label
            self.current_segment_start = self.frame_count

    def toggle(self, frame_w, frame_h, fps=None):
        if self.is_recording:
            self.is_recording = False
            ts_print(f"[REC] Recording PAUSED at frame {self.frame_count}")
        else:
            if self.writer is None:
                self.start(frame_w, frame_h, fps)
            else:
                self.is_recording = True
                ts_print(f"[REC] Recording RESUMED at frame {self.frame_count}")

    def stop(self):
        if self.writer is not None:
            if self.frame_count > self.current_segment_start:
                self.segments.append((self.current_segment_start, self.frame_count, self.current_segment_label))

            self.writer.release()
            self.writer = None

            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None

            elapsed = time.time() - self.start_time if self.start_time > 0 else 0.0
            ts_print(f"[REC] Video saved: {self.video_path} ({self.frame_count} frames, {elapsed:.1f}s)")
            ts_print(f"[REC] Timeline CSV: {self.csv_path}")

            # Save auto-annotation snippet for VIDEO.PY
            annot_path = self.video_path.rsplit(".", 1)[0] + "_annotations.py"
            try:
                with open(annot_path, "w", encoding="utf-8") as f:
                    f.write("# Auto-generated annotations from realtime session\n")
                    f.write(f"# Video: {os.path.basename(self.video_path)}\n")
                    f.write(f"# Total frames: {self.frame_count}\n\n")
                    f.write("annotations = [\n")
                    for start_f, end_f, lbl in self.segments:
                        f.write(f'    ({start_f}, {end_f}, "{lbl}"),\n')
                    f.write("]\n")
                ts_print(f"[REC] Annotations generated: {annot_path} (Ready for VIDEO.PY training!)")
            except Exception as e:
                ts_print(f"[REC WARN] Could not save annotations: {e}")

            self.is_recording = False


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

    def can_transition(self, action, containment=None, red_box=None, blue_box=None, main_box=None,
                       dist_red=999.0, dist_blue=999.0, dist_main=999.0):
        if action == "idle":
            return True, "ok"

        if action == "open_box":
            if self.box_open:
                return False, "box already open"
            if main_box is None:
                return False, "no box detected in scene"
            if dist_main > 0.45:
                return False, "hands not near box"
            return True, "ok"

        if action == "pick_red":
            if not self.box_open:
                return False, "box must be open before picking red"
            if self.red_placed_out:
                return False, "red already placed out"
            if self.red_picked:
                return False, "red already in hand"
            # Require red object presence: cannot pick red if no red object exists in camera view
            if red_box is None and (containment is None or containment.red_status == "UNKNOWN") and not self.red_picked:
                return False, "no red object detected in camera view"
            if red_box is not None and dist_red > 0.45 and (containment is None or not containment.red_is_held):
                return False, "hand not near red object"
            return True, "ok"

        if action == "place_red_out":
            if not self.box_open:
                return False, "box must be open"
            if self.red_placed_out:
                return False, "red already placed out"
            if not self.red_picked:
                return False, "red must be picked up first"
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
            # If blue_box is visible, verify hand reach. If occluded by hand grasp, allow pick!
            if blue_box is not None and dist_blue > 0.60 and (containment is None or not containment.blue_is_held):
                return False, "hand not near blue object"
            return True, "ok"

        if action == "place_blue_in":
            if not self.box_open:
                return False, "box must be open to place blue inside"
            if self.blue_placed_in:
                return False, "blue already placed inside"
            if not self.blue_picked:
                return False, "blue must be picked up first"
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
            if main_box is None:
                return False, "no box detected in scene"
            if dist_main > 0.45:
                return False, "hands not near box"
            if containment and containment.red_status == "INSIDE" and not self.red_placed_out:
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

    apply = apply_transition

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


# Canonical step order for the procedure
# (Source: Context-Aware Assistance paper — MTM-style step taxonomy)
STEP_ORDER = [
    "idle",
    "open_box",
    "pick_red",
    "place_red_out",
    "pick_blue",
    "place_blue_in",
    "close_box",
]


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
                 cycle_cooldown_sec=CYCLE_COOLDOWN_SEC,
                 use_fsm=USE_FSM,
                 ema_alpha=EMA_ALPHA,
                 dwell_sec=DWELL_TIME_SEC):
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        self.cooldown_sec = cooldown_sec
        self.cycle_cooldown_sec = cycle_cooldown_sec
        self.use_fsm = use_fsm
        self.ema_alpha = ema_alpha
        self.dwell_sec = dwell_sec

        self.smoothed_probs = None
        self.recent = deque(maxlen=stability_window)
        self.current_state = "idle"
        self.last_action_state = "idle"   # tracks last confirmed NON-IDLE step for skip-alert
        self.last_change_time = 0.0
        self.dwell_until = 0.0
        self.last_rejected_reason = ""
        self.dominant_color = "NONE"
        self.causal_logic = PhysicalCausalLogic()
        # Skip-alert throttle: fire at most one beep per 3s per skip event
        # (Source: Context-Aware Assistance paper — alert on step skip)
        self.last_skip_alert_time = 0.0
        self.skip_alert_cooldown = 3.0

        # FSM Violation / Rejection terminal throttle
        self.last_fsm_block_time = 0.0
        self.last_fsm_blocked_action = ""
        self.fsm_block_cooldown = 1.5

        # Sustained mistake latch (requires 3 consecutive frames of real intentional action >= 0.52)
        self.mistake_candidate = ""
        self.mistake_consecutive = 0

        # Physics-Assisted Empty Hand Release Counters
        self.release_counter_red = 0
        self.release_counter_blue = 0

        # Motion-triggered and timed SOP transitions (3 to 4 sec)
        self.motion_start_time = 0.0          # Tracks when motion started from idle
        self.blue_in_confirm_time = 0.0       # Tracks when place_blue_in occurred

    def check_operator_mistake(self, raw_label, raw_conf, expected_action, now,
                                containment=None, red_box=None, blue_box=None, main_box=None,
                                dist_red=999.0, dist_blue=999.0, dist_main=999.0):
        """
        Detects deliberate out-of-order operator mistakes based on raw neural inference BEFORE attenuation.
        Returns: (is_mistake: bool, detected_action: str, reason: str)
        """
        if raw_label is None or raw_label == "idle" or raw_conf < 0.52:
            self.mistake_candidate = ""
            self.mistake_consecutive = 0
            return False, "", ""

        # Trailing gesture inertia & past step filter:
        # Never flag actions that were already completed earlier in this cycle (e.g. open_box, pick_red)
        # when hand retracts or reaches across the scene.
        if raw_label == self.last_action_state:
            self.mistake_candidate = ""
            self.mistake_consecutive = 0
            return False, "", ""
        if self.last_action_state == "place_red_out" and raw_label in ("pick_red", "open_box"):
            self.mistake_candidate = ""
            self.mistake_consecutive = 0
            return False, "", ""
        if self.last_action_state == "place_blue_in" and raw_label in ("pick_blue", "place_red_out", "open_box"):
            self.mistake_candidate = ""
            self.mistake_consecutive = 0
            return False, "", ""

        # SOP Sequence Directionality Filter:
        # A true forward mistake is jumping ahead to a FUTURE step before completing earlier steps
        # (e.g. attempting pick_blue or close_box before earlier steps are done).
        # Actions that are already past (e.g. open_box while at step 4) are physically impossible artifacts, not deliberate operator errors!
        sop_step_names = ["open_box", "pick_red", "place_red_out", "pick_blue", "place_blue_in", "close_box"]
        if expected_action in sop_step_names and raw_label in sop_step_names:
            exp_idx = sop_step_names.index(expected_action)
            raw_idx = sop_step_names.index(raw_label)
            if raw_idx < exp_idx:
                # This step is already in the past! Hand is merely reaching, not trying to re-open
                self.mistake_candidate = ""
                self.mistake_consecutive = 0
                return False, "", ""

        # If the action matches what the procedure expects next, it's correct
        if expected_action and raw_label == expected_action:
            self.mistake_candidate = ""
            self.mistake_consecutive = 0
            return False, "", ""

        # Physical Proximity Guard for close_box mistake detection:
        # If the operator is merely reaching forward for blue on the desk, the hand is FAR from the box lid.
        # Closing the box requires hands to physically reach/touch the main box (dist_main <= 0.35).
        # Without hands at the box, reaching forward cannot be a deliberate 'close_box' mistake!
        if raw_label == "close_box":
            if dist_main > 0.35 or raw_conf < 0.65:
                self.mistake_candidate = ""
                self.mistake_consecutive = 0
                return False, "", ""

        # General confidence threshold for forward mistakes (intentional act >= 0.62)
        if raw_conf < 0.62:
            self.mistake_candidate = ""
            self.mistake_consecutive = 0
            return False, "", ""

        # Check physical/procedural causal validity
        allowed, reason = self.causal_logic.can_transition(
            raw_label, containment=containment,
            red_box=red_box, blue_box=blue_box, main_box=main_box,
            dist_red=dist_red, dist_blue=dist_blue, dist_main=dist_main
        )

        if not allowed:
            # Sustained latch: require 4 consecutive frames (~130ms) of intentional gesture
            if raw_label == self.mistake_candidate:
                self.mistake_consecutive += 1
            else:
                self.mistake_candidate = raw_label
                self.mistake_consecutive = 1

            if self.mistake_consecutive >= 4:
                is_new_action = (raw_label != self.last_fsm_blocked_action)
                cooldown_ok = (now - self.last_fsm_block_time) > self.fsm_block_cooldown
                if is_new_action or cooldown_ok:
                    self.last_fsm_block_time = now
                    self.last_fsm_blocked_action = raw_label
                    return True, raw_label, reason
        else:
            self.mistake_candidate = ""
            self.mistake_consecutive = 0

        return False, "", ""

    def update(self, raw_probs, motion, now, expected_action=None, dominant_color="NONE", containment=None,
               red_box=None, blue_box=None, main_box=None, dist_red=999.0, dist_blue=999.0, dist_main=999.0):
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

        # 3b. Physics-Assisted Empty Hand Release Rule:
        # Rule A: If Red was picked and hands are now empty / red released on desk -> deduce place_red_out!
        if self.use_fsm and self.causal_logic.red_picked and not self.causal_logic.red_placed_out:
            if containment and containment.red_status in ("OUTSIDE", "UNKNOWN") and not containment.red_is_held:
                self.release_counter_red += 1
                if self.release_counter_red >= 4:  # 4 stable frames of empty hand (~0.15s)
                    self.release_counter_red = 0
                    candidate = "place_red_out"
                    self.current_state = candidate
                    self.last_change_time = now
                    self.dwell_until = now + self.dwell_sec
                    self.last_action_state = candidate
                    self.recent.clear()
                    self.causal_logic.apply_transition(candidate)
                    self.last_rejected_reason = "physics: red released on desk"
                    ts_print(f"[PHYSICS EVENT] -> place_red_out (Hands empty, red released on desk)")
                    return True, self.current_state, candidate, 0.95, "physics: red released on desk"
            else:
                self.release_counter_red = 0

        # Rule B: If Blue was picked and hands are now empty / blue released in container -> deduce place_blue_in!
        if self.use_fsm and self.causal_logic.blue_picked and not self.causal_logic.blue_placed_in:
            if containment and containment.blue_status in ("INSIDE", "UNKNOWN") and not containment.blue_is_held:
                self.release_counter_blue += 1
                if self.release_counter_blue >= 4:  # 4 stable frames of empty hand (~0.15s)
                    self.release_counter_blue = 0
                    candidate = "place_blue_in"
                    self.current_state = candidate
                    self.last_change_time = now
                    self.dwell_until = now + self.dwell_sec
                    self.last_action_state = candidate
                    self.recent.clear()
                    self.causal_logic.apply_transition(candidate)
                    self.blue_in_confirm_time = now
                    self.last_rejected_reason = "physics: blue placed in box"
                    ts_print(f"[PHYSICS EVENT] -> place_blue_in (Hands empty, blue placed in box)")
                    return True, self.current_state, candidate, 0.95, "physics: blue placed in box"
            else:
                self.release_counter_blue = 0

        # 3c. Automatic Motion-Triggered Open Box (Within 3 to 4 seconds of movement)
        if self.use_fsm and not self.causal_logic.box_open and (expected_action in ("open_box", None)):
            if motion >= (MOTION_THRESHOLD * 0.70):
                if self.motion_start_time == 0.0:
                    self.motion_start_time = now
                elif (now - self.motion_start_time) >= 3.5:
                    self.motion_start_time = 0.0
                    candidate = "open_box"
                    self.current_state = candidate
                    self.last_change_time = now
                    self.dwell_until = now + self.dwell_sec
                    self.last_action_state = candidate
                    self.recent.clear()
                    self.causal_logic.apply_transition(candidate)
                    self.last_rejected_reason = "timed: 3.5s motion confirmed open_box"
                    ts_print(f"[PROCEDURE EVENT] -> open_box (3.5s motion detected, container opened)")
                    return True, self.current_state, candidate, 0.95, "timed: 3.5s motion confirmed open_box"
            else:
                self.motion_start_time = 0.0

        # 3d. Automatic Close Box After Blue Placed In (Within 3 to 4 seconds)
        if self.use_fsm and self.causal_logic.blue_placed_in and self.causal_logic.box_open:
            if self.blue_in_confirm_time == 0.0:
                self.blue_in_confirm_time = now
            elif (now - self.blue_in_confirm_time) >= 3.5:
                self.blue_in_confirm_time = 0.0
                candidate = "close_box"
                self.current_state = candidate
                self.last_change_time = now
                self.dwell_until = now + self.dwell_sec
                self.last_action_state = candidate
                self.recent.clear()
                self.causal_logic.apply_transition(candidate)
                self.last_rejected_reason = "timed: 3.5s after blue in confirmed close_box"
                ts_print(f"[PROCEDURE EVENT] -> close_box (3.5s after blue in, container closed)")
                return True, self.current_state, candidate, 0.95, "timed: 3.5s after blue in confirmed close_box"

        # 4. Motion Gate: non-idle actions require actual physical movement
        if predicted_class != IDLE_LABEL and motion < MOTION_THRESHOLD:
            self.last_rejected_reason = "low motion"
            return False, self.current_state, predicted_class, confidence, self.last_rejected_reason

        # 5. Cooldown Gate: enforce pause between actions so hand retraction cannot trigger
        active_cooldown = self.cycle_cooldown_sec if self.last_action_state == "close_box" else self.cooldown_sec
        if self.last_change_time > 0 and (now - self.last_change_time) < active_cooldown:
            rem = active_cooldown - (now - self.last_change_time)
            self.last_rejected_reason = f"cooldown ({rem:.1f}s)"
            return False, self.current_state, predicted_class, confidence, self.last_rejected_reason

        # 6. Physical Causal Reality Gate: block actions that are physically impossible according to FSM
        if self.use_fsm:
            allowed, reason = self.causal_logic.can_transition(
                predicted_class, containment=containment,
                red_box=red_box, blue_box=blue_box, main_box=main_box,
                dist_red=dist_red, dist_blue=dist_blue, dist_main=dist_main
            )
            if not allowed:
                self.last_rejected_reason = f"blocked: {reason}"

                # Trailing gesture inertia filter: NEVER flag trailing actions from recently completed steps!
                # e.g., user just completed pick_red, hand is moving to desk while holding red -> model still sees pick_red.
                # e.g., user just placed red out, hand is leaving red -> model still sees pick_red.
                # e.g., user just opened box, hand retracts downwards -> model momentarily flickers close_box.
                is_trailing_past_action = (
                    predicted_class == self.last_action_state or
                    (self.last_action_state == "open_box" and predicted_class == "close_box") or
                    (self.last_action_state == "pick_red" and predicted_class in ("pick_red", "open_box")) or
                    (self.last_action_state == "place_red_out" and predicted_class == "pick_red") or
                    (self.last_action_state == "pick_blue" and predicted_class in ("pick_blue", "place_red_out")) or
                    (self.last_action_state == "place_blue_in" and predicted_class == "pick_blue")
                )

                sop_step_names = ["open_box", "pick_red", "place_red_out", "pick_blue", "place_blue_in", "close_box"]
                is_past_step = False
                if expected_action in sop_step_names and predicted_class in sop_step_names:
                    is_past_step = (sop_step_names.index(predicted_class) < sop_step_names.index(expected_action))

                # An operator mistake requires high confidence (intentional act >= 0.68), cannot be trailing inertia or past step
                is_real_mistake_candidate = (
                    confidence >= 0.68 and
                    predicted_class != "idle" and
                    predicted_class != expected_action and
                    not is_trailing_past_action and
                    not is_past_step
                )

                if is_real_mistake_candidate:
                    if predicted_class == self.mistake_candidate:
                        self.mistake_consecutive += 1
                    else:
                        self.mistake_candidate = predicted_class
                        self.mistake_consecutive = 1

                    # Require at least 3 consecutive high-confidence frames (~100ms) to trigger a true mistake alarm
                    if self.mistake_consecutive >= 3:
                        is_new_action = (predicted_class != self.last_fsm_blocked_action)
                        cooldown_ok = (now - self.last_fsm_block_time) > self.fsm_block_cooldown
                        if is_new_action or cooldown_ok:
                            exp_str = f" | Expected: '{expected_action}'" if expected_action else ""
                            status_summary = self.causal_logic.get_status_summary(self.dominant_color)
                            ts_print(f"[⚠️ FSM MISTAKE BLOCKED] Detected: '{predicted_class}' (conf {confidence:.2f}) -> REJECTED: {reason}{exp_str} [{status_summary}]")
                            self.last_fsm_block_time = now
                            self.last_fsm_blocked_action = predicted_class
                            return False, self.current_state, predicted_class, confidence, f"MISTAKE: {reason}"
                else:
                    self.mistake_candidate = ""
                    self.mistake_consecutive = 0

                return False, self.current_state, predicted_class, confidence, self.last_rejected_reason
            else:
                self.mistake_candidate = ""
                self.mistake_consecutive = 0

        # 6b. Step-Skip Alert: detect when model jumps over a mandatory procedure step
        if self.use_fsm and predicted_class != "idle" and predicted_class in STEP_ORDER:
            ref_state = self.last_action_state  # true physical step progress
            cur_idx = STEP_ORDER.index(ref_state) if ref_state in STEP_ORDER else 0
            pred_idx_step = STEP_ORDER.index(predicted_class)
            # A skip is when the proposed step is 2+ positions ahead of physical progress
            if pred_idx_step > cur_idx + 1 and (now - self.last_skip_alert_time) > self.skip_alert_cooldown:
                expected_next = STEP_ORDER[cur_idx + 1] if cur_idx + 1 < len(STEP_ORDER) else "(end)"
                skip_msg = (f"[FSM STEP SKIP] Detected: '{predicted_class}' (conf {confidence:.2f}) | "
                            f"Last confirmed: '{ref_state}' | Expected next: '{expected_next}'")
                ts_print(skip_msg)
                play_sound(freq=400, dur=180)
                self.last_skip_alert_time = now

        # 7. Confidence Gate: adaptive threshold per class and SOP expectation
        base_thresh = CLASS_CONF_THRESHOLDS.get(predicted_class, self.confidence_threshold)
        req_thresh = base_thresh if (expected_action and predicted_class == expected_action) else max(0.55, base_thresh + 0.08)
        if confidence < req_thresh:
            self.last_rejected_reason = f"low conf ({confidence:.2f}<{req_thresh:.2f})"
            return False, self.current_state, predicted_class, confidence, self.last_rejected_reason

        # 8. Adaptive Fast Stability Latch:
        # Require sustained intentional action: 2 out of last 3 frames for expected (~60ms!), 3 out of last 4 for unprompted
        self.recent.append(predicted_class)
        is_expected = bool(expected_action and predicted_class == expected_action)
        required_matches = 2 if is_expected else 3
        sub_window = 3 if is_expected else 4
        recent_slice = list(self.recent)[-sub_window:]
        match_cnt = recent_slice.count(predicted_class)
        if len(recent_slice) < sub_window or match_cnt < required_matches:
            self.last_rejected_reason = f"latching ({match_cnt}/{required_matches})"
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
            self.last_action_state = candidate   # track physical SOP progress (never reset by idle-settle)
        self.causal_logic.apply_transition(candidate)
        if candidate == "place_blue_in":
            self.blue_in_confirm_time = now
        elif candidate == "close_box":
            self.blue_in_confirm_time = 0.0
            self.motion_start_time = 0.0
        elif candidate == "open_box":
            self.motion_start_time = 0.0
        self.last_rejected_reason = "accepted"
        return True, self.current_state, candidate, confidence, self.last_rejected_reason

    def confirm_spotted_action(self, grounded_probs, now, expected_action=None, containment=None,
                               red_box=None, blue_box=None, main_box=None,
                               dist_red=999.0, dist_blue=999.0, dist_main=999.0):
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
        active_cooldown = self.cycle_cooldown_sec if self.last_action_state == "close_box" else self.cooldown_sec
        if self.last_change_time > 0 and (now - self.last_change_time) < active_cooldown:
            rem = active_cooldown - (now - self.last_change_time)
            self.last_rejected_reason = f"cooldown ({rem:.1f}s)"
            return False, self.current_state, cand, conf, self.last_rejected_reason

        # 2. Strict Confidence Gate: adaptive threshold per class and SOP expectation
        base_thresh = CLASS_CONF_THRESHOLDS.get(cand, 0.48)
        min_conf = base_thresh if (expected_action and cand == expected_action) else max(0.60, base_thresh + 0.10)
        if conf < min_conf:
            self.last_rejected_reason = f"low spotted conf ({conf:.2f} < {min_conf:.2f})"
            return False, self.current_state, cand, conf, self.last_rejected_reason

        # 3. Strict Human Physical Causal Logic Check
        if self.use_fsm:
            allowed, reason = self.causal_logic.can_transition(
                cand, containment=containment or self.containment,
                red_box=red_box, blue_box=blue_box, main_box=main_box,
                dist_red=dist_red, dist_blue=dist_blue, dist_main=dist_main
            )
            if not allowed:
                self.last_rejected_reason = f"blocked: {reason}"
                is_new_action = (cand != self.last_fsm_blocked_action)
                cooldown_ok = (now - self.last_fsm_block_time) > self.fsm_block_cooldown
                if is_new_action or cooldown_ok:
                    exp_str = f" | Expected: '{expected_action}'" if expected_action else ""
                    status_summary = self.causal_logic.get_status_summary(self.dominant_color)
                    ts_print(f"[FSM BLOCKED (SPOTTED)] Gesture detected: '{cand}' (conf {conf:.2f}) -> REJECTED: {reason}{exp_str} [{status_summary}]")
                    self.last_fsm_block_time = now
                    self.last_fsm_blocked_action = cand
                return False, self.current_state, cand, conf, self.last_rejected_reason

        # 3b. Step-Skip Alert for Spotted Action
        if self.use_fsm and cand != "idle" and cand in STEP_ORDER:
            ref_state = self.last_action_state
            cur_idx = STEP_ORDER.index(ref_state) if ref_state in STEP_ORDER else 0
            pred_idx_step = STEP_ORDER.index(cand)
            if pred_idx_step > cur_idx + 1 and (now - self.last_skip_alert_time) > self.skip_alert_cooldown:
                expected_next = STEP_ORDER[cur_idx + 1] if cur_idx + 1 < len(STEP_ORDER) else "(end)"
                skip_msg = (f"[FSM STEP SKIP (SPOTTED)] Gesture detected: '{cand}' (conf {conf:.2f}) | "
                            f"Last confirmed: '{ref_state}' | Expected next: '{expected_next}'")
                ts_print(skip_msg)
                play_sound(freq=400, dur=180)
                self.last_skip_alert_time = now

        # All gates passed -> State transition accepted!
        self.current_state = cand
        self.last_change_time = now
        self.dwell_until = now + self.dwell_sec
        self.last_action_state = cand   # keep skip-alert tracker in sync (same as update() path)
        self.recent.clear()
        self.causal_logic.apply_transition(cand)
        if cand == "place_blue_in":
            self.blue_in_confirm_time = now
        elif cand == "close_box":
            self.blue_in_confirm_time = 0.0
            self.motion_start_time = 0.0
        elif cand == "open_box":
            self.motion_start_time = 0.0
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
    MUST match train_tar.py's TARDataset normalization exactly.
    Updated to per-channel z-score (axis=0) to match the upgraded training pipeline:
    each of the 332 feature channels is normalized independently, preserving the
    distinct distributions of position vs. velocity features.
    Using a different normalization than training is a silent accuracy killer.
    """
    arr = np.array(window, dtype=np.float32)    # (48, 332)
    mean = arr.mean(axis=0, keepdims=True)       # (1, 332) — per-channel mean
    std  = arr.std(axis=0, keepdims=True) + 1e-6 # (1, 332) — per-channel std
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
    with torch.inference_mode():
        x = torch.from_numpy(normed).unsqueeze(0)
        logits = tar_model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()
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
                 start_motion_thresh=0.014,   # Motion to trigger gesture tracking
                 settle_motion_thresh=0.012,  # Realistic settle motion threshold
                 min_action_frames=8,         # Minimum duration of a valid gesture (~0.25s)
                 max_action_frames=45,        # Maximum duration before auto-triggering (~1.5s)
                 settle_required_frames=2,    # Number of calm frames needed to confirm motion end (~0.07s)
                 cooldown_sec=0.40):          # Fast 0.40s cooldown prevents duplicate re-triggers
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


class AsyncYOLODetector:
    """
    Runs YOLO deep learning inference asynchronously in a background daemon thread.
    Allows the main camera processing loop to run fluidly at 25+ FPS without waiting 60ms per frame.
    """
    def __init__(self, yolo_model):
        self.model = yolo_model
        self.latest_detections = []
        self.lock = threading.Lock()
        self.next_frame = None
        self.has_new_frame = threading.Event()
        self.running = True
        if self.model is not None:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def _worker(self):
        while self.running:
            self.has_new_frame.wait(timeout=0.1)
            if not self.running:
                break
            self.has_new_frame.clear()
            frame = None
            with self.lock:
                if self.next_frame is not None:
                    frame = self.next_frame.copy()
                    self.next_frame = None
            if frame is not None:
                dets = run_yolo(self.model, frame)
                with self.lock:
                    self.latest_detections = dets

    def update_frame(self, frame):
        if self.model is None:
            return
        with self.lock:
            self.next_frame = frame
        self.has_new_frame.set()

    def get_detections(self):
        with self.lock:
            return list(self.latest_detections)

    def stop(self):
        self.running = False
        self.has_new_frame.set()


def validate_yolo_detection(frame, rect, cls_name, pose_landmarks=None):
    """
    Color, Aspect Ratio, and Torso Exclusion Validation Guard.
    Prevents skin tones, red clothing on torso, wrist threads, or background reflections
    from being mistaken for actual experiment objects.
    """
    x, y, w, h = rect
    img_h, img_w = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return False

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    area = (x2 - x1) * (y2 - y1)
    ar = (x2 - x1) / float(max(1, y2 - y1))

    # ── Workspace Distance & Horizon Limit ──────────────────────────────────
    # Active packaging happens on the table in the foreground / lower 75% of the camera.
    # Distant background objects (clutter, shelves, objects on distant desks/walls)
    # are physically outside the operator's reachable interaction zone.
    if y2 < (img_h * 0.28):
        # Entire detection is in the far upper background/horizon
        return False

    # Distant background objects appear tiny: reject tiny detections unless close to hand
    if area < 600 and (y1 < img_h * 0.45):
        return False

    # ── Torso & Upper-Body Exclusion Guard ───────────────────────────────────
    # A desk box is located on the table or in hand, NEVER floating on the person's
    # chest, torso, neck, or shirt unless actively held in a hand.
    if pose_landmarks is not None and len(pose_landmarks.landmark) > 16:
        lms = pose_landmarks.landmark
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        
        # Check if wrist (lm 15 or 16) is grasping/touching the detection perimeter (NOT center!)
        wrist_dist_px = 999.0
        for wi in (15, 16):
            wx, wy = lms[wi].x * img_w, lms[wi].y * img_h
            dx = max(0, max(x1 - wx, wx - x2))
            dy = max(0, max(y1 - wy, wy - y2))
            d = np.hypot(dx, dy)
            if d < wrist_dist_px:
                wrist_dist_px = d
        is_grasped = wrist_dist_px <= 45.0

        # Reachability Limit: Objects being interacted with must be within human reach!
        diag = np.hypot(img_w, img_h)
        norm_wrist_dist = wrist_dist_px / diag
        if norm_wrist_dist > 0.62:
            # Object is too far away from the operator's hands (distant background object)
            return False

        # Sub-boxes (Choco Pie / Cadbury Silk) cannot be giant
        if "red" in cls_name or "blue" in cls_name:
            if (x2 - x1) > 280 or (y2 - y1) > 280 or area > 35000:
                return False

        if not is_grasped:
            # 1. Above chest level (neck, chin, head, shoulders)
            shoulder_y = min(lms[11].y, lms[12].y) * img_h
            if cy < shoulder_y:
                return False

            # 2. Inside Torso / Arm Silhouette (chest, stomach, torso, sleeves down to table)
            body_xs = [lms[i].x * img_w for i in [11, 12, 13, 14] if lms[i].visibility > 0.1]
            if not body_xs:
                body_xs = [lms[11].x * img_w, lms[12].x * img_w]
            torso_x_min = min(body_xs) - 40
            torso_x_max = max(body_xs) + 40

            if torso_x_min <= cx <= torso_x_max and cy <= (shoulder_y + img_h * 0.28):
                return False

            # 3. Floor / Bottom Corner Rejection: sub-boxes far from active hand interaction
            if wrist_dist_px > 300.0 and (y2 > img_h * 0.82 or x1 < img_w * 0.12 or x2 > img_w * 0.88):
                return False

    if "red" in cls_name:
        # Lotte Choco Pie Box: Vivid scarlet packaging (S >= 100, V >= 70)
        # Red box on desk cannot be a tall narrow vertical strip like an arm (ar < 0.50)
        if area < 700 or ar < 0.50 or ar > 2.8:
            return False

        # Reject detection if aligned with human arm landmarks
        if pose_landmarks is not None and len(pose_landmarks.landmark) > 16:
            lms = pose_landmarks.landmark
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            for wi in (13, 14, 15, 16):
                if np.hypot(cx - lms[wi].x * img_w, cy - lms[wi].y * img_h) < 75 and (ar < 0.65 or area > 10000):
                    return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # S >= 135, V >= 60 isolates scarlet packaging from skin (skin S < 125) and shadows
        m1_hi = cv2.inRange(hsv, np.array([0, 135, 60]), np.array([14, 255, 255]))
        m2_hi = cv2.inRange(hsv, np.array([168, 135, 60]), np.array([180, 255, 255]))
        hi_sat_pixels = cv2.countNonZero(m1_hi | m2_hi)
        return (hi_sat_pixels / float(area)) >= 0.16

    elif "blue" in cls_name:
        # Cadbury Dairy Milk Silk: Royal navy blue & purple packaging (Hue 112-170, S >= 48, V >= 28)
        # Strictly separates Cadbury Silk packaging from pale blue/cyan/sky blue shirts (Hue 95-110)
        if area < 400 or ar < 0.25 or ar > 3.5:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([112, 48, 28]), np.array([170, 255, 255]))
        # 8% minimum color density accounts for logo, text, graphics, and grasping hand
        return (cv2.countNonZero(mask) / float(area)) >= 0.08

    elif "main" in cls_name or "box" in cls_name:
        # Main box validation:
        # Reject ceiling lights / fixtures
        if y2 < (img_h * 0.32) or (y1 < img_h * 0.10 and y2 < img_h * 0.45):
            return False
        # Reject narrow vertical slivers (doorway pillar) and extreme wide strips
        if ar < 0.40 or ar > 2.8:
            return False
        # Main box must have substantial area
        return area > 3000

    return True


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


# ── YOLO Color Gate constants ────────────────────────────────────────────────
# Distance threshold (normalized by frame diagonal) within which a wrist is
# considered to be actively holding/interacting with a box.
_YOLO_HOLD_DIST = 0.20

# How strongly YOLO identity redirects pick probability when the hand is
# confirmed near a specific-colored box. 0.80 = hard redirect, 0.20 stays
# with original model distribution.
_COLOR_GATE_STRENGTH = 0.80
# ─────────────────────────────────────────────────────────────────────────────


def ground_probs_with_yolo(raw_probs, yolo_red, yolo_blue, red_box, blue_box,
                           pose_landmarks, frame_shape, causal_logic, expected_action=None,
                           main_box=None, containment=None):
    """
    Dual-Level YOLO Grounding with Physical Reality Verification:

    Level 0 — Bayesian Physical Causal Prior Attenuation:
      If an action is physically impossible at the current state (e.g. no box, no red item,
      or box closed), strongly attenuate its probability (x0.01) so it cannot trigger.

    Level 1 — YOLO Hard Color Gate (pick_red vs pick_blue disambiguation):
      When the model is choosing between pick_red / pick_blue, YOLO object
      identity resolves the ambiguity deterministically.

    Level 2 — Gentle Proximity Boost (place actions + non-ambiguous picks).

    Level 3 — SOP Expected Action Prior (ONLY applied if physical prerequisites are met).
    """
    if raw_probs is None:
        return raw_probs, "NONE"

    probs = raw_probs.copy()
    idx_pick_red   = fu.LABELS.index("pick_red")
    idx_place_red  = fu.LABELS.index("place_red_out")
    idx_pick_blue  = fu.LABELS.index("pick_blue")
    idx_place_blue = fu.LABELS.index("place_blue_in")

    # Resolve effective box references (YOLO fallback for HSV boxes)
    eff_red  = red_box  or yolo_red
    eff_blue = blue_box or yolo_blue

    dist_red  = get_box_wrist_distance(eff_red,  pose_landmarks, frame_shape)
    dist_blue = get_box_wrist_distance(eff_blue, pose_landmarks, frame_shape)
    dist_main = get_box_wrist_distance(main_box, pose_landmarks, frame_shape)

    dominant_color = "NONE"

    # ── Level 0: Bayesian Physical Causal Prior Attenuation ──────────────────
    # If an action is physically impossible at the current state, strongly attenuate its
    # probability (x0.01) so it cannot cannibalize the argmax slot of valid actions.
    if causal_logic is not None:
        for lbl in fu.LABELS:
            if lbl == "idle":
                continue
            allowed, _ = causal_logic.can_transition(
                lbl, containment=containment,
                red_box=eff_red, blue_box=eff_blue, main_box=main_box,
                dist_red=dist_red, dist_blue=dist_blue, dist_main=dist_main
            )
            if not allowed:
                lbl_idx = fu.LABELS.index(lbl)
                probs[lbl_idx] *= 0.01

    # ── Level 1: Hard YOLO Color Gate for pick disambiguation ────────────────
    # Trigger only when the model is uncertain between pick_red and pick_blue
    # AND YOLO provides a clear identity signal (hand near a specific-color box).
    pick_mass = probs[idx_pick_red] + probs[idx_pick_blue]
    is_picking_ambiguous = pick_mass > 0.30   # model thinks a pick is happening

    if is_picking_ambiguous:
        red_box_confirmed  = (eff_red  is not None and dist_red  < _YOLO_HOLD_DIST)
        blue_box_confirmed = (eff_blue is not None and dist_blue < _YOLO_HOLD_DIST)

        if red_box_confirmed and not blue_box_confirmed:
            # YOLO confirms: hand is near RED box → redirect pick mass to pick_red
            dominant_color = "RED"
            redirected = pick_mass * _COLOR_GATE_STRENGTH
            probs[idx_pick_red]  = probs[idx_pick_red]  * (1 - _COLOR_GATE_STRENGTH) + redirected
            probs[idx_pick_blue] = probs[idx_pick_blue] * (1 - _COLOR_GATE_STRENGTH)

        elif blue_box_confirmed and not red_box_confirmed:
            # YOLO confirms: hand is near BLUE box → redirect pick mass to pick_blue
            dominant_color = "BLUE"
            redirected = pick_mass * _COLOR_GATE_STRENGTH
            probs[idx_pick_blue] = probs[idx_pick_blue] * (1 - _COLOR_GATE_STRENGTH) + redirected
            probs[idx_pick_red]  = probs[idx_pick_red]  * (1 - _COLOR_GATE_STRENGTH)

    # ── Level 2: Gentle Proximity Boost (place actions + non-ambiguous picks) ─
    if dominant_color == "NONE":   # don't double-count if Level 1 already fired
        if dist_red < 0.25:
            dominant_color = "RED"
            boost_red = 1.50 if expected_action == "pick_red" else 1.25
            probs[idx_pick_red]  *= boost_red
            probs[idx_place_red] *= 1.25
        elif dist_blue < 0.25:
            dominant_color = "BLUE"
            boost = 1.50 if expected_action == "pick_blue" else 1.30
            probs[idx_pick_blue]  *= boost
            probs[idx_place_blue] *= 1.25

    # ── Level 3: SOP Expected Action Prior Boost ─────────────────────────────
    # ONLY boost if the expected action is actually physically possible and target object exists!
    if expected_action and expected_action in fu.LABELS and causal_logic is not None:
        allowed, _ = causal_logic.can_transition(
            expected_action, containment=containment,
            red_box=eff_red, blue_box=eff_blue, main_box=main_box,
            dist_red=dist_red, dist_blue=dist_blue, dist_main=dist_main
        )
        if allowed:
            exp_idx = fu.LABELS.index(expected_action)
            probs[exp_idx] *= 1.25

    # Ground floor for idle: ensure resting/transitioning hands don't blow up attenuated noise
    idx_idle = fu.LABELS.index("idle")
    probs[idx_idle] = max(probs[idx_idle], 0.25)

    # Normalize probability distribution
    total = float(np.sum(probs))
    if total > 1e-6:
        probs = probs / total

    return probs, dominant_color


def process_frame(feat, motion, window, tar_model, stabilizer, spotter, now,
                  yolo_red=None, yolo_blue=None, red_box=None, blue_box=None,
                  main_box=None, pose_landmarks=None, frame_shape=None,
                  expected_action=None, containment=None):
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

    # Pre-calculate wrist distances to all active objects for physical grounding
    eff_red = red_box or yolo_red
    eff_blue = blue_box or yolo_blue
    eff_main = main_box
    dist_red = get_box_wrist_distance(eff_red, pose_landmarks, frame_shape)
    dist_blue = get_box_wrist_distance(eff_blue, pose_landmarks, frame_shape)
    dist_main = get_box_wrist_distance(eff_main, pose_landmarks, frame_shape)

    # Path 1: Motion Action Spotter (Boundary-Aware Peak Scoring)
    triggered, action_clip, spotter_status = spotter.update(feat, motion, now)
    result["spotter_status"] = spotter_status

    if triggered and action_clip is not None:
        sp_label, sp_conf, sp_probs = predict_from_window(tar_model, action_clip)

        # Check for deliberate operator mistake on raw spotted action before attenuation
        is_sp_mistake, sp_mistake_act, sp_mistake_reason = stabilizer.check_operator_mistake(
            sp_label, sp_conf, expected_action, now,
            containment=containment,
            red_box=eff_red, blue_box=eff_blue, main_box=eff_main,
            dist_red=dist_red, dist_blue=dist_blue, dist_main=dist_main
        )
        if is_sp_mistake:
            result["is_mistake"] = True
            result["mistake_detected"] = sp_mistake_act
            result["mistake_reason"] = sp_mistake_reason

        grounded_probs, dominant_color = ground_probs_with_yolo(
            sp_probs, yolo_red, yolo_blue, red_box, blue_box,
            pose_landmarks, frame_shape, stabilizer.causal_logic,
            expected_action=expected_action, main_box=main_box, containment=containment
        )
        changed, current_state, pred_class, conf, status = stabilizer.confirm_spotted_action(
            grounded_probs, now, expected_action=expected_action, containment=containment,
            red_box=eff_red, blue_box=eff_blue, main_box=eff_main,
            dist_red=dist_red, dist_blue=dist_blue, dist_main=dist_main
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
    if len(window) >= 12:
        if len(window) < SEQ_LEN:
            early_clip = resample_sequence(np.array(window, dtype=np.float32), SEQ_LEN)
            raw_label, raw_conf, raw_probs = predict_from_window(tar_model, early_clip)
        else:
            # Fast-path: When hands are motionless, already idle, and no active object is being held, bypass neural pass
            has_active_box = (eff_red is not None and dist_red < 0.35) or (eff_blue is not None and dist_blue < 0.35)
            if stabilizer.current_state == "idle" and motion < (MOTION_THRESHOLD * 0.70) and not has_active_box:
                idle_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
                idle_probs[fu.LABELS.index("idle")] = 1.0
                raw_label, raw_conf, raw_probs = "idle", 1.0, idle_probs
            else:
                raw_label, raw_conf, raw_probs = predict_from_window(tar_model, list(window))

        # Check for deliberate operator mistake on raw rolling prediction before attenuation
        is_raw_mistake, raw_mistake_act, raw_mistake_reason = stabilizer.check_operator_mistake(
            raw_label, raw_conf, expected_action, now,
            containment=containment,
            red_box=eff_red, blue_box=eff_blue, main_box=eff_main,
            dist_red=dist_red, dist_blue=dist_blue, dist_main=dist_main
        )
        if is_raw_mistake:
            result["is_mistake"] = True
            result["mistake_detected"] = raw_mistake_act
            result["mistake_reason"] = raw_mistake_reason

        # Ground probabilities with gentle physical & SOP guidance
        grounded_probs, dominant_color = ground_probs_with_yolo(
            raw_probs, yolo_red, yolo_blue, red_box, blue_box,
            pose_landmarks, frame_shape, stabilizer.causal_logic,
            expected_action=expected_action, main_box=main_box, containment=containment
        )

        changed, current_state, pred_class, conf, status = stabilizer.update(
            grounded_probs, motion, now,
            expected_action=expected_action,
            dominant_color=dominant_color,
            containment=containment,
            red_box=eff_red, blue_box=eff_blue, main_box=eff_main,
            dist_red=dist_red, dist_blue=dist_blue, dist_main=dist_main
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
        if "MISTAKE:" in status:
            result["is_mistake"] = True
            result["mistake_detected"] = pred_class
            result["mistake_reason"] = status.replace("MISTAKE: ", "")

    return result


class BoxTracker:
    """Maintains last-known box coordinates across brief occlusions without infinite feedback."""
    def __init__(self, max_missing=20):
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
    """Tracks SOP execution through the 6 core packaging steps with out-of-order mistake alerts."""
    def __init__(self):
        self.current_step = 0
        self.completed = [False] * len(SOP_STEPS)
        self.cycle_count = 0
        self.cycle_complete_time = 0.0
        self.mistake_msg = ""
        self.mistake_reason = ""
        self.mistake_time = 0.0

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

    @property
    def is_cycle_completed(self):
        return self.current_step >= len(SOP_STEPS)

    def update(self, state, now):
        # Auto-reset checklist to ready state after cycle complete pause
        if self.current_step >= len(SOP_STEPS) and (now - self.cycle_complete_time) >= 2.0:
            self.reset()

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
                self.current_step = len(SOP_STEPS)   # Hold cycle complete state
                play_sound(1400, 90)
            return

        # Fast cycle restart if user opens box right after cycle cooldown
        if state == "open_box" and self.current_step >= len(SOP_STEPS) and (now - self.cycle_complete_time) >= 1.8:
            self.reset()
            self.completed[0] = True
            self.current_step = 1
            play_sound(1100, 60)
            return

        # Step-Order Mistake Detection: operator performed a recognized step out of order
        sop_action_dict = dict(SOP_STEPS)
        if state in sop_action_dict and state != expected:
            # Only sound alert every 2.5 seconds to avoid continuous buzz
            if (now - self.mistake_time) >= 2.5:
                detected_title = sop_action_dict[state]
                expected_title = self.expected_action_title
                self.mistake_msg = f"Expected: {expected_title}  |  You did: {detected_title}"
                self.mistake_time = now
                play_sound(400, 250)  # Warning buzz
                ts_print(f"[⚠️ MISTAKE ALERT] Out-of-sequence! Expected '{expected_title}', but detected '{detected_title}'")
            return

    def reset(self):
        self.current_step = 0
        self.completed = [False] * len(SOP_STEPS)
        self.mistake_msg = ""
        self.mistake_reason = ""
        self.mistake_time = 0.0

    def trigger_mistake(self, detected_action, expected_action=None, reason=""):
        now = time.time()
        if (now - self.mistake_time) >= 2.0:
            sop_action_dict = dict(SOP_STEPS)
            detected_title = sop_action_dict.get(detected_action, detected_action)
            expected_title = self.expected_action_title
            self.mistake_msg = f"Expected: {expected_title}  |  You did: {detected_title}"
            self.mistake_reason = reason
            self.mistake_time = now
            play_sound(400, 250)
            reason_str = f" ({reason})" if reason else ""
            ts_print(f"[⚠️ MISTAKE ALERT] Out-of-sequence! Expected '{expected_title}', but detected '{detected_title}'{reason_str}")

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

        # Big Out-of-Sequence Mistake Alert Banner
        if (now - self.mistake_time) < 3.0:
            cx = w // 2
            by0 = 20
            bw_half = 300
            bh = 72 if getattr(self, "mistake_reason", "") else 56
            flash = (int(now * 5) % 2 == 0)
            box_color = (0, 0, 200) if flash else (0, 0, 150)
            cv2.rectangle(frame, (cx - bw_half, by0), (cx + bw_half, by0 + bh), box_color, -1)
            cv2.rectangle(frame, (cx - bw_half, by0), (cx + bw_half, by0 + bh), (0, 255, 255), 2)
            cv2.putText(frame, "!! OUT OF SEQUENCE MISTAKE !!", (cx - 190, by0 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, self.mistake_msg, (cx - 280, by0 + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 255, 255), 1)
            if getattr(self, "mistake_reason", ""):
                cv2.putText(frame, f"Reason: {self.mistake_reason}", (cx - 280, by0 + 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 230, 255), 1)

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

    torch.set_num_threads(2)
    tar_model = load_tar_model(MODEL_PATH)
    yolo_model = load_yolo(YOLO_MODEL_PATH)
    async_yolo = AsyncYOLODetector(yolo_model)
    stabilizer = DecisionStabilizer(CONFIDENCE_THRESHOLD, STABILITY_WINDOW, COOLDOWN_SEC, CYCLE_COOLDOWN_SEC)
    spotter = MotionActionSpotter()
    box_tracker = BoxTracker(max_missing=8)
    containment = GeometricContainmentEngine(buffer_size=8)
    sop_tracker = SOPTracker()
    recorder = SessionVideoRecorder(output_dir=VIDEO_OUTPUT_DIR)

    window = deque(maxlen=SEQ_LEN)
    prev_feat = None
    prev_base = None

    feedback_banner = ""
    feedback_banner_color = (0, 255, 0)
    feedback_banner_time = 0.0

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    ts_print(f"[Realtime] confidence>={CONFIDENCE_THRESHOLD}  stability={STABILITY_WINDOW} frames  "
             f"cooldown={COOLDOWN_SEC}s  motion_threshold={MOTION_THRESHOLD}")
    ts_print("[Controls] [C]=Mark Correct  |  [0-6]=Correct Label  |  [R]=Reset  |  [V]=Rec Toggle  |  [X]=Flag Wrong  |  [Q]=Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            ts_print("[WARN] Failed to grab frame from camera (ret=False). Exiting.")
            break

        now = time.time()
        clean_raw_frame = frame.copy()   # Pristine raw frame saved for future dataset training

        # Start recording automatically on first frame
        if recorder.writer is None:
            fh, fw = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 60:
                fps = 25.0
            recorder.start(fw, fh, fps)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        hand_res = hands.process(rgb)

        # --- YOLO detections (deep learning box detector, asynchronous non-blocking) ---
        async_yolo.update_frame(frame)
        yolo_detections = async_yolo.get_detections()
        yolo_red, yolo_blue, yolo_main = None, None, None
        for det in yolo_detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            box_item = {"rect": rect, "area": rect[2] * rect[3], "score": det["conf"]}
            name = det["name"]
            cur_lms = pose_res.pose_landmarks if pose_res else None
            if "red" in name:
                if validate_yolo_detection(frame, rect, name, pose_landmarks=cur_lms):
                    # Prefer higher confidence unless already found close in foreground
                    if yolo_red is None or det["conf"] > yolo_red["score"]:
                        yolo_red = box_item
            elif "blue" in name:
                if validate_yolo_detection(frame, rect, name, pose_landmarks=cur_lms):
                    # Proximity prioritization: when choosing between blue boxes, choose the one closest to hands/table foreground!
                    if yolo_blue is None:
                        yolo_blue = box_item
                    else:
                        # Compare vertical position (foreground is lower down, larger Y) and distance to wrists
                        if cur_lms and len(cur_lms.landmark) > 16:
                            cx1, cy1 = rect[0] + rect[2]/2.0, rect[1] + rect[3]/2.0
                            old_r = yolo_blue["rect"]
                            cx0, cy0 = old_r[0] + old_r[2]/2.0, old_r[1] + old_r[3]/2.0
                            d1 = min(np.hypot(cx1 - cur_lms.landmark[15].x*frame.shape[1], cy1 - cur_lms.landmark[15].y*frame.shape[0]),
                                     np.hypot(cx1 - cur_lms.landmark[16].x*frame.shape[1], cy1 - cur_lms.landmark[16].y*frame.shape[0]))
                            d0 = min(np.hypot(cx0 - cur_lms.landmark[15].x*frame.shape[1], cy0 - cur_lms.landmark[15].y*frame.shape[0]),
                                     np.hypot(cx0 - cur_lms.landmark[16].x*frame.shape[1], cy0 - cur_lms.landmark[16].y*frame.shape[0]))
                            if d1 < d0:
                                yolo_blue = box_item
                        elif det["conf"] > yolo_blue["score"]:
                            yolo_blue = box_item
            elif "main" in name or "box" in name:
                if yolo_main is None or det["conf"] > yolo_main["score"]:
                    if validate_yolo_detection(frame, rect, name, pose_landmarks=cur_lms):
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
            main_box=main_box,
            pose_landmarks=pose_res.pose_landmarks if pose_res else None,
            frame_shape=frame.shape,
            expected_action=sop_tracker.expected_action,
            containment=containment
        )

        # Trigger mistake alerts if detected contrary to SOP sequence
        if result.get("is_mistake") and result.get("mistake_detected"):
            sop_tracker.trigger_mistake(
                result["mistake_detected"],
                sop_tracker.expected_action,
                result.get("mistake_reason", "")
            )

        if result["state_changed"] and result['current_state'] != "idle":
            ts_print(f"[STATE] -> {result['current_state']}  (confidence {result['confidence']:.2f})")
            sop_tracker.update(result['current_state'], now)
            if result['current_state'] in ("close_box", "open_box"):
                containment.reset()


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
        hints = "[C] Correct  |  [0-6] Fix Label  |  [R] Reset  |  [V] Rec Toggle  |  [X] Mistake  |  [Q] Quit"
        cv2.putText(frame, hints, (10, h_frame - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        # Real-time Timestamp & Video Recording HUD Badge (Bottom Right)
        fh, fw = frame.shape[:2]
        now_dt = datetime.now()
        ts_clock = now_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]

        rec_badge_w = 265
        rec_badge_h = 38
        rec_x0 = fw - rec_badge_w - 10
        rec_y0 = fh - rec_badge_h - 10
        cv2.rectangle(frame, (rec_x0, rec_y0), (fw - 10, fh - 10), (20, 20, 20), -1)
        cv2.rectangle(frame, (rec_x0, rec_y0), (fw - 10, fh - 10), (60, 60, 60), 1)

        cv2.putText(frame, ts_clock, (rec_x0 + 8, rec_y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        if recorder.is_recording:
            blink = int(now * 2) % 2 == 0
            dot_color = (0, 0, 255) if blink else (80, 80, 200)
            cv2.circle(frame, (rec_x0 + 15, rec_y0 + 27), 4, dot_color, -1)
            rec_sec = int(time.time() - recorder.start_time)
            rec_str = f"REC {rec_sec // 60:02d}:{rec_sec % 60:02d} ({recorder.frame_count}f) [V]"
            cv2.putText(frame, rec_str, (rec_x0 + 26, rec_y0 + 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 100, 255), 1)
        else:
            cv2.putText(frame, "❚❚ REC PAUSED [V to record]", (rec_x0 + 10, rec_y0 + 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        # Record clean raw frame for ML dataset training
        recorder.write_frame(
            clean_raw_frame,
            fsm_state=result.get("current_state"),
            pred_label=result.get("predicted_label"),
            confidence=result.get("confidence", 0.0),
            status=result.get("status", "")
        )

        cv2.imshow("Realtime HAR", frame)

        # Auto-close window when cycle is completed (with brief celebration display)
        if sop_tracker.is_cycle_completed and (now - sop_tracker.cycle_complete_time) >= 1.2:
            ts_print(f"[CYCLE COMPLETE] Full SOP sequence finished successfully! Closing window.")
            break

        # --- Keyboard Interactions ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('v'), ord('V')):
            # Toggle session video recording
            fh, fw = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            recorder.toggle(fw, fh, fps)
            if recorder.is_recording:
                feedback_banner = f"REC: Recording session ({os.path.basename(recorder.video_path)})"
                feedback_banner_color = (0, 100, 255)
            else:
                feedback_banner = "REC: Paused"
                feedback_banner_color = (0, 220, 255)
            feedback_banner_time = time.time()

        elif key in (ord('r'), ord('R')):
            # Reset cycle and physical state to start
            stabilizer.causal_logic.reset()
            containment.reset()
            stabilizer.current_state = "idle"
            stabilizer.last_action_state = "idle"
            stabilizer.last_change_time = 0.0
            stabilizer.recent.clear()
            stabilizer.release_counter_red = 0
            stabilizer.release_counter_blue = 0
            spotter.state = spotter.STATE_IDLE
            spotter.buffer.clear()
            sop_tracker.reset()
            feedback_banner = "RESET: Cycle & physical logic reset to start"
            feedback_banner_color = (0, 255, 255)
            feedback_banner_time = time.time()
            ts_print("[RESET] Physical causal state, 2.5D containment, and SOP reset.")

        elif key in (ord('c'), ord('C')):
            # User marks current sequence as CORRECT
            curr_label = result.get('predicted_label') or stabilizer.current_state or "idle"
            label_id = fu.LABELS.index(curr_label) if curr_label in fu.LABELS else 0
            ok, msg = save_feedback(window, "CORRECT", label_id, curr_label)
            feedback_banner = msg
            feedback_banner_color = (0, 255, 0)
            feedback_banner_time = time.time()
            ts_print(f"[FEEDBACK] {msg}")

        elif key in (ord('x'), ord('X')):
            # User marks current sequence as WRONG / MISTAKE
            curr_label = result.get('predicted_label') or "unknown"
            label_id = fu.LABELS.index(curr_label) if curr_label in fu.LABELS else 0
            ok, msg = save_feedback(window, "MISTAKE", label_id, curr_label)
            feedback_banner = msg
            feedback_banner_color = (0, 0, 255)
            feedback_banner_time = time.time()
            ts_print(f"[FEEDBACK] {msg}")

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
            ts_print(f"[FEEDBACK] {msg}")

    async_yolo.stop()
    recorder.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()