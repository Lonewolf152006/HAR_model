"""
Advanced pose+hands+objects data collector.

Extends the previous collector with:
  - Big box (main container) detection via SHAPE/contour analysis, not
    color - the main box is very unlikely to be a strong, distinct hue the
    way the red/blue sub-boxes are, so color thresholding won't find it
    reliably. This reuses the contour-area/solidity/quadrilateral-fit
    scoring approach from the color-based detector, just run on an edge
    mask instead of a color mask.
  - Continuous size (area) for ALL THREE tracked objects, not just
    position - lets the model learn "big vs small" from real numbers,
    the same way it already learns red vs blue from position/area rather
    than a hand-written threshold. A hard-coded size cutoff would be as
    brittle as the original hard-coded color thresholds were.
  - Wrist-to-object relational distances, now including the main box
    (6-dim instead of 4).
  - First-order velocity (frame-to-frame delta of the base feature
    vector) - fixes a real gap from the earlier mean/std-pooled approach,
    which couldn't distinguish open_box from close_box since they're
    near-time-reversals of each other and pooling across time throws away
    direction entirely.

FEATURE VECTOR LAYOUT (source of truth - keep model_def.py's FEATURE_DIM
in sync with TOTAL_DIM below):

  Pose              66   hip-centered, torso-scaled            [0:66]
  Left hand         42   wrist-centered, palm-scaled           [66:108]
  Right hand        42   wrist-centered, palm-scaled           [108:150]
  Red object         3   (cx, cy, area), frame-normalized      [150:153]
  Blue object         3   (cx, cy, area), frame-normalized      [153:156]
  Main box            3   (cx, cy, area), frame-normalized      [156:159]
  Wrist->object       6   L/R wrist to each of red/blue/main,   [159:165]
  distances                normalized by frame diagonal
  Lid complexity      1   edge density inside the main box's    [165:166]
                           interior region (raw continuous score,
                           not thresholded - see _interior_complexity)
  -----------------------------------------------------------------
  BASE_DIM          166
  Velocity          166   frame[t] - frame[t-1] of the above    [166:332]
  -----------------------------------------------------------------
  TOTAL_DIM         332

NOTE: BASE_DIM changed from 165 to 166 (TOTAL_DIM 330 to 332) with the
addition of the lid-complexity feature. If you already set FEATURE_DIM=330
in model_def.py, it needs to become 332, and any dataset collected before
this change is NOT compatible with dataset collected after it - same rule
as every previous feature-dimension change in this project.

NOTE ON CLASS LABELS: this keeps the 7-class protocol used throughout
(0=idle .. 6=close_box) - NOT the 6-class taxonomy from the architecture
doc you pasted earlier (which merges pick_red/pick_blue into color-
agnostic pick/place actions). That's a real protocol change, not just a
feature change - it would need the locked protocol doc and FSM updated
too, so it isn't applied here without confirming that's actually wanted.

Usage:
    python pose_extract_advanced.py
Controls:
    0-6   = select label
    s     = save current SEQ_LEN-frame window
    e     = clear buffer & pause (discard a bad take)
    r     = resume
    q     = quit
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import re
import csv
import time
from collections import deque

# ================= CONFIG =================
OUTPUT_DIR = "dataset_advanced"
SEQ_LEN = 48          # matches the architecture doc's model - change here if your model_def.py differs
CAMERA_ID = 0
FLIP_FRAME = False

ACTIONS = [
    (0, "idle"),
    (1, "open_box"),
    (2, "pick_red"),
    (3, "place_red_out"),
    (4, "pick_blue"),
    (5, "place_blue_in"),
    (6, "close_box"),
]
LABEL_NAMES = dict(ACTIONS)

METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

# ================= YOLO DETECTION CONFIG =================
YOLO_MODEL_PATH = "yolo_boxes.pt"
YOLO_CONF_THRESHOLD = 0.25
ALLOWED_YOLO_CLASSES = {
    "box", "cardboard box", "package", "container", "crate", "suitcase",
    "red_box", "blue_box", "main_box"
}


def load_yolo(path):
    if not os.path.exists(path):
        print(f"[YOLO] Model file '{path}' not found - continuing with fallback HSV/edge detector.")
        return None
    try:
        from ultralytics import YOLO
        print(f"[YOLO] Loaded custom model: {path}")
        return YOLO(path)
    except Exception as e:
        print(f"[YOLO] Could not load '{path}': {e}. Continuing with fallback HSV/edge detector.")
        return None


def run_yolo(yolo_model, frame, conf_threshold=YOLO_CONF_THRESHOLD, allowed_classes=ALLOWED_YOLO_CLASSES):
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
                if allowed_classes and not any(target in cls_name for target in allowed_classes):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append({"name": cls_name, "conf": conf, "box": (x1, y1, x2, y2)})
        return detections
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        return []


def get_yolo_override_boxes(yolo_model, frame):
    """Runs YOLO on the frame and maps detections to (yolo_red, yolo_blue, yolo_main)."""
    if yolo_model is None:
        return (None, None, None)
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
    return (yolo_red, yolo_blue, yolo_main)


# Calibrated red ranges: strictly matches vivid Lotte Choco Pie scarlet packaging.
# S >= 155 completely separates scarlet Choco Pie packaging from human skin (skin saturation <= 130).
RED_RANGES = [((0, 155, 80), (8, 255, 255)), ((170, 155, 80), (180, 255, 255))]
# Calibrated blue/purple ranges: matches Cadbury Silk purple/violet and royal blue packaging (Hue 95 to 170)
BLUE_RANGE = [((95, 30, 25), (170, 255, 255))]
COLOR_MIN_AREA_RED = 1500   # Choco Pie box is substantial; rejects small threads/clothing patches
COLOR_MIN_AREA_BLUE = 800    # Solid Cadbury Silk box; rejects tiny reflections
COLOR_MIN_AREA = COLOR_MIN_AREA_RED

# ================= SHAPE DETECTION (main box fallback - no reliable color) =================
MAIN_BOX_MIN_AREA = 3000     # main box should read larger than sub-boxes - tune against your setup
IOU_EXCLUDE_THRESHOLD = 0.3  # skip shape candidates that overlap a detected sub-box this much

# ================= AUTO MODE CONFIG =================
# Set this to True and just click Run in your IDE - this is the reliable way
# to launch auto mode when your IDE's Run button doesn't pass command-line
# flags through (many don't, by default). The --auto / --manual flags below
# still work too if you're launching from an actual terminal - they override
# this switch either way, so you don't need to remember to flip it back for
# terminal use.
DEFAULT_TO_AUTO_MODE = False

AUTO_SAMPLES_PER_LABEL = 5
AUTO_COUNTDOWN_SEC = 3
AUTO_RECORD_SEC = 3
AUTO_REST_SEC = 1.5
AUTO_MIN_POSE_DETECTION_RATE = 0.70   # below this, auto-discard the take and retry


# ================= NORMALIZATION HELPERS =================
def normalize_pose_landmarks(pose_landmarks):
    """66-dim vector: 33 (x, y) normalized coords, centered at hip midpoint, scaled by torso length."""
    if pose_landmarks is None:
        return [0.0] * 66
    lms = pose_landmarks.landmark
    hip_x = (lms[23].x + lms[24].x) / 2.0
    hip_y = (lms[23].y + lms[24].y) / 2.0
    scale = np.hypot(lms[11].x - lms[23].x, lms[11].y - lms[23].y)
    scale = max(scale, 1e-4)

    feats = []
    for lm in lms:
        feats.append((lm.x - hip_x) / scale)
        feats.append((lm.y - hip_y) / scale)
    return feats


def normalize_hand_landmarks(hand_landmarks):
    """42-dim vector: 21 (x, y) normalized coords, centered at wrist (idx 0), scaled by palm size."""
    if hand_landmarks is None:
        return [0.0] * 42
    lms = hand_landmarks.landmark
    wrist_x, wrist_y = lms[0].x, lms[0].y
    scale = np.hypot(lms[9].x - wrist_x, lms[9].y - wrist_y)
    scale = max(scale, 1e-4)

    feats = []
    for lm in lms:
        feats.append((lm.x - wrist_x) / scale)
        feats.append((lm.y - wrist_y) / scale)
    return feats


# ================= SHARED CONTOUR SCORING =================
def _score_contour(cnt, min_area, is_red=False):
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    # Box dimensions: must have substantial width and height (threads/folds are too thin)
    if w < 30 or h < 30:
        return None
    ar = w / float(max(1, h))
    # Red box (Choco Pie) cannot be a tall narrow vertical strip like an arm (ar < 0.55)
    if is_red and (ar < 0.55 or ar > 2.6):
        return None
    elif not is_red and (ar < 0.25 or ar > 4.0):
        return None
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) if hull is not None else 0
    solidity = (area / hull_area) if hull_area > 0 else 0
    if solidity < 0.45:  # Rejects thin curved threads or sparse sleeve folds
        return None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    approx_ok = len(approx) == 4
    score = area * 0.5 + solidity * 2000 + (300 if approx_ok else 0)
    return {"rect": (x, y, w, h), "area": area, "solidity": solidity, "approx_ok": approx_ok, "score": score}


def _candidates_from_mask(mask, min_area, is_red=False):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in (_score_contour(cnt, min_area, is_red=is_red) for cnt in contours) if c is not None]
    candidates.sort(key=lambda c: -c["score"])
    return candidates


# ================= COLOR-BASED SUB-BOX DETECTION =================
def _mask_from_ranges(hsv, ranges):
    mask = None
    for low, high in ranges:
        m = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def _clean_mask(mask):
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open, iterations=1)
    return m


_LAST_VALID_BODY_MASK = None

def get_body_exclusion_mask(pose_landmarks, frame_shape):
    """
    Creates a binary mask (255=desk/search area, 0=person head/face/upper torso).
    Ensures human face, lips, skin, and torso clothing are NEVER misidentified as boxes.
    Caches last known mask across brief pose dropouts.
    """
    global _LAST_VALID_BODY_MASK
    h, w = frame_shape[:2]

    # If pose drops out, reuse the last known body mask rather than exposing the face
    if not pose_landmarks:
        if _LAST_VALID_BODY_MASK is not None and _LAST_VALID_BODY_MASK.shape == (h, w):
            return _LAST_VALID_BODY_MASK.copy()
        mask = np.ones((h, w), dtype=np.uint8) * 255
        cv2.rectangle(mask, (0, 0), (w, int(h * 0.42)), 0, -1)
        return mask

    lms = pose_landmarks.landmark
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # 1. Full Body Column exclusion (masks head, neck, torso, shirt, sleeves, and lap)
    shoulder_y = int(min(lms[11].y, lms[12].y) * h)
    body_xs = [lms[i].x * w for i in [11, 12, 13, 14] if lms[i].visibility > 0.1]
    if not body_xs:
        body_xs = [lms[11].x * w, lms[12].x * w]
    body_left = max(0, int(min(body_xs)) - 100)
    body_right = min(w, int(max(body_xs)) + 100)

    # Mask entire person: head/neck down to the bottom of the frame
    cv2.rectangle(mask, (0, 0), (w, max(0, shoulder_y - 20)), 0, -1)
    cv2.rectangle(mask, (body_left, max(0, shoulder_y - 30)), (body_right, h), 0, -1)

    # 2. Upper arm exclusion (keep hands/wrists unmasked so held boxes are not destroyed)
    if len(lms) > 16:
        cv2.line(mask, (int(lms[11].x * w), int(lms[11].y * h)), (int(lms[13].x * w), int(lms[13].y * h)), 0, 110)
        cv2.line(mask, (int(lms[12].x * w), int(lms[12].y * h)), (int(lms[14].x * w), int(lms[14].y * h)), 0, 110)

    # 3. Face & head exclusion (landmarks 0 to 10: nose, eyes, ears, mouth)
    face_x = [lms[i].x * w for i in range(11)]
    face_y = [lms[i].y * h for i in range(11)]
    fx1 = max(0, int(min(face_x) - w * 0.12))
    fx2 = min(w, int(max(face_x) + w * 0.12))
    fy1 = max(0, int(min(face_y) - h * 0.18))
    fy2 = min(h, int(max(face_y) + h * 0.15))
    cv2.rectangle(mask, (fx1, fy1), (fx2, fy2), 0, -1)

    _LAST_VALID_BODY_MASK = mask.copy()
    return mask


def detect_color_boxes(frame, exclusion_mask=None, pose_landmarks=None):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    red_mask = _clean_mask(_mask_from_ranges(hsv, RED_RANGES))
    blue_mask = _clean_mask(_mask_from_ranges(hsv, BLUE_RANGE))
    if exclusion_mask is not None:
        # ONLY apply body exclusion mask to red (skin/lips/torso). Never apply to blue/purple!
        red_mask = cv2.bitwise_and(red_mask, red_mask, mask=exclusion_mask)

    red_candidates = _candidates_from_mask(red_mask, COLOR_MIN_AREA_RED, is_red=True)
    blue_candidates = _candidates_from_mask(blue_mask, COLOR_MIN_AREA_BLUE, is_red=False)

    h, w = frame.shape[:2]
    face_limit_y = int(h * 0.38)

    def _filter_best(candidates, is_red=False):
        for b in candidates:
            bx, by, bw, bh = b["rect"]
            ar = bw / float(max(1, bh))

            # Reject extreme edge slivers (shelves, door frames on room boundaries)
            if not is_red:
                if (bx > w - 75 and bw < 65) or (bx < 25 and bw < 50):
                    continue

            # Strictly reject vertical elongated shapes for red box (human arm)
            if is_red and ar < 0.55:
                continue

            # Wrist thread / forearm rejection
            if pose_landmarks is not None and len(pose_landmarks.landmark) > 16:
                lms = pose_landmarks.landmark
                bcx = bx + bw / 2.0
                bcy = by + bh / 2.0
                near_arm = False
                for wi in (13, 14, 15, 16):
                    if np.hypot(bcx - lms[wi].x * w, bcy - lms[wi].y * h) < 75:
                        near_arm = True
                        break
                if near_arm and is_red and (ar < 0.65 or b["area"] > 10000):
                    continue

            # Workspace Distance & Horizon Limit: reject distant background objects
            if (by + bh) < int(h * 0.28):
                continue

            # Reachability Limit: Object must be within reachable human interaction range
            if pose_landmarks is not None and len(pose_landmarks.landmark) > 16:
                lms = pose_landmarks.landmark
                cx = bx + bw / 2.0
                cy = by + bh / 2.0
                wrist_dist_px = min(
                    np.hypot(cx - lms[15].x * w, cy - lms[15].y * h),
                    np.hypot(cx - lms[16].x * w, cy - lms[16].y * h)
                )
                diag = np.hypot(w, h)
                if (wrist_dist_px / diag) > 0.68:
                    continue

            # Upper 38% screen check: if candidate has solid area (>= 3000 px), it is a real held box shown to camera!
            # Only tiny shapes (< 3000 px) in the top 38% need strict wrist proximity to reject ceiling lights.
            if by < face_limit_y and b["area"] < 3000:
                wrist_near = False
                if pose_landmarks is not None and len(pose_landmarks.landmark) > 16:
                    lms = pose_landmarks.landmark
                    cx = bx + bw / 2.0
                    cy = by + bh / 2.0
                    for wi in (15, 16):
                        wx, wy = lms[wi].x * w, lms[wi].y * h
                        if np.hypot(cx - wx, cy - wy) < 220:
                            wrist_near = True
                            break
                if not wrist_near:
                    continue

            return b
        return None

    red = _filter_best(red_candidates, is_red=True)
    blue = _filter_best(blue_candidates, is_red=False)
    return red, blue


# ================= SHAPE-BASED MAIN BOX DETECTION =================
def _iou(rect_a, rect_b):
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detect_main_box(frame, exclusion_mask=None, exclude_rects=None):
    """
    Finds the largest solid, roughly-rectangular region using edges rather
    than color. Candidates overlapping a detected sub-box or person body are excluded.
    """
    exclude_rects = exclude_rects or []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    if exclusion_mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=exclusion_mask)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        c = _score_contour(cnt, MAIN_BOX_MIN_AREA)
        if c is None:
            continue
        if any(_iou(c["rect"], ex) > IOU_EXCLUDE_THRESHOLD for ex in exclude_rects):
            continue
        candidates.append(c)
    if not candidates:
        return None, edges
    candidates.sort(key=lambda c: -c["score"])
    return candidates[0], edges


# ================= FEATURE ASSEMBLY =================
def _box_stats(box, frame_shape):
    """(cx, cy, area) normalized to [0,1] relative to frame - zeros if not found."""
    h, w = frame_shape[:2]
    if box is None:
        return [0.0, 0.0, 0.0]
    x, y, bw, bh = box["rect"]
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    area_norm = (bw * bh) / (w * h)
    return [cx, cy, area_norm]


def _wrist_distances(pose_landmarks, red, blue, main, frame_shape):
    """6-dim: [WL-red, WL-blue, WL-main, WR-red, WR-blue, WR-main], normalized by frame diagonal."""
    h, w = frame_shape[:2]
    diag = (h ** 2 + w ** 2) ** 0.5
    if not pose_landmarks:
        return [1.0] * 6  # "maximally far" default when no person detected

    lms = pose_landmarks.landmark
    l_wrist = np.array([lms[15].x * w, lms[15].y * h])
    r_wrist = np.array([lms[16].x * w, lms[16].y * h])

    def center_px(box):
        if box is None:
            return None
        x, y, bw, bh = box["rect"]
        return np.array([x + bw / 2.0, y + bh / 2.0])

    dists = []
    for wrist in (l_wrist, r_wrist):
        for box in (red, blue, main):
            c = center_px(box)
            d = float(np.linalg.norm(wrist - c) / diag) if c is not None else 1.0
            dists.append(d)
    return dists


def _interior_complexity(frame, box, shrink_ratio=0.15):
    """
    Edge density inside the main box's detected region, shrunk inward by
    shrink_ratio to avoid picking up the box's own outer edge/border as
    'complexity'.
    """
    if box is None:
        return 0.0
    x, y, w, h = box["rect"]
    dx, dy = int(w * shrink_ratio), int(h * shrink_ratio)
    x0, y0 = max(0, x + dx), max(0, y + dy)
    x1, y1 = min(frame.shape[1], x + w - dx), min(frame.shape[0], y + h - dy)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    return float(np.count_nonzero(edges) / edges.size)


BASE_DIM = 66 + 42 + 42 + 3 + 3 + 3 + 6 + 1   # = 166
TOTAL_DIM = BASE_DIM * 2                       # = 332 (base + velocity)


def extract_base_features(pose_res, hand_res, frame, override_boxes=None):
    pose_landmarks_obj = pose_res.pose_landmarks if pose_res else None
    pose_feats = normalize_pose_landmarks(pose_landmarks_obj)

    left_hand, right_hand = [0.0] * 42, [0.0] * 42
    if hand_res and hand_res.multi_hand_landmarks and hand_res.multi_handedness:
        for i, hand_lms in enumerate(hand_res.multi_hand_landmarks):
            label = hand_res.multi_handedness[i].classification[0].label
            normed = normalize_hand_landmarks(hand_lms)
            if label == "Left":
                left_hand = normed
            elif label == "Right":
                right_hand = normed

    # 1. Mask out person's face/head and upper body so they are NEVER detected as boxes
    body_mask = get_body_exclusion_mask(pose_landmarks_obj, frame.shape)

    # 2. Prioritize YOLO detections if provided (YOLO was specifically trained on the actual boxes)
    red, blue, main = None, None, None
    if override_boxes is not None:
        o_red, o_blue, o_main = override_boxes
        red = o_red
        blue = o_blue
        main = o_main

    # 3. Run color detection for reliable sub-box localization:
    # Cadbury Silk purple has distinct packaging color. If HSV finds a solid purple box (area >= 1200),
    # prefer it over YOLO's noisy/oversized boxes!
    hsv_red, hsv_blue = detect_color_boxes(frame, exclusion_mask=body_mask, pose_landmarks=pose_landmarks_obj)
    if red is None:
        red = hsv_red
    elif hsv_red is not None and red.get("score", 1.0) < 0.35:
        red = hsv_red

    if blue is None:
        blue = hsv_blue
    elif hsv_blue is not None:
        if blue.get("area", 0) > 40000 or blue.get("score", 1.0) < 0.40 or hsv_blue["area"] > 2500:
            blue = hsv_blue

    edge_debug = None
    if main is None:
        exclude_sub = [b["rect"] for b in (red, blue) if b is not None]
        main, edge_debug = detect_main_box(frame, exclusion_mask=body_mask, exclude_rects=exclude_sub)

    red_stats = _box_stats(red, frame.shape)
    blue_stats = _box_stats(blue, frame.shape)
    main_stats = _box_stats(main, frame.shape)
    dists = _wrist_distances(pose_landmarks_obj, red, blue, main, frame.shape)
    lid_complexity = [_interior_complexity(frame, main)]

    base = pose_feats + left_hand + right_hand + red_stats + blue_stats + main_stats + dists + lid_complexity
    assert len(base) == BASE_DIM, f"base feature length {len(base)} != {BASE_DIM}"

    detections = {"red": red is not None, "blue": blue is not None, "main": main is not None}
    return np.array(base, dtype=np.float32), detections, (red, blue, main), edge_debug, lid_complexity[0]





# ================= METADATA =================
def ensure_metadata():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["filename", "label_id", "label_name", "red_detect_rate", "blue_detect_rate",
                 "main_detect_rate", "num_frames", "timestamp"])


def append_metadata(row):
    with open(METADATA_PATH, "a", newline="") as f:
        csv.writer(f).writerow(row)


def remove_metadata_row(filename):
    """Rewrites metadata.csv excluding the given filename - used by auto
    mode's redo, so a deleted .npy doesn't leave a stale row behind that
    claims a sample exists which no longer does."""
    if not os.path.exists(METADATA_PATH):
        return
    with open(METADATA_PATH, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, body = rows[0], rows[1:]
    filtered = [r for r in body if not r or r[0] != filename]
    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(filtered)


def next_sample_index(label_id):
    """Auto-resumes numbering per label - same overwrite-bug fix as extract_from_video.py."""
    pattern = re.compile(rf"label_{label_id}_(\d+)\.npy")
    existing = []
    if os.path.isdir(OUTPUT_DIR):
        for fname in os.listdir(OUTPUT_DIR):
            m = pattern.match(fname)
            if m:
                existing.append(int(m.group(1)))
    return max(existing, default=-1) + 1


# ================= RESAMPLING / VELOCITY (used by auto mode) =================
def resample_base_sequence(base_list, target_len):
    """
    Uniformly resamples a variable-length list of BASE_DIM vectors to
    exactly target_len frames via per-dimension linear interpolation.
    Auto mode records for a fixed number of SECONDS, not a fixed number of
    FRAMES, so the raw frame count varies with camera fps and timing -
    this is what makes a fixed RECORD_SEC safe regardless: a recording
    only needs to comfortably contain the action, not match its length
    exactly. (This replaces naive `sequence[:SEQ_LEN]` truncation, which
    would silently cut off actions that run past the frame budget instead
    of capturing all of them.)
    """
    seq = np.array(base_list, dtype=np.float32)
    n_frames = seq.shape[0]
    if n_frames == target_len:
        return seq
    orig_idx = np.linspace(0, 1, n_frames)
    target_idx = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, BASE_DIM), dtype=np.float32)
    for d in range(BASE_DIM):
        resampled[:, d] = np.interp(target_idx, orig_idx, seq[:, d])
    return resampled


def add_velocity(base_resampled):
    """Frame-to-frame velocity computed on the already-resampled series,
    so it reflects motion on the canonical SEQ_LEN grid rather than the
    camera's raw, possibly uneven native frame timing."""
    velocity = np.zeros_like(base_resampled)
    velocity[1:] = base_resampled[1:] - base_resampled[:-1]
    return np.concatenate([base_resampled, velocity], axis=1)


def _person_bbox(pose_landmarks, frame_shape):
    """
    Bounding box around all detected pose landmarks - a clear, at-a-glance
    'is a person actually in frame and tracked' indicator, distinct from
    the skeleton overlay (which is easy to miss from a few feet back).
    Returns (x, y, w, h) in pixels, or None if no person detected.
    """
    if not pose_landmarks:
        return None
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in pose_landmarks.landmark]
    ys = [lm.y * h for lm in pose_landmarks.landmark]
    x0, x1 = max(0, min(xs)), min(w, max(xs))
    y0, y1 = max(0, min(ys)), min(h, max(ys))
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _draw_person_indicator(frame, pose_landmarks):
    """Draws a bounding box + label when a person is detected, or a hard-to-miss
    warning banner when one isn't - used identically by manual and auto mode."""
    bbox = _person_bbox(pose_landmarks, frame.shape)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, "PERSON", (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        fh, fw = frame.shape[:2]
        cv2.putText(frame, "!! NO PERSON DETECTED !!", (fw // 2 - 220, fh - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    return bbox is not None


# ================= SHARED MEDIAPIPE INIT =================
def _init_mediapipe():
    try:
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
    return pose, hands, mp_drawing, mp_pose, mp_hands


# ================= MAIN LOOP (manual mode) =================
def main():
    pose, hands, mp_drawing, mp_pose, mp_hands = _init_mediapipe()
    ensure_metadata()
    yolo_model = load_yolo(YOLO_MODEL_PATH)

    fused_buffer = deque(maxlen=SEQ_LEN)
    detect_buffer = deque(maxlen=SEQ_LEN)
    prev_base = None
    current_label = 0
    recording = True

    print(f"[Collector] BASE_DIM={BASE_DIM}  TOTAL_DIM (with velocity)={TOTAL_DIM}  SEQ_LEN={SEQ_LEN}")
    print("\nCONTROLS: 0-6 = label | s = save | e = clear/pause | r = resume | q = quit\n")
    for k, v in ACTIONS:
        print(f"  [{k}] {v}")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        hand_res = hands.process(rgb)

        # Run YOLO box detection and feed into feature extraction
        yolo_boxes = get_yolo_override_boxes(yolo_model, frame)
        base, detections, (red, blue, main), edge_debug, lid_score = extract_base_features(
            pose_res, hand_res, frame, override_boxes=yolo_boxes
        )

        if recording:
            velocity = base - prev_base if prev_base is not None else np.zeros(BASE_DIM, dtype=np.float32)
            prev_base = base
            fused = np.concatenate([base, velocity])
            fused_buffer.append(fused)
            detect_buffer.append(detections)

        # Visual feedback: landmarks and bounding boxes (always visible)
        if pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        _draw_person_indicator(frame, pose_res.pose_landmarks)
        if hand_res.multi_hand_landmarks:
            for h in hand_res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        for box, color, b_name in ((red, (0, 0, 255), "RED"),
                                   (blue, (255, 0, 0), "BLUE"),
                                   (main, (0, 255, 0), "MAIN")):
            if box is not None:
                x, y, w, h = box["rect"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                score_str = f" {box['score']:.2f}" if "score" in box and box["score"] is not None else ""
                cv2.putText(frame, f"{b_name}{score_str}", (x, max(20, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Live readout of lid-complexity score
        if lid_score is not None:
            cv2.putText(frame, f"Lid complexity: {lid_score:.3f}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Debug overlay: edge mask used for main-box detection, picture-in-picture.
        if edge_debug is not None:
            h, w = frame.shape[:2]
            small = cv2.resize(cv2.cvtColor(edge_debug, cv2.COLOR_GRAY2BGR), (int(w * 0.25), int(h * 0.25)))
            frame[0:small.shape[0], w - small.shape[1]:w] = small

        label_name = LABEL_NAMES.get(current_label, "?")
        cv2.putText(frame, f"Seq: {len(fused_buffer)}/{SEQ_LEN}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Label: {current_label} ({label_name})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"{'REC' if recording else 'PAUSED'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Advanced Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('e'):
            fused_buffer.clear()
            detect_buffer.clear()
            prev_base = None
            recording = False
            print("[!] Cleared & paused")
        elif key == ord('r'):
            recording = True
            print("[+] Resumed")
        elif ord('0') <= key <= ord('9'):
            current_label = int(chr(key))
            print(f"[+] Label -> {current_label} ({LABEL_NAMES.get(current_label, '?')})")
        elif key == ord('s'):
            if len(fused_buffer) == SEQ_LEN:
                seq = np.array(fused_buffer)
                sample_idx = next_sample_index(current_label)
                fname = f"label_{current_label}_{sample_idx}.npy"
                np.save(os.path.join(OUTPUT_DIR, fname), seq)

                red_rate = sum(d["red"] for d in detect_buffer) / len(detect_buffer)
                blue_rate = sum(d["blue"] for d in detect_buffer) / len(detect_buffer)
                main_rate = sum(d["main"] for d in detect_buffer) / len(detect_buffer)
                append_metadata([fname, current_label, label_name,
                                  round(red_rate, 2), round(blue_rate, 2), round(main_rate, 2),
                                  SEQ_LEN, time.time()])
                print(f"[+] Saved {fname}  shape={seq.shape}  "
                      f"(red_det={red_rate:.0%} blue_det={blue_rate:.0%} main_det={main_rate:.0%})")
            else:
                print(f"❌ Need {SEQ_LEN} frames (have {len(fused_buffer)})")

    cap.release()
    cv2.destroyAllWindows()


# ================= AUTO MODE (python pose_extract_advanced.py --auto) =================
def main_auto(samples_per_label=AUTO_SAMPLES_PER_LABEL):
    """
    Single-loop COUNTDOWN -> RECORD -> REST state machine - hands-free
    once started, cycling through all actions automatically.

    Differences from a naive version of this pattern, both load-bearing:
      - RECORD accumulates raw per-frame features for the full RECORD_SEC
        window, then resample_base_sequence() maps them to exactly
        SEQ_LEN frames - a fixed-length slice instead would silently
        truncate any action that runs past the frame budget.
      - Before saving, the fraction of frames with a detected person is
        checked against AUTO_MIN_POSE_DETECTION_RATE. Below that, the
        take is discarded and retried automatically instead of saving a
        clip where tracking dropped out.
      - 'r' (redo) actually deletes the most recently saved file for the
        CURRENT label if one exists, before restarting the countdown -
        not just recording another sample alongside the bad one.
    """
    pose, hands, mp_drawing, mp_pose, mp_hands = _init_mediapipe()
    ensure_metadata()
    yolo_model = load_yolo(YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    label_idx = 0
    sample_count = 0
    mode = "COUNTDOWN"
    timer_start = time.time()
    countdown_accum = 0.0     # accumulated countdown time - only advances while a person is present
    last_tick_time = time.time()
    base_sequence = []
    pose_flags = []
    detection_sequence = []
    last_saved_path = None
    last_saved_label_idx = None
    boxes = None

    print(f"[AutoCollector] {samples_per_label} samples x {len(ACTIONS)} actions")
    print("Q = quit | R = redo (deletes the last saved sample for this label if any)\n")
    print(f"NEXT ACTION: {ACTIONS[label_idx][1]}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        now = time.time()
        elapsed = now - timer_start
        dt = now - last_tick_time
        last_tick_time = now
        current_label_id, current_label_name = ACTIONS[label_idx]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        hand_res = hands.process(rgb)
        person_present = _draw_person_indicator(frame, pose_res.pose_landmarks)

        # Detect boxes via YOLO
        yolo_boxes = get_yolo_override_boxes(yolo_model, frame)

        if mode == "COUNTDOWN":
            # Accumulate countdown time only while a person is actually
            # present - pausing (not resetting) on a miss means brief
            # single-frame detection flicker, which happens routinely even
            # with someone clearly in view, doesn't wipe out all progress
            # and leave the countdown stuck never reaching zero.
            if person_present:
                countdown_accum += dt
            remaining = max(0.0, AUTO_COUNTDOWN_SEC - countdown_accum)

            if not person_present:
                cv2.putText(frame, "Waiting for you to enter frame...", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                if countdown_accum > 0:
                    cv2.putText(frame, f"(paused, {remaining:.1f}s left)", (30, 175),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                cv2.putText(frame, f"Get ready: {current_label_name}", (30, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
                cv2.putText(frame, f"{remaining:.1f}s", (30, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

            if countdown_accum >= AUTO_COUNTDOWN_SEC:
                base_sequence, pose_flags, detection_sequence = [], [], []
                mode = "RECORD"
                timer_start = time.time()

        elif mode == "RECORD":
            base, detections, boxes, edge_dbg, lid_score = extract_base_features(
                pose_res, hand_res, frame, override_boxes=yolo_boxes
            )
            base_sequence.append(base)
            detection_sequence.append(detections)
            pose_flags.append(pose_res.pose_landmarks is not None)

            remaining = max(0.0, AUTO_RECORD_SEC - elapsed)
            cv2.putText(frame, f"RECORDING: {current_label_name}", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            cv2.putText(frame, f"{remaining:.1f}s", (30, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, f"Lid complexity: {lid_score:.3f}", (30, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if elapsed >= AUTO_RECORD_SEC:
                detection_rate = sum(pose_flags) / len(pose_flags) if pose_flags else 0.0
                if detection_rate < AUTO_MIN_POSE_DETECTION_RATE:
                    print(f"  [discard] pose detected in only {detection_rate:.0%} of frames - retrying")
                    mode = "COUNTDOWN"
                    timer_start = time.time()
                    countdown_accum = 0.0
                else:
                    base_resampled = resample_base_sequence(base_sequence, SEQ_LEN)
                    fused = add_velocity(base_resampled)

                    red_rate = sum(d["red"] for d in detection_sequence) / len(detection_sequence)
                    blue_rate = sum(d["blue"] for d in detection_sequence) / len(detection_sequence)
                    main_rate = sum(d["main"] for d in detection_sequence) / len(detection_sequence)

                    sample_idx = next_sample_index(current_label_id)
                    fname = f"label_{current_label_id}_{sample_idx}.npy"
                    fpath = os.path.join(OUTPUT_DIR, fname)
                    np.save(fpath, fused)
                    append_metadata([fname, current_label_id, current_label_name,
                                      round(red_rate, 2), round(blue_rate, 2), round(main_rate, 2),
                                      SEQ_LEN, time.time()])
                    print(f"  [+] Saved {fname}  (pose {detection_rate:.0%}, red {red_rate:.0%}, "
                          f"blue {blue_rate:.0%}, main {main_rate:.0%}; {len(base_sequence)} raw "
                          f"frames -> resampled to {SEQ_LEN})")

                    last_saved_path = fpath
                    last_saved_label_idx = label_idx
                    sample_count += 1
                    mode = "REST"
                    timer_start = time.time()

        elif mode == "REST":
            cv2.putText(frame, "Rest...", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 3)
            if elapsed >= AUTO_REST_SEC:
                if sample_count >= samples_per_label:
                    label_idx += 1
                    sample_count = 0
                    if label_idx >= len(ACTIONS):
                        print("\n✅ ALL DATA COLLECTED")
                        break
                    print(f"\n➡️  NEXT ACTION: {ACTIONS[label_idx][1]}")
                mode = "COUNTDOWN"
                timer_start = time.time()
                countdown_accum = 0.0

        # Visual feedback: landmarks and bounding boxes
        if pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if hand_res.multi_hand_landmarks:
            for h in hand_res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        active_boxes = boxes if (mode == "RECORD" and boxes is not None) else yolo_boxes
        if active_boxes is not None:
            for box, color, b_name in ((active_boxes[0], (0, 0, 255), "RED"),
                                       (active_boxes[1], (255, 0, 0), "BLUE"),
                                       (active_boxes[2], (0, 255, 0), "MAIN")):
                if box is not None:
                    x, y, w, h = box["rect"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    score_str = f" {box['score']:.2f}" if "score" in box and box["score"] is not None else ""
                    cv2.putText(frame, f"{b_name}{score_str}", (x, max(20, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.putText(frame, f"Action: {current_label_name}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Sample: {sample_count}/{samples_per_label}", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Mode: {mode}", (30, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Auto Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            if last_saved_path is not None and last_saved_label_idx == label_idx:
                try:
                    os.remove(last_saved_path)
                    remove_metadata_row(os.path.basename(last_saved_path))
                    print(f"  🔁 Deleted {os.path.basename(last_saved_path)} (file + metadata row), redoing this sample")
                except OSError as e:
                    print(f"  ⚠ Could not delete {last_saved_path}: {e}")
                sample_count = max(0, sample_count - 1)
                last_saved_path = None
            else:
                print("  🔁 Nothing saved yet for this label - restarting current attempt")
            base_sequence, pose_flags, detection_sequence = [], [], []
            mode = "COUNTDOWN"
            timer_start = time.time()
            countdown_accum = 0.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Pose & Object Feature Extractor")
    parser.add_argument("--auto", action="store_true",
                         help="Run the automated countdown/record/rest collector")
    parser.add_argument("--manual", action="store_true",
                         help="Force manual mode, overriding DEFAULT_TO_AUTO_MODE")
    parser.add_argument("--samples", "--reps", type=int, default=AUTO_SAMPLES_PER_LABEL,
                         help="Number of repetitions per action in auto mode (default: 5)")
    args, _unknown = parser.parse_known_args()  # ignore unrecognized args an IDE might inject

    if args.manual:
        run_auto = False
    elif args.auto:
        run_auto = True
    else:
        run_auto = DEFAULT_TO_AUTO_MODE

    print(f"[Launch] mode={'AUTO' if run_auto else 'MANUAL'}  "
          f"(DEFAULT_TO_AUTO_MODE={DEFAULT_TO_AUTO_MODE}, --auto={args.auto}, --manual={args.manual}, reps={args.samples})")
    main_auto(samples_per_label=args.samples) if run_auto else main()