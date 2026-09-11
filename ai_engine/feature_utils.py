import numpy as np
import cv2

# ==============================================================================
# LABELS (7 classes — matching pose_extract.py / pose_extract_advanced.py)
# ==============================================================================
LABELS = [
    "idle",
    "open_box",
    "pick_red",
    "place_red_out",
    "pick_blue",
    "place_blue_in",
    "close_box"
]

# ==============================================================================
# NORMALIZATION HELPERS
# ==============================================================================

def normalize_pose(landmarks):
    """Extracts 66-dim pose feature (33 landmarks × 2), hip-centered and torso-scaled."""
    if landmarks is None:
        return np.zeros(66)

    pts = np.array([[lm.x, lm.y] for lm in landmarks.landmark])

    # use hips as center
    center = (pts[23] + pts[24]) / 2
    pts = pts - center

    # torso scale
    scale = np.linalg.norm(pts[11] - pts[12]) + 1e-6
    pts = pts / scale

    return pts.flatten()


def normalize_hand(hand_landmarks):
    """Extracts 42-dim hand feature (21 landmarks × 2), wrist-centered and palm-scaled."""
    if hand_landmarks is None:
        return np.zeros(42)

    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

    # wrist center
    center = pts[0]
    pts = pts - center

    scale = np.linalg.norm(pts[0] - pts[9]) + 1e-6
    pts = pts / scale

    return pts.flatten()


# ==============================================================================
# COLOR OBJECT DETECTION (HSV-based)
# ==============================================================================

def detect_color_objects(frame):
    """
    Detects red and blue OBJECTS (boxes) via HSV thresholding.

    Three-stage false-positive rejection to avoid detecting threads, bracelets,
    clothing patches, or other small/thin coloured items:
      1. MIN_AREA    : contour must be >= MIN_AREA px² (threads are tiny ~200-800px²)
      2. ASPECT RATIO: bounding box w/h must be between 0.25-4.0 (threads are very
                       elongated; a box is roughly square)
      3. SOLIDITY    : filled_area / convex_hull_area >= 0.50 (threads are sparse
                       line-like shapes; boxes are solid rectangles)

    Returns:
        dict: e.g. {"red": {"center": (cx, cy), "area": float, "box": (x,y,w,h)},
                     "blue": {"center": (cx, cy), "area": float, "box": (x,y,w,h)}}
    """
    MIN_AREA      = 1000   # px² — rejects threads/bracelets (~200-800px²)
    MIN_DIMENSION = 30     # px  — both width and height must exceed this
    MIN_SOLIDITY  = 0.50   # ratio — rejects thin, sparse shapes
    ASPECT_MIN    = 0.25   # width/height — rejects extreme tall-thin shapes
    ASPECT_MAX    = 4.00   # width/height — rejects extreme wide-thin shapes

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = {}

    # --- RED (wraps around H=0, so two ranges) ---
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([160, 100, 80]), np.array([180, 255, 255]))
    mask_red = mask_red1 | mask_red2

    # --- BLUE / CADBURY SILK PURPLE ---
    mask_blue = cv2.inRange(hsv, np.array([95, 30, 25]), np.array([170, 255, 255]))

    for color_name, mask in [("red", mask_red), ("blue", mask_blue)]:
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # 1. Area gate — threads/bracelets are far too small
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(largest)

        # 2. Minimum dimension — both sides must be substantial
        if w < MIN_DIMENSION or h < MIN_DIMENSION:
            continue

        # 3. Aspect ratio — rejects elongated thread/strip shapes
        aspect = w / (h + 1e-6)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        # 4. Solidity — rejects sparse line-like shapes (threads score ~0.1-0.3)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        if solidity < MIN_SOLIDITY:
            continue

        cx = x + w // 2
        cy = y + h // 2
        detections[color_name] = {
            "center": (cx, cy),
            "area": area,
            "box": (x, y, w, h)
        }

    return detections


# ==============================================================================
# OBJECT FEATURES (6-dim)
# ==============================================================================

def extract_object_features(color_dets, frame_w, frame_h):
    """
    Returns 6-dim: (red_cx, red_cy, red_area, blue_cx, blue_cy, blue_area)
    All normalized to [0, 1] range.
    """
    frame_area = frame_w * frame_h + 1e-6

    feats = []
    for color in ["red", "blue"]:
        if color in color_dets:
            cx, cy = color_dets[color]["center"]
            area = color_dets[color]["area"]
            feats.extend([cx / frame_w, cy / frame_h, area / frame_area])
        else:
            feats.extend([0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float64)


# ==============================================================================
# HAND-OBJECT DISTANCE FEATURES (4-dim)
# ==============================================================================

def extract_distance_features(pose_landmarks, color_dets, frame_w, frame_h):
    """
    Returns 4-dim: distances from (left_wrist, right_wrist) to (red, blue).
    Distances are normalized by frame diagonal.
    """
    diag = np.sqrt(frame_w**2 + frame_h**2) + 1e-6

    # Get wrist positions in pixel coords
    if pose_landmarks is not None:
        lm = pose_landmarks.landmark
        left_wrist = np.array([lm[15].x * frame_w, lm[15].y * frame_h])
        right_wrist = np.array([lm[16].x * frame_w, lm[16].y * frame_h])
    else:
        left_wrist = np.array([0.0, 0.0])
        right_wrist = np.array([0.0, 0.0])

    dists = []
    for color in ["red", "blue"]:
        if color in color_dets:
            obj_center = np.array(color_dets[color]["center"], dtype=np.float64)
        else:
            obj_center = np.array([0.0, 0.0])

        d_left = np.linalg.norm(left_wrist - obj_center) / diag
        d_right = np.linalg.norm(right_wrist - obj_center) / diag

        dists.extend([d_left, d_right])

    return np.array(dists, dtype=np.float64)


# ==============================================================================
# FULL FEATURE EXTRACTION (320-dim with velocity)
# ==============================================================================

# Internal state for velocity computation
_prev_base_feat = None


def reset_velocity_state():
    """Call this when starting a new sequence/recording to reset velocity memory."""
    global _prev_base_feat
    _prev_base_feat = None


def extract_full_features(pose_res, hands_res, color_dets, frame_w, frame_h):
    """
    Extracts the complete 320-dim feature vector:
        Base (160): pose(66) + left_hand(42) + right_hand(42) + objects(6) + distances(4)
        Velocity (160): feature(t) - feature(t-1)

    Args:
        pose_res: MediaPipe pose result
        hands_res: MediaPipe hands result
        color_dets: dict from detect_color_objects()
        frame_w: frame width in pixels
        frame_h: frame height in pixels

    Returns:
        np.ndarray of shape (320,)
    """
    global _prev_base_feat

    # Pose (66)
    pose_feat = normalize_pose(pose_res.pose_landmarks)

    # Hands (84 = 42 + 42)
    left_hand = None
    right_hand = None

    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_lms, handedness in zip(
            hands_res.multi_hand_landmarks,
            hands_res.multi_handedness
        ):
            label = handedness.classification[0].label
            if label == "Left":
                left_hand = hand_lms
            else:
                right_hand = hand_lms

    left_feat = normalize_hand(left_hand)
    right_feat = normalize_hand(right_hand)

    # Objects (6)
    obj_feat = extract_object_features(color_dets, frame_w, frame_h)

    # Distances (4)
    dist_feat = extract_distance_features(
        pose_res.pose_landmarks, color_dets, frame_w, frame_h
    )

    # Base feature (160)
    base_feat = np.concatenate([pose_feat, left_feat, right_feat, obj_feat, dist_feat])

    # Velocity (160)
    if _prev_base_feat is None:
        vel = np.zeros_like(base_feat)
    else:
        vel = base_feat - _prev_base_feat

    _prev_base_feat = base_feat.copy()

    # Final (320)
    return np.concatenate([base_feat, vel])


# ==============================================================================
# LEGACY SUPPORT: 150-dim → 320-dim conversion
# ==============================================================================

def pad_old_features(data_150):
    """
    Converts old 150-dim data (T, 150) to new 320-dim format (T, 320).

    Layout:
        Old 150 = pose(66) + left_hand(42) + right_hand(42)
        New 320 = [pose(66) + left(42) + right(42) + obj(6) + dist(4)] + velocity(160)

    Object/distance channels are zero-padded (no object info in old data).
    Velocity is computed frame-by-frame from the padded base features.
    """
    T = data_150.shape[0]
    assert data_150.shape[1] == 150, f"Expected 150 columns, got {data_150.shape[1]}"

    # Pad base: 150 → 160 (add 10 zeros for obj + dist)
    base_padded = np.hstack([data_150, np.zeros((T, 10), dtype=data_150.dtype)])

    # Compute velocity
    vel = np.zeros_like(base_padded)
    vel[1:] = base_padded[1:] - base_padded[:-1]

    # Combine: 160 + 160 = 320
    return np.hstack([base_padded, vel])


# ==============================================================================
# TEMPORAL RESAMPLING
# ==============================================================================

def resample_sequence(sequence, target_len):
    """
    Resamples a variable-length sequence to exactly target_len frames
    using linear interpolation.

    Args:
        sequence: list or ndarray of shape (T, D)
        target_len: desired output length

    Returns:
        np.ndarray of shape (target_len, D)
    """
    data = np.array(sequence, dtype=np.float32)
    T, D = data.shape

    if T == target_len:
        return data

    indices = np.linspace(0, T - 1, target_len)
    resampled = np.zeros((target_len, D), dtype=np.float32)

    for i, idx in enumerate(indices):
        low = int(np.floor(idx))
        high = min(low + 1, T - 1)
        frac = idx - low
        resampled[i] = data[low] * (1 - frac) + data[high] * frac

    return resampled