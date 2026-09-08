"""
Automated (COUNTDOWN -> RECORDING -> REST) data collector - hands-free
once started, cycling through all 7 actions automatically instead of
needing a manual key press + 's' for every single sample. Matches the
guided state-machine style described for pose_extract.py in your
architecture doc.

Reuses ALL detection/normalization/feature logic from
pose_extract_advanced.py by importing it directly (not re-implementing
it) - there's exactly one source of truth for what a feature vector
means. This script only adds the automated timing/state-machine layer.

Per sample:
  1. COUNTDOWN (default 3s) - on-screen countdown, time to get into
     position for the upcoming action.
  2. RECORDING (default 3s, overridable per-action) - captures every
     frame's base features at whatever the camera's native frame rate
     is, THEN RESAMPLES to exactly SEQ_LEN frames via linear
     interpolation. This means you don't need to know or guess your
     camera's fps, and a recording window only needs to comfortably
     CONTAIN the action, not match its length exactly.
  3. AUTO QUALITY CHECK - if a person wasn't detected in enough of the
     raw captured frames (default: under 70%), the sample is discarded
     and retried automatically (up to MAX_RETRIES) instead of silently
     saving a bad clip - the "discard bad takes" principle from manual
     collection, adapted for an unattended loop.
  4. REST (default 1.5s) - brief pause before the next sample.

Usage:
    python pose_extract_advanced_auto.py

Press 'q' at any point to stop early - whatever's already saved stays.
For full manual control over any single sample (e.g. redoing one you
didn't like the look of), pose_extract_advanced.py is still there.
"""

import os
import time

import cv2
import numpy as np
import mediapipe as mp

from pose_extract_advanced import (
    ACTIONS, BASE_DIM, TOTAL_DIM, SEQ_LEN,
    extract_base_features, next_sample_index, ensure_metadata, append_metadata,
    OUTPUT_DIR, CAMERA_ID, FLIP_FRAME,
)

# ================= AUTOMATION CONFIG =================
SAMPLES_PER_ACTION = 15
COUNTDOWN_SECONDS = 3
ACTION_DURATION = 3.0            # default seconds of RECORDING per sample
ACTION_DURATION_OVERRIDES = {}   # e.g. {1: 4.0} to give open_box more time
REST_SECONDS = 1.5
MIN_POSE_DETECTION_RATE = 0.70   # below this fraction of frames, auto-discard and retry
MAX_RETRIES = 3


def resample_base_sequence(base_list, target_len):
    """
    Uniformly resamples a variable-length list of BASE_DIM vectors to
    exactly target_len frames via per-dimension linear interpolation.
    This decouples the saved sequence length from the camera's native
    frame rate or minor timing variation in how long an action actually
    took within the recording window.
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
    """
    Computes frame-to-frame velocity on the already-resampled series, so
    velocity reflects motion on the canonical SEQ_LEN grid rather than
    the camera's raw, possibly-uneven native frame timing.
    """
    velocity = np.zeros_like(base_resampled)
    velocity[1:] = base_resampled[1:] - base_resampled[:-1]
    return np.concatenate([base_resampled, velocity], axis=1)


def _draw_hud(frame, text, y=50, color=(0, 255, 255), scale=1.1):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 3)


def countdown_phase(cap, seconds, upcoming_label):
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            return False
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)
        remaining = seconds - (time.time() - start)
        _draw_hud(frame, f"Get ready: {upcoming_label}", 50, (0, 255, 255))
        _draw_hud(frame, f"Starting in {remaining:.1f}s", 100, (0, 200, 255), 0.9)
        cv2.imshow("Automated Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


def recording_phase(cap, pose, hands, duration, label_name):
    base_list = []
    pose_flags = []
    detection_dicts = []
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        hand_res = hands.process(rgb)

        base, detections, boxes, edge_dbg, _ = extract_base_features(pose_res, hand_res, frame)
        base_list.append(base)
        detection_dicts.append(detections)
        pose_flags.append(pose_res.pose_landmarks is not None)

        remaining = duration - (time.time() - start)
        _draw_hud(frame, f"RECORDING: {label_name}", 50, (0, 0, 255))
        _draw_hud(frame, f"{remaining:.1f}s left", 100, (0, 0, 255), 0.9)
        cv2.imshow("Automated Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return None, None, None

    return base_list, pose_flags, detection_dicts


def rest_phase(cap, seconds):
    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            return False
        if FLIP_FRAME:
            frame = cv2.flip(frame, 1)
        _draw_hud(frame, "REST", 50, (200, 200, 200))
        cv2.imshow("Automated Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True


def main():
    try:
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
    except AttributeError:
        raise RuntimeError(
            "This mediapipe install doesn't expose the legacy `mp.solutions` API. "
            "Pin an older version: pip install mediapipe==0.10.9"
        )

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
    ensure_metadata()

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    total_target = SAMPLES_PER_ACTION * len(ACTIONS)
    print(f"[AutoCollector] {SAMPLES_PER_ACTION} samples x {len(ACTIONS)} actions = {total_target} target samples")
    print("Press 'q' at any time to stop early - everything saved so far stays.\n")

    stop = False
    for label_id, label_name in ACTIONS:
        if stop:
            break
        duration = ACTION_DURATION_OVERRIDES.get(label_id, ACTION_DURATION)
        print(f"\n=== {label_name} (label {label_id}) - target {SAMPLES_PER_ACTION} samples, {duration}s each ===")

        collected = 0
        while collected < SAMPLES_PER_ACTION and not stop:
            attempt = 0
            accepted = False

            while attempt < MAX_RETRIES and not accepted:
                attempt += 1
                if not countdown_phase(cap, COUNTDOWN_SECONDS, label_name):
                    stop = True
                    break

                base_list, pose_flags, detection_dicts = recording_phase(cap, pose, hands, duration, label_name)
                if base_list is None:
                    stop = True
                    break

                detection_rate = sum(pose_flags) / len(pose_flags) if pose_flags else 0.0
                if detection_rate < MIN_POSE_DETECTION_RATE:
                    print(f"  [retry {attempt}/{MAX_RETRIES}] pose detected in only "
                          f"{detection_rate:.0%} of frames - discarding, retrying")
                    continue

                base_resampled = resample_base_sequence(base_list, SEQ_LEN)
                fused = add_velocity(base_resampled)
                assert fused.shape == (SEQ_LEN, TOTAL_DIM)

                red_rate = sum(d["red"] for d in detection_dicts) / len(detection_dicts)
                blue_rate = sum(d["blue"] for d in detection_dicts) / len(detection_dicts)
                main_rate = sum(d["main"] for d in detection_dicts) / len(detection_dicts)

                sample_idx = next_sample_index(label_id)
                fname = f"label_{label_id}_{sample_idx}.npy"
                np.save(os.path.join(OUTPUT_DIR, fname), fused)
                append_metadata([fname, label_id, label_name,
                                  round(red_rate, 2), round(blue_rate, 2), round(main_rate, 2),
                                  SEQ_LEN, time.time()])
                print(f"  [+] Saved {fname}  (pose {detection_rate:.0%}, red {red_rate:.0%}, "
                      f"blue {blue_rate:.0%}, main {main_rate:.0%}; {len(base_list)} raw frames "
                      f"-> resampled to {SEQ_LEN})")
                accepted = True
                collected += 1

            if attempt >= MAX_RETRIES and not accepted and not stop:
                print(f"  [!] Gave up after {MAX_RETRIES} retries - check the person is visible "
                      f"to the camera. Skipping this sample slot.")
                collected += 1

            if not stop:
                if not rest_phase(cap, REST_SECONDS):
                    stop = True

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Automated collection finished" + (" (stopped early)" if stop else "") + ".")


if __name__ == "__main__":
    main()