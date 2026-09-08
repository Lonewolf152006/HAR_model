"""
TAR Model Video Test Runner & Performance Evaluator

Brings test_video_tar.py up-to-date with realtime.py:
  1. Level 1: 2.5D Geometric Box Containment Engine (Red/Blue inside/outside/held + 3D wireframe).
  2. Dual-Path Inference: Motion Action Spotter (peak scoring on gesture settle) + Decision Stabilizer.
  3. Physical Causal Logic: strict sequential gating preventing domino-rushing.
  4. Real-time diagnostic HUD: state badges, next expected step, 2.5D containment status,
     Motion Spotter telemetry widget, and SOP step checklist.
  5. Deterministic video evaluation with comprehensive console and markdown report generation.

Usage:
    # Run on default/first available video in videos/
    python test_video_tar.py

    # Run on a specific video
    python test_video_tar.py --video videos/1.mp4

    # Run headless (fast, without GUI) and save performance report
    python test_video_tar.py --video videos/1.mp4 --headless

    # Run and save annotated video output
    python test_video_tar.py --video videos/1.mp4 --save-video output_test1.mp4

    # Test only the first N frames
    python test_video_tar.py --video videos/1.mp4 --max-frames 300

Controls in Viewer:
    SPACE        : Play / Pause
    D / Right    : Jump +30 frames forward
    A / Left     : Jump -30 frames backward
    S            : Toggle playback speed (1.0x -> 2.0x -> 0.5x)
    O            : Toggle probability distribution overlay
    L            : Toggle skeleton / hand landmarks
    B            : Toggle bounding boxes & 2.5D container wireframe
    R            : Reset SOP cycle, physical state & containment
    Q / ESC      : Finish test & generate report
"""

import os
import sys
import time
import json
import argparse
from collections import deque, Counter
from datetime import datetime

import cv2
import numpy as np
import torch

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model_def import TARModel, NUM_CLASSES, FEATURE_DIM, SEQ_LEN
import feature_utils as fu
from pose_extract_advanced import extract_base_features, BASE_DIM

from realtime import (
    GeometricContainmentEngine,
    PhysicalCausalLogic,
    MotionActionSpotter,
    DecisionStabilizer,
    ground_probs_with_yolo,
    process_frame,
    predict_from_window,
    compute_motion,
    BoxTracker,
    load_tar_model,
    load_yolo,
    run_yolo,
    CONFIDENCE_THRESHOLD,
    STABILITY_WINDOW,
    COOLDOWN_SEC,
    MOTION_THRESHOLD,
)

# ==============================================================================
# 🎬 [VIDEO CONFIGURATION] - PASTE YOUR VIDEO PATH HERE
# ==============================================================================
# Simply paste the path to your video inside the quotes below!
#
# Examples:
#   VIDEO_PATH = r"videos/1.mp4"
#   VIDEO_PATH = r"C:\Users\Vedant\Videos\my_recording.mp4"
#
# Tip: On Windows, you can right-click any video -> "Copy as path",
# and simply paste it directly inside the quotes!
# ==============================================================================
VIDEO_PATH = r"videos/1.mp4"


def clean_path(path_str):
    """Cleans up paths pasted with quotes on Windows (e.g. "C:\\path\\video.mp4")."""
    if not path_str:
        return ""
    p = str(path_str).strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()
    return p


# Color palette for classes (BGR)
CLASS_COLORS = {
    "idle": (140, 140, 140),          # Gray
    "open_box": (0, 215, 255),        # Amber / Yellow
    "pick_red": (60, 60, 255),        # Red
    "place_red_out": (180, 105, 255), # Pink/Magenta
    "pick_blue": (255, 144, 30),      # Blue
    "place_blue_in": (255, 215, 0),   # Cyan
    "close_box": (50, 205, 50),       # Green
}

SOP_STEPS = [
    ("open_box", "1. Open Box"),
    ("pick_red", "2. Pick Red"),
    ("place_red_out", "3. Place Red Out"),
    ("pick_blue", "4. Pick Blue"),
    ("place_blue_in", "5. Place Blue In"),
    ("close_box", "6. Close Box"),
]

DEFAULT_MODEL_PATH = "best_tar_model1.pth" if os.path.exists(os.path.join(PROJECT_ROOT, "best_tar_model1.pth")) else "best_tar_model.pth"
DEFAULT_YOLO_PATH = "yolo_boxes.pt"


# ==============================================================================
# VIDEO SOP FLOW TRACKER
# ==============================================================================
class VideoSOPTracker:
    """Tracks SOP compliance, step times, and cycle history across video playback."""
    def __init__(self):
        self.current_step = 0
        self.completed = [False] * len(SOP_STEPS)
        self.step_times = [None] * len(SOP_STEPS)
        self.cycle_count = 0
        self.cycle_history = []
        self.cycle_start_time = None
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

    def update(self, state, vid_time):
        if state is None or state == "idle":
            return

        expected = self.expected_action

        # Sequential advance
        if state == expected:
            if self.current_step == 0 and self.cycle_start_time is None:
                self.cycle_start_time = vid_time

            self.completed[self.current_step] = True
            self.step_times[self.current_step] = vid_time

            if self.current_step < len(SOP_STEPS) - 1:
                self.current_step += 1
            else:
                self.cycle_count += 1
                self.cycle_complete_time = vid_time
                cycle_dur = (vid_time - self.cycle_start_time) if self.cycle_start_time else 0.0
                self.cycle_history.append({
                    "cycle": self.cycle_count,
                    "duration": round(cycle_dur, 2),
                    "end_time": round(vid_time, 2)
                })
                self.reset()
            return

        # Automatic new cycle start on re-opening box after previous cycle completed
        if state == "open_box" and self.current_step > 0 and (vid_time - self.cycle_complete_time) > 1.5:
            self.reset()
            self.completed[0] = True
            self.step_times[0] = vid_time
            self.current_step = 1
            self.cycle_start_time = vid_time

    def reset(self):
        self.current_step = 0
        self.completed = [False] * len(SOP_STEPS)
        self.cycle_start_time = None


# ==============================================================================
# PERFORMANCE METRICS RECORDER
# ==============================================================================
class VideoTestMetrics:
    """Collects frame-by-frame telemetries, transitions, and generates statistics."""
    def __init__(self, video_path, total_frames, fps):
        self.video_path = video_path
        self.total_frames = total_frames
        self.fps = fps
        self.duration_sec = total_frames / max(1.0, fps)

        self.frames_evaluated = 0
        self.frame_records = []
        self.state_timeline = []
        self.current_segment = None

        self.class_time_counts = Counter()
        self.confidence_list = []
        self.jitter_count = 0

    def record_frame(self, frame_idx, vid_time, raw_probs, pred_label, conf, state, motion, status):
        self.frames_evaluated += 1
        if pred_label is not None:
            self.confidence_list.append(conf)

        active_state = state if state is not None else "idle"
        self.class_time_counts[active_state] += 1

        rec = {
            "frame": frame_idx,
            "time": vid_time,
            "pred": pred_label,
            "conf": conf,
            "state": active_state,
            "probs": raw_probs.tolist() if raw_probs is not None else [],
            "motion": motion,
            "status": status
        }
        self.frame_records.append(rec)

        # Update timeline segments
        if self.current_segment is None:
            self.current_segment = {
                "state": active_state,
                "start_time": vid_time,
                "start_frame": frame_idx,
                "end_time": vid_time,
                "end_frame": frame_idx,
                "confs": [conf]
            }
        elif self.current_segment["state"] == active_state:
            self.current_segment["end_time"] = vid_time
            self.current_segment["end_frame"] = frame_idx
            self.current_segment["confs"].append(conf)
        else:
            dur = self.current_segment["end_time"] - self.current_segment["start_time"]
            self.current_segment["duration"] = max(0.01, dur)
            self.current_segment["avg_conf"] = float(np.mean(self.current_segment["confs"]))
            self.current_segment["min_conf"] = float(np.min(self.current_segment["confs"]))
            self.current_segment["max_conf"] = float(np.max(self.current_segment["confs"]))

            if self.current_segment["duration"] < 0.35 and self.current_segment["state"] != "idle":
                self.jitter_count += 1

            self.state_timeline.append(self.current_segment)

            self.current_segment = {
                "state": active_state,
                "start_time": vid_time,
                "start_frame": frame_idx,
                "end_time": vid_time,
                "end_frame": frame_idx,
                "confs": [conf]
            }

    def finalize(self):
        if self.current_segment is not None and self.current_segment not in self.state_timeline:
            dur = self.current_segment["end_time"] - self.current_segment["start_time"]
            self.current_segment["duration"] = max(0.01, dur)
            self.current_segment["avg_conf"] = float(np.mean(self.current_segment["confs"]))
            self.current_segment["min_conf"] = float(np.min(self.current_segment["confs"]))
            self.current_segment["max_conf"] = float(np.max(self.current_segment["confs"]))
            self.state_timeline.append(self.current_segment)

    def generate_summary(self, sop_tracker):
        self.finalize()

        confs = np.array(self.confidence_list) if self.confidence_list else np.array([0.0])
        avg_conf = float(np.mean(confs))
        high_conf_pct = float(np.mean(confs >= 0.80) * 100)
        med_conf_pct = float(np.mean((confs >= 0.60) & (confs < 0.80)) * 100)
        low_conf_pct = float(np.mean(confs < 0.60) * 100)

        total_tracked_frames = max(1, sum(self.class_time_counts.values()))
        class_distribution = {}
        for label in fu.LABELS:
            cnt = self.class_time_counts[label]
            class_distribution[label] = {
                "frames": cnt,
                "seconds": round(cnt / self.fps, 2),
                "percentage": round((cnt / total_tracked_frames) * 100, 1)
            }

        # Analyze low confidence ambiguities
        confusions = Counter()
        for r in self.frame_records:
            if r["conf"] < 0.65 and len(r["probs"]) == NUM_CLASSES:
                top2 = np.argsort(r["probs"])[-2:]
                pair = f"{fu.LABELS[top2[1]]} vs {fu.LABELS[top2[0]]}"
                confusions[pair] += 1

        top_ambiguities = [{"pair": k, "count": v} for k, v in confusions.most_common(3)]

        sop_summary = {
            "cycles_completed": sop_tracker.cycle_count,
            "cycle_history": sop_tracker.cycle_history,
            "current_cycle_steps": [
                {"step": title, "action": action, "completed": sop_tracker.completed[i], "time": sop_tracker.step_times[i]}
                for i, (action, title) in enumerate(SOP_STEPS)
            ]
        }

        return {
            "video_path": self.video_path,
            "total_video_frames": self.total_frames,
            "frames_evaluated": self.frames_evaluated,
            "video_duration_sec": round(self.duration_sec, 2),
            "video_fps": round(self.fps, 2),
            "overall_avg_confidence": round(avg_conf, 3),
            "high_confidence_pct": round(high_conf_pct, 1),
            "med_confidence_pct": round(med_conf_pct, 1),
            "low_confidence_pct": round(low_conf_pct, 1),
            "jitter_short_flips": self.jitter_count,
            "class_distribution": class_distribution,
            "timeline": self.state_timeline,
            "sop_summary": sop_summary,
            "top_ambiguities": top_ambiguities
        }


# ==============================================================================
# VISUAL HUD & OVERLAYS RENDERER
# ==============================================================================
def draw_test_hud(frame, vid_time, total_time, frame_idx, total_frames, fps,
                  state, pred_label, conf, status, raw_probs,
                  sop_tracker, flags, is_paused, speed_mult,
                  stabilizer=None, containment=None, spotter_status=None,
                  det_str=""):
    """Renders state-of-the-art telemetry HUD on top of the video frame."""
    h, w = frame.shape[:2]

    # --- TOP HEADER BAR ---
    top_bar_h = 82
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, top_bar_h), (w, top_bar_h), (60, 60, 60), 1)

    # Active State Badge
    state_name = state if state is not None else "idle"
    state_color = CLASS_COLORS.get(state_name, (0, 255, 0))

    cv2.putText(frame, "STATE:", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    cv2.putText(frame, state_name.upper(), (95, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, state_color, 2)

    # Next expected SOP action
    exp_title = sop_tracker.expected_action_title if sop_tracker else "Complete"
    cv2.putText(frame, f"Next: {exp_title}", (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    # Telemetry Subtitle
    conf_pct = int(conf * 100)
    pred_display = pred_label if pred_label else "..."
    latch_cnt = len(stabilizer.recent) if stabilizer else 0
    stab_win = stabilizer.stability_window if stabilizer else STABILITY_WINDOW
    status_display = f" | {status}" if status else ""
    telemetry = f"Pred: {pred_display} ({conf_pct}%) | Latch: {latch_cnt}/{stab_win}{status_display}"
    cv2.putText(frame, telemetry, (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)

    # Video Time & Playback info on top-right
    cur_m, cur_s = divmod(int(vid_time), 60)
    tot_m, tot_s = divmod(int(total_time), 60)
    time_str = f"{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d}  [{frame_idx}/{total_frames}]"
    cv2.putText(frame, time_str, (w - 290, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    status_tag = f"{'PAUSED' if is_paused else 'PLAYING'} ({speed_mult:.1f}x) | {fps:.0f} FPS"
    tag_color = (0, 165, 255) if is_paused else (0, 255, 0)
    cv2.putText(frame, status_tag, (w - 290, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, tag_color, 1)

    # --- DIAGNOSTIC TELEMETRY (Left Side below header) ---
    causal_line = stabilizer.causal_logic.get_status_summary(stabilizer.dominant_color) if stabilizer else "Causal Logic"
    cv2.putText(frame, f"{causal_line} | {det_str}", (15, top_bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 255), 1)

    # 2.5D Geometric Containment readout
    if containment is not None:
        cv2.putText(frame, containment.get_summary(), (15, top_bar_h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 100), 1)

    # Motion Spotter Telemetry Widget
    if spotter_status:
        sp_mode = spotter_status["spotter_state"]
        sp_frames = spotter_status["motion_frames"]
        sp_mot = spotter_status["smooth_motion"]

        if sp_mode == "TRACKING":
            txt = f"[ACTIVE GESTURE] Frames: {sp_frames} | Velocity: {sp_mot:.3f}"
            cv2.putText(frame, txt, (15, top_bar_h + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 165, 255), 1)
            bar_w = min(150, int((sp_mot / 0.04) * 150))
            cv2.rectangle(frame, (15, top_bar_h + 68), (15 + bar_w, top_bar_h + 74), (0, 165, 255), -1)
            cv2.rectangle(frame, (15, top_bar_h + 68), (165, top_bar_h + 74), (80, 80, 80), 1)
        elif sp_mode == "COOLDOWN":
            txt = "[SPOTTER] Action Scored! Settling..."
            cv2.putText(frame, txt, (15, top_bar_h + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 180), 1)
        else:
            txt = "[SPOTTER] Ready (Hands Calm)"
            cv2.putText(frame, txt, (15, top_bar_h + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 140), 1)

    # --- SOP FLOW PANEL (Top Right) ---
    box_w, box_h = 240, 185
    px, py = w - box_w - 10, top_bar_h + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + box_w, py + box_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (px, py), (px + box_w, py + box_h), (70, 70, 70), 1)

    cv2.putText(frame, f"SOP FLOW  (Cycles: {sop_tracker.cycle_count})", (px + 10, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    for i, (action, title) in enumerate(SOP_STEPS):
        sy = py + 46 + i * 22
        t_info = f" ({sop_tracker.step_times[i]:.1f}s)" if sop_tracker.step_times[i] is not None else ""
        if sop_tracker.completed[i]:
            icon, color = "[v]", (50, 220, 50)
        elif i == sop_tracker.current_step:
            icon, color = "[>]", (0, 230, 255)
        else:
            icon, color = "[ ]", (120, 120, 120)
        cv2.putText(frame, f"{icon} {title}{t_info}", (px + 10, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    # --- PROBABILITY DISTRIBUTION OVERLAY (Left Panel) ---
    if flags.get("show_probs", True) and raw_probs is not None:
        pw, ph = 270, 185
        ox, oy = 15, top_bar_h + 84
        overlay = frame.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + pw, oy + ph), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (ox, oy), (ox + pw, oy + ph), (70, 70, 70), 1)

        cv2.putText(frame, "ACTION PROBABILITIES", (ox + 10, oy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)

        bar_max_w = 110
        for i, label in enumerate(fu.LABELS):
            prob = raw_probs[i] if i < len(raw_probs) else 0.0
            by = oy + 40 + i * 20
            color = CLASS_COLORS.get(label, (200, 200, 200))
            is_top = (i == int(np.argmax(raw_probs)))

            lbl_color = (255, 255, 255) if is_top else (170, 170, 170)
            cv2.putText(frame, f"{label[:12]:<12}", (ox + 8, by + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, lbl_color, 1)

            bw = int(prob * bar_max_w)
            cv2.rectangle(frame, (ox + 105, by), (ox + 105 + bar_max_w, by + 12), (50, 50, 50), -1)
            if bw > 0:
                cv2.rectangle(frame, (ox + 105, by), (ox + 105 + bw, by + 12), color, -1)

            pct_text = f"{int(prob * 100)}%"
            cv2.putText(frame, pct_text, (ox + 105 + bar_max_w + 6, by + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

    # --- BOTTOM TIMELINE & CONTROL BAR ---
    bot_h = 36
    by0 = h - bot_h
    cv2.rectangle(frame, (0, by0), (w, h), (18, 18, 18), -1)
    cv2.line(frame, (0, by0), (w, by0), (55, 55, 55), 1)

    prog_ratio = min(1.0, max(0.0, frame_idx / max(1, total_frames)))
    prog_w = int(prog_ratio * w)
    cv2.rectangle(frame, (0, by0), (prog_w, by0 + 4), (0, 200, 255), -1)

    controls_text = "[SPACE] Play/Pause | [D/A] +/-30f | [S] Speed | [O] Probs | [L] Pose | [B] Boxes | [R] Reset | [Q] Done"
    cv2.putText(frame, controls_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)


# ==============================================================================
# REPORT FORMATTERS (CONSOLE & MARKDOWN)
# ==============================================================================
def print_console_report(summary):
    """Outputs a formatted diagnostic report to terminal."""
    bar = "=" * 76
    sub_bar = "-" * 76

    print("\n" + bar)
    print("               TAR MODEL VIDEO PERFORMANCE REPORT")
    print(bar)

    print(f" Video Tested        : {summary['video_path']}")
    print(f" Total Duration      : {summary['video_duration_sec']}s ({summary['total_video_frames']} frames @ {summary['video_fps']} FPS)")
    print(f" Frames Evaluated    : {summary['frames_evaluated']}")
    print(f" Overall Confidence  : {summary['overall_avg_confidence'] * 100:.1f}%")
    print(f" Confidence Profile  : High (>=80%): {summary['high_confidence_pct']}% | Med: {summary['med_confidence_pct']}% | Low (<60%): {summary['low_confidence_pct']}%")
    print(f" Stability Jitter    : {summary['jitter_short_flips']} rapid flips detected")
    print(sub_bar)

    print("\n[1] SOP PACKAGING WORKFLOW AUDIT:")
    sop = summary["sop_summary"]
    print(f"  Total Cycles Completed : {sop['cycles_completed']}")
    for step in sop["current_cycle_steps"]:
        mark = "[PASS]" if step["completed"] else "[MISS]"
        t_str = f"at {step['time']:.1f}s" if step["time"] is not None else "not triggered"
        print(f"    {mark:<7} {step['step']:<20} -> {t_str}")

    print("\n[2] ACTION TIMELINE (Continuous Segments Detected):")
    print(f"  {'Start':<8} {'End':<8} {'Duration':<9} {'Action':<16} {'Avg Conf':<10} {'Peak Conf'}")
    print("  " + "-" * 62)
    for seg in summary["timeline"]:
        dur_str = f"{seg['duration']:.1f}s"
        start_str = f"{seg['start_time']:.1f}s"
        end_str = f"{seg['end_time']:.1f}s"
        print(f"  {start_str:<8} {end_str:<8} {dur_str:<9} {seg['state']:<16} {seg['avg_conf']*100:>5.1f}%     {seg['max_conf']*100:>5.1f}%")

    print("\n[3] TIME BREAKDOWN BY ACTION:")
    print(f"  {'Action':<16} {'Time (s)':<10} {'Frames':<10} {'Share %'}")
    print("  " + "-" * 45)
    for act, data in summary["class_distribution"].items():
        bar_len = int(data["percentage"] / 5)
        bar_vis = "#" * bar_len
        print(f"  {act:<16} {data['seconds']:<10} {data['frames']:<10} {data['percentage']:>5.1f}%  {bar_vis}")

    if summary["top_ambiguities"]:
        print("\n[4] UNCERTAINTY & CONFUSION DIAGNOSTICS:")
        for amb in summary["top_ambiguities"]:
            print(f"  * Frequent ambiguity during low confidence: {amb['pair']} ({amb['count']} frames)")

    print(bar + "\n")


def save_markdown_report(summary, output_file):
    """Saves a rich GitHub Markdown report file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    lines = [
        f"# TAR Model Test Report: `{os.path.basename(summary['video_path'])}`",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Video File**: `{summary['video_path']}`  ",
        f"**Duration**: {summary['video_duration_sec']}s ({summary['total_video_frames']} frames @ {summary['video_fps']} FPS)  ",
        "",
        "## 1. Executive Performance Summary",
        "",
        "| Metric | Result | Health Assessment |",
        "| :--- | :--- | :--- |",
        f"| **Overall Avg Confidence** | **{summary['overall_avg_confidence']*100:.1f}%** | {'Optimal (>=80%)' if summary['overall_avg_confidence']>=0.80 else 'Acceptable' if summary['overall_avg_confidence']>=0.65 else 'Needs Review'} |",
        f"| **High Confidence Ratio (>=80%)** | {summary['high_confidence_pct']}% | Solid recognition confidence |",
        f"| **Low Confidence Ratio (<60%)** | {summary['low_confidence_pct']}% | Uncertain frame percentage |",
        f"| **State Jitter / Fast Flips** | {summary['jitter_short_flips']} | {'Rock solid' if summary['jitter_short_flips'] <= 2 else 'Moderate flickering'} |",
        f"| **SOP Packaging Cycles** | {summary['sop_summary']['cycles_completed']} completed | Flow adherence score |",
        "",
        "---",
        "",
        "## 2. SOP Packaging Flow Compliance",
        "",
        "| Step # | Standard Operating Procedure | Status | Trigger Time |",
        "| :--- | :--- | :---: | :--- |"
    ]

    for s in summary["sop_summary"]["current_cycle_steps"]:
        st = "[x] Completed" if s["completed"] else "[ ] Missed"
        t_val = f"{s['time']:.2f}s" if s["time"] is not None else "--"
        lines.append(f"| {s['step'][:2]} | {s['step'][3:]} | {st} | {t_val} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Action Chronology (Continuous State Timeline)",
        "",
        "| Start Time | End Time | Duration | Action Detected | Avg Confidence | Peak Confidence |",
        "| :---: | :---: | :---: | :--- | :---: | :---: |"
    ])

    for seg in summary["timeline"]:
        lines.append(
            f"| {seg['start_time']:.1f}s | {seg['end_time']:.1f}s | {seg['duration']:.1f}s | "
            f"`{seg['state']}` | {seg['avg_conf']*100:.1f}% | {seg['max_conf']*100:.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Time Distribution by Action",
        "",
        "| Action Class | Duration (s) | Frame Count | Percentage Share |",
        "| :--- | :---: | :---: | :--- |"
    ])

    for act, data in summary["class_distribution"].items():
        lines.append(f"| `{act}` | {data['seconds']}s | {data['frames']} | {data['percentage']}% |")

    if summary["top_ambiguities"]:
        lines.extend([
            "",
            "---",
            "",
            "## 5. Ambiguity & Confusion Diagnostics",
            "",
            "The following action pairs had competing probabilities when confidence dipped below 65%:",
            ""
        ])
        for amb in summary["top_ambiguities"]:
            lines.append(f"- **{amb['pair']}** ({amb['count']} frames)")

    lines.append("")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================
def find_test_videos():
    """Finds candidate video files in standard locations."""
    candidates = []
    video_dir = os.path.join(PROJECT_ROOT, "videos")
    if os.path.exists(video_dir):
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            candidates.extend([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.lower().endswith(ext)])

    for ext in (".mp4", ".avi", ".mov"):
        candidates.extend([os.path.join(PROJECT_ROOT, f) for f in os.listdir(PROJECT_ROOT) if f.lower().endswith(ext)])

    return sorted(list(set(candidates)))


def run_test(video_path,
             model_path=DEFAULT_MODEL_PATH,
             yolo_path=DEFAULT_YOLO_PATH,
             headless=False,
             save_video_path=None,
             max_frames=None,
             confidence_thresh=CONFIDENCE_THRESHOLD,
             stability_window=STABILITY_WINDOW,
             cooldown_sec=COOLDOWN_SEC):

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file '{video_path}' does not exist.")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video '{video_path}'.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_time = total_frames / fps

    print("\n" + "=" * 70)
    print(f" [TAR TEST RUNNER] Starting evaluation on: {video_path}")
    print(f" Video Info: {width}x{height} @ {fps:.1f} FPS | Total: {total_frames} frames ({total_time:.1f}s)")
    print(f" Gates: conf>={confidence_thresh:.2f} | stability={stability_window} frames | cooldown={cooldown_sec:.2f}s")
    print("=" * 70)

    # MediaPipe Initialization
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
    except AttributeError:
        raise RuntimeError("MediaPipe solutions API unavailable. Pin mediapipe==0.10.9.")

    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MODEL] Loading TAR model '{model_path}' on {device}...")
    tar_model = load_tar_model(model_path)

    print(f"[YOLO] Loading YOLO detector '{yolo_path}'...")
    yolo_model = load_yolo(yolo_path)

    # Video Writer if recording requested
    writer = None
    if save_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
        print(f"[RECORDING] Will save annotated video to: {save_video_path}")

    # Core Engines matching realtime.py
    stabilizer = DecisionStabilizer(confidence_thresh, stability_window, cooldown_sec)
    spotter = MotionActionSpotter()
    box_tracker = BoxTracker(max_missing=8)
    containment = GeometricContainmentEngine(buffer_size=8)
    sop_tracker = VideoSOPTracker()
    metrics = VideoTestMetrics(video_path, total_frames, fps)

    window = deque(maxlen=SEQ_LEN)
    prev_feat = None
    prev_base = None

    flags = {
        "show_probs": True,
        "show_pose": True,
        "show_boxes": True
    }

    frame_idx = 0
    is_paused = False
    speed_mult = 1.0

    window_name = f"TAR Test: {os.path.basename(video_path)}"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    try:
        while cap.isOpened():
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    print(f"[INFO] Reached max frames limit ({max_frames}). Stopping test.")
                    break

                vid_time = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 1. Pose & Hands Detection
                pose_res = pose_detector.process(rgb)
                hand_res = hands_detector.process(rgb)

                # 2. YOLO Box Detection
                yolo_detections = run_yolo(yolo_model, frame, conf_threshold=0.20)
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

                # 3. 332-dim Feature Extraction (Base 166 + Velocity 166)
                base, det_flags, (red_box, blue_box, main_box), edge_debug, lid_score = extract_base_features(
                    pose_res, hand_res, frame, override_boxes=(yolo_red, yolo_blue, yolo_main)
                )

                # Smooth brief detection dropouts
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

                # 4. Dual-Path Temporal Action Prediction
                result = process_frame(
                    feat, motion, window, tar_model, stabilizer, spotter, vid_time,
                    yolo_red=yolo_red, yolo_blue=yolo_blue,
                    red_box=red_box, blue_box=blue_box,
                    pose_landmarks=pose_res.pose_landmarks if pose_res else None,
                    frame_shape=frame.shape,
                    expected_action=sop_tracker.expected_action,
                    containment=containment
                )

                current_state = result["current_state"]
                pred_label = result.get("predicted_label")
                conf = result["confidence"]
                status = result["status"]
                raw_probs = result.get("probs")

                if result["state_changed"] and current_state != "idle":
                    sop_tracker.update(current_state, vid_time)

                # 5. Record Metrics
                metrics.record_frame(frame_idx, vid_time, raw_probs, pred_label, conf, current_state, motion, status)

                # 6. Render Overlays
                display_frame = frame.copy()

                # Draw Bounding Boxes & 2.5D Wireframe
                if flags["show_boxes"]:
                    if red_box is not None:
                        rx, ry, rw, rh = red_box["rect"]
                        cv2.rectangle(display_frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)
                        cv2.putText(display_frame, "Red Box", (rx, max(18, ry - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                    if blue_box is not None:
                        bx, by, bw, bh = blue_box["rect"]
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
                        cv2.putText(display_frame, "Blue Box", (bx, max(18, by - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

                    if main_box is not None:
                        mx, my, mw, mh = main_box["rect"]
                        cv2.rectangle(display_frame, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)

                    # Draw 2.5D container volume wireframe
                    containment.draw(display_frame, main_box)

                    # Draw YOLO detections
                    for det in yolo_detections:
                        x1, y1, x2, y2 = [int(v) for v in det["box"]]
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                # Draw Pose / Hands Landmarks
                if flags["show_pose"]:
                    if pose_res.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            display_frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                    if hand_res.multi_hand_landmarks:
                        for h_lm in hand_res.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                display_frame, h_lm, mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
                            )

                # Detection string
                r_ok = "YES" if red_box is not None else "NO"
                b_ok = "YES" if blue_box is not None else "NO"
                m_ok = "YES" if main_box is not None else "NO"
                det_str = f"[R: {r_ok}] [B: {b_ok}] [M: {m_ok}]"

                # Draw Visual Telemetry HUD
                draw_test_hud(
                    display_frame, vid_time, total_time, frame_idx, total_frames, fps,
                    current_state, pred_label, conf, status, raw_probs,
                    sop_tracker, flags, is_paused, speed_mult,
                    stabilizer=stabilizer,
                    containment=containment,
                    spotter_status=result.get("spotter_status"),
                    det_str=det_str
                )

                if writer is not None:
                    writer.write(display_frame)

                if headless and frame_idx % 60 == 0:
                    prog_pct = (frame_idx / total_frames) * 100
                    st_str = current_state if current_state else "idle"
                    print(f" [PROGRESS] {frame_idx}/{total_frames} ({prog_pct:.1f}%) | Time: {vid_time:.1f}s | State: {st_str} ({int(conf*100)}%)")

            # GUI Mode Display & Keyboard Handling
            if not headless:
                cv2.imshow(window_name, display_frame)

                wait_ms = 1 if is_paused else max(1, int((1000.0 / fps) / speed_mult))
                key = cv2.waitKey(wait_ms) & 0xFF

                if key in (ord('q'), 27):  # Q or ESC
                    print("\n[USER] Stopping video test early via key press.")
                    break
                elif key == ord(' '):      # SPACE: Pause / Play
                    is_paused = not is_paused
                elif key in (ord('d'), 83):# D or Right Arrow (+30 frames)
                    new_pos = min(total_frames - 1, frame_idx + 30)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_idx = new_pos
                elif key in (ord('a'), 81):# A or Left Arrow (-30 frames)
                    new_pos = max(0, frame_idx - 30)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    frame_idx = new_pos
                    window.clear()
                    prev_feat, prev_base = None, None
                    stabilizer.recent.clear()
                    spotter.buffer.clear()
                    spotter.state = spotter.STATE_IDLE
                    containment.reset()
                elif key in (ord('r'), ord('R')):  # R: Reset cycle
                    stabilizer.causal_logic.reset()
                    containment.reset()
                    stabilizer.current_state = "idle"
                    spotter.state = spotter.STATE_IDLE
                    spotter.buffer.clear()
                    sop_tracker.reset()
                    print(f"[RESET] Reset physical causal state and SOP at frame {frame_idx} ({vid_time:.1f}s).")
                elif key == ord('s'):      # S: Cycle playback speed
                    speeds = [1.0, 2.0, 0.5]
                    curr_idx = speeds.index(speed_mult) if speed_mult in speeds else 0
                    speed_mult = speeds[(curr_idx + 1) % len(speeds)]
                elif key == ord('o'):      # O: Toggle Probabilities HUD
                    flags["show_probs"] = not flags["show_probs"]
                elif key == ord('l'):      # L: Toggle Skeleton Landmarks
                    flags["show_pose"] = not flags["show_pose"]
                elif key == ord('b'):      # B: Toggle Bounding Boxes
                    flags["show_boxes"] = not flags["show_boxes"]

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[OK] Saved annotated output video to: {save_video_path}")
        if not headless:
            cv2.destroyAllWindows()

    # Generate Performance Reports
    summary = metrics.generate_summary(sop_tracker)
    print_console_report(summary)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    report_md_path = os.path.join(PROJECT_ROOT, "test_reports", f"report_{base_name}.md")
    report_json_path = os.path.join(PROJECT_ROOT, "test_reports", f"report_{base_name}.json")

    save_markdown_report(summary, report_md_path)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVED] Performance Markdown Report : {report_md_path}")
    print(f"[SAVED] Machine-Readable JSON Data : {report_json_path}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Test TAR model on videos with 2.5D containment & gesture spotting.")
    parser.add_argument("--video", "-v", default=None, help="Path to test video file (e.g. videos/1.mp4).")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_PATH, help="Path to TAR model checkpoint.")
    parser.add_argument("--yolo", "-y", default=DEFAULT_YOLO_PATH, help="Path to YOLO box detector.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display (faster processing).")
    parser.add_argument("--save-video", default=None, help="Path to save annotated output video.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to test.")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold for state transition.")
    parser.add_argument("--stability", type=int, default=STABILITY_WINDOW, help="Temporal stability window (frames).")
    parser.add_argument("--cooldown", type=float, default=COOLDOWN_SEC, help="Cooldown between action transitions (sec).")

    args = parser.parse_args()

    # 1. First priority: CLI argument (--video)
    video_path = clean_path(args.video)

    # 2. Second priority: VIDEO_PATH set in code at line ~90
    if not video_path and VIDEO_PATH:
        cleaned = clean_path(VIDEO_PATH)
        if os.path.isabs(cleaned) and os.path.exists(cleaned):
            video_path = cleaned
        elif os.path.exists(os.path.join(PROJECT_ROOT, cleaned)):
            video_path = os.path.join(PROJECT_ROOT, cleaned)
        elif os.path.exists(cleaned):
            video_path = os.path.abspath(cleaned)
        else:
            print(f"[WARN] File specified in VIDEO_PATH not found: '{VIDEO_PATH}'")

    # 3. Fallback: Auto-detect available video in videos/
    if not video_path:
        candidates = find_test_videos()
        if not candidates:
            print("[ERROR] No video specified and no video found in 'videos/' or project folder.")
            print("        Open test_video_tar.py and paste your video path at VIDEO_PATH (line 90).")
            return
        video_path = candidates[0]
        print(f"[AUTO] Auto-selected available video: {video_path}")
    else:
        print(f"[VIDEO] Selected video: {video_path}")

    run_test(
        video_path=video_path,
        model_path=args.model,
        yolo_path=args.yolo,
        headless=args.headless,
        save_video_path=args.save_video,
        max_frames=args.max_frames,
        confidence_thresh=args.conf,
        stability_window=args.stability,
        cooldown_sec=args.cooldown
    )


if __name__ == "__main__":
    main()
