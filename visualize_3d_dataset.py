"""
3D Dataset & Human Motion Digital Twin Visualizer
Interactive 3D simulation of dataset_advanced samples in Python (Matplotlib 3D).

Features:
  1. Full 3D Articulated Human Skeleton (Torso, Arms, Head, Hands with 5 fingers each).
  2. 3D Workspace: Desk Surface, Main Box (open container), Red Box, and Blue Box.
  3. Real-Time Physical Containment (Mathematically computes if Red/Blue is IN or OUT of Main Box).
  4. Live Model Inference: Evaluates best_tar_model1.pth on the exact sample and displays predictions.
  5. Interactive GUI Controls:
     - Mouse: Click and drag to orbit/rotate in 3D from any perspective.
     - Play / Pause button to run smooth 30 FPS animation.
     - Frame Slider (0 to 47) to scrub forward/backward.
     - Prev / Next buttons to cycle through dataset recordings.
     - Keyboard: Space = Play/Pause, Left/Right = Frame step, N = Next sample, P = Prev sample.

Usage:
    python visualize_3d_dataset.py
    python visualize_3d_dataset.py --file dataset_advanced/label_2_0.npy
    python visualize_3d_dataset.py --label pick_blue
"""

import os
import glob
import sys
import shutil
import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model_def import TARModel, NUM_CLASSES, FEATURE_DIM, SEQ_LEN
import feature_utils as fu

# Dark Theme Styling
plt.style.use('dark_background')

# ================= POSE & HAND SKELETON TOPOLOGY =================
POSE_BONES = [
    (11, 12),           # Shoulder to Shoulder
    (11, 13), (13, 15), # Left Arm: Shoulder -> Elbow -> Wrist
    (12, 14), (14, 16), # Right Arm: Shoulder -> Elbow -> Wrist
    (11, 23), (12, 24), # Torso sides: Shoulder -> Hip
    (23, 24),           # Hip to Hip
    (0, 11), (0, 12),   # Neck / Head to Shoulders
    (0, 1), (1, 2),     # Left Eye
    (0, 4), (4, 5),     # Right Eye
]

HAND_FINGER_CHAINS = [
    [0, 1, 2, 3, 4],       # Thumb
    [0, 5, 6, 7, 8],       # Index
    [0, 9, 10, 11, 12],    # Middle
    [0, 13, 14, 15, 16],   # Ring
    [0, 17, 18, 19, 20],   # Pinky
    [5, 9, 13, 17],        # Palm Knuckles
]


def load_model(path="best_tar_model1.pth"):
    if not os.path.exists(path):
        path = "best_tar_model.pth"
    if not os.path.exists(path):
        return None
    model = TARModel(input_size=FEATURE_DIM, num_classes=NUM_CLASSES)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_window(window):
    arr = np.array(window, dtype=np.float32)
    mean = np.mean(arr)
    std = np.std(arr) + 1e-6
    arr = (arr - mean) / std
    return np.clip(arr, -5.0, 5.0)


def predict_sample(model, data):
    if model is None:
        return "N/A", 0.0, np.zeros(NUM_CLASSES)
    normed = normalize_window(data)
    with torch.no_grad():
        x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return fu.LABELS[pred_idx], float(probs[pred_idx]), probs


def make_box_faces(cx, cy, cz, sx, sy, sz):
    """Creates quad faces for a 3D box."""
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    z0, z1 = cz - sz / 2.0, cz + sz / 2.0
    faces = [
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], # Bottom
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], # Top
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]], # Front
        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], # Back
        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]], # Left
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], # Right
    ]
    return faces


class Visualizer3D:
    def __init__(self, file_list, initial_idx=0):
        self.file_list = file_list
        self.current_idx = initial_idx % len(file_list)
        self.tar_model = load_model()
        
        self.data = None
        self.frame_idx = 0
        self.is_playing = False
        self.timer = None

        self.setup_gui()
        self.load_sample(self.current_idx)

    def setup_gui(self):
        self.fig = plt.figure(figsize=(13, 8), facecolor='#121212')
        self.fig.canvas.manager.set_window_title("3D Motion & Task Digital Twin - Dataset Visualizer")

        # 3D Main Scene Axis
        self.ax = self.fig.add_axes([0.02, 0.15, 0.72, 0.80], projection='3d')
        self.ax.set_facecolor('#141414')

        # Telemetry & Stats Axis (Right Panel)
        self.ax_info = self.fig.add_axes([0.76, 0.15, 0.22, 0.80])
        self.ax_info.set_facecolor('#1a1a1a')
        self.ax_info.set_xticks([])
        self.ax_info.set_yticks([])

        # Bottom Controls
        self.ax_slider = self.fig.add_axes([0.15, 0.05, 0.50, 0.03], facecolor='#222222')
        self.slider = Slider(self.ax_slider, 'Frame', 0, SEQ_LEN - 1, valinit=0, valstep=1, color='#00d2ff')
        self.slider.on_changed(self.on_slider_changed)

        self.ax_play = self.fig.add_axes([0.68, 0.045, 0.06, 0.04])
        self.btn_play = Button(self.ax_play, 'Play', color='#282828', hovercolor='#3a3a3a')
        self.btn_play.on_clicked(self.toggle_play)

        self.ax_prev = self.fig.add_axes([0.03, 0.045, 0.05, 0.04])
        self.btn_prev = Button(self.ax_prev, 'Prev', color='#282828', hovercolor='#3a3a3a')
        self.btn_prev.on_clicked(self.prev_sample)

        self.ax_next = self.fig.add_axes([0.09, 0.045, 0.05, 0.04])
        self.btn_next = Button(self.ax_next, 'Next', color='#282828', hovercolor='#3a3a3a')
        self.btn_next.on_clicked(self.next_sample)

        # Dataset Cleaning Controls
        self.ax_quarantine = self.fig.add_axes([0.76, 0.045, 0.11, 0.04])
        self.btn_quarantine = Button(self.ax_quarantine, 'Discard [X]', color='#551122', hovercolor='#881133')
        self.btn_quarantine.on_clicked(self.quarantine_current_sample)

        self.ax_relabel = self.fig.add_axes([0.88, 0.045, 0.10, 0.04])
        self.btn_relabel = Button(self.ax_relabel, 'Relabel [R]', color='#183355', hovercolor='#284477')
        self.btn_relabel.on_clicked(self.relabel_current_sample)

        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def load_sample(self, idx):
        if not self.file_list:
            return
        self.current_idx = idx % len(self.file_list)
        filepath = self.file_list[self.current_idx]
        self.data = np.load(filepath)
        self.filename = os.path.basename(filepath)

        # Parse ground truth label from filename (e.g. label_2_0.npy)
        parts = self.filename.split('_')
        self.label_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        self.label_name = fu.LABELS[self.label_id] if self.label_id < len(fu.LABELS) else f"label_{self.label_id}"

        # Run model prediction on full 48-frame window
        self.pred_label, self.pred_conf, self.probs = predict_sample(self.tar_model, self.data)

        self.frame_idx = 0
        self.slider.set_val(0)
        self.update_plot()

    def on_slider_changed(self, val):
        self.frame_idx = int(val)
        self.update_plot()

    def toggle_play(self, event=None):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Pause' if self.is_playing else 'Play')
        if self.is_playing:
            self.animate()

    def animate(self):
        if not self.is_playing:
            return
        nxt = (self.frame_idx + 1) % SEQ_LEN
        self.slider.set_val(nxt)
        self.fig.canvas.draw_idle()
        plt.pause(0.033) # ~30 FPS
        if self.is_playing:
            self.fig.canvas.manager.window.after(33, self.animate) if hasattr(self.fig.canvas.manager, 'window') and hasattr(self.fig.canvas.manager.window, 'after') else self.animate()

    def prev_sample(self, event=None):
        if self.file_list:
            self.load_sample((self.current_idx - 1) % len(self.file_list))

    def next_sample(self, event=None):
        if self.file_list:
            self.load_sample((self.current_idx + 1) % len(self.file_list))

    def quarantine_current_sample(self, event=None):
        """Moves current sample to dataset_quarantine/ and advances to next."""
        if not self.file_list:
            return
        current_file = self.file_list[self.current_idx]
        quarantine_dir = os.path.join(PROJECT_ROOT, "dataset_quarantine")
        os.makedirs(quarantine_dir, exist_ok=True)
        dest = os.path.join(quarantine_dir, os.path.basename(current_file))
        try:
            shutil.move(current_file, dest)
            print(f"[QUARANTINED] Moved {os.path.basename(current_file)} -> {quarantine_dir}")
            del self.file_list[self.current_idx]
            if not self.file_list:
                print("[INFO] No more samples left in current view.")
                self.ax.clear()
                self.ax.text(0, 0, 0, "No Samples Remaining in Set", color='#ffffff')
                self.ax_info.clear()
                self.ax_info.text(0.1, 0.5, "View Finished", color='#00ffcc', fontsize=12)
                self.fig.canvas.draw_idle()
                return
            self.current_idx = self.current_idx % len(self.file_list)
            self.load_sample(self.current_idx)
        except Exception as e:
            print(f"[ERROR] Could not quarantine file: {e}")

    def relabel_current_sample(self, event=None):
        """Prompts to relabel the current sample file."""
        if not self.file_list:
            return
        current_file = self.file_list[self.current_idx]
        dirname = os.path.dirname(current_file)
        fname = os.path.basename(current_file)
        
        print("\n--- RELABEL SAMPLE ---")
        for i, lbl in enumerate(fu.LABELS):
            print(f"  {i}: {lbl}")
        print(f"Current: {self.label_name} ({self.label_id})")
        val = input("Enter new label ID (0-6) or press Enter to cancel: ").strip()
        if not val.isdigit() or int(val) < 0 or int(val) >= len(fu.LABELS):
            print("[CANCELLED] Relabel aborted.")
            return
        
        new_id = int(val)
        ts = int(time.time())
        new_fname = f"label_{new_id}_{ts}.npy"
        new_path = os.path.join(dirname, new_fname)
        try:
            os.rename(current_file, new_path)
            print(f"[RELABELED] Renamed {fname} -> {new_fname} ({fu.LABELS[new_id]})")
            self.file_list[self.current_idx] = new_path
            self.load_sample(self.current_idx)
        except Exception as e:
            print(f"[ERROR] Could not relabel file: {e}")

    def on_key(self, event):
        if event.key == ' ':
            self.toggle_play()
        elif event.key in ('right', 'd'):
            self.slider.set_val((self.frame_idx + 1) % SEQ_LEN)
        elif event.key in ('left', 'a'):
            self.slider.set_val((self.frame_idx - 1) % SEQ_LEN)
        elif event.key in ('n', 'N'):
            self.next_sample()
        elif event.key in ('p', 'P'):
            self.prev_sample()
        elif event.key in ('x', 'X', 'delete'):
            self.quarantine_current_sample()
        elif event.key in ('r', 'R'):
            self.relabel_current_sample()
        elif event.key in ('k', 'K'):
            print(f"[VERIFIED] Clean sample: {self.filename}")
            self.next_sample()

    def update_plot(self):
        t = self.frame_idx
        feat = self.data[t]

        self.ax.clear()
        self.ax_info.clear()

        # Camera Perspective & View Limits
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-0.4, 1.6)
        self.ax.set_zlim(-0.8, 1.4)
        self.ax.set_xlabel('X (Lateral)', color='#888888', labelpad=-4)
        self.ax.set_ylabel('Y (Depth / Forward)', color='#888888', labelpad=-4)
        self.ax.set_zlabel('Z (Height)', color='#888888', labelpad=-4)
        self.ax.tick_params(colors='#666666', labelsize=8)

        # 1. Desk Surface (3D Plane)
        desk_x = np.array([[-1.0, 1.0], [-1.0, 1.0]])
        desk_y = np.array([[0.3, 0.3], [1.5, 1.5]])
        desk_z = np.array([[-0.25, -0.25], [-0.25, -0.25]])
        self.ax.plot_surface(desk_x, desk_y, desk_z, color='#252830', alpha=0.5, edgecolor='#383c48', linewidth=0.5)

        # 2. Extract Human Pose Landmarks
        pose_x = feat[0:66:2]
        pose_y = feat[1:66:2]
        # Map normalized 2D image coordinates to 3D world:
        # Z3D = -pose_y (vertical, head up, hips at 0)
        # X3D = pose_x (left/right)
        # Y3D = estimated forward depth (hips at 0.0, reaching arms push into Y > 0)
        lms_3d = np.zeros((33, 3), dtype=np.float32)
        lms_3d[:, 0] = pose_x
        lms_3d[:, 2] = -pose_y  # Head is high positive Z, hips at ~0

        # Estimate arm forward reach from wrist-to-torso displacement
        lms_3d[13, 1] = 0.25  # Left elbow slightly forward
        lms_3d[14, 1] = 0.25  # Right elbow slightly forward
        lms_3d[15, 1] = 0.55 + max(0.0, -pose_y[15] * 0.4) # Left wrist reaches forward towards desk
        lms_3d[16, 1] = 0.55 + max(0.0, -pose_y[16] * 0.4) # Right wrist reaches forward towards desk

        # Draw Pose Joints & Bones
        self.ax.scatter(lms_3d[:, 0], lms_3d[:, 1], lms_3d[:, 2], color='#00ffcc', s=18, alpha=0.85)
        for i1, i2 in POSE_BONES:
            if i1 < len(lms_3d) and i2 < len(lms_3d):
                p1, p2 = lms_3d[i1], lms_3d[i2]
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#00d2ff', linewidth=2.2)

        # Draw Head Sphere
        head = lms_3d[0]
        self.ax.scatter([head[0]], [head[1]], [head[2] + 0.08], color='#ffffff', s=120, edgecolors='#00ffff', linewidth=1.5)

        # 3. Draw Left & Right Hands (Finger Wireframes)
        lh_feat = feat[66:108]
        rh_feat = feat[108:150]
        for hand_raw, wrist_pt, h_color in [(lh_feat, lms_3d[15], '#ff9900'), (rh_feat, lms_3d[16], '#ffcc00')]:
            if np.any(hand_raw != 0):
                hx = hand_raw[0:42:2] * 0.15 + wrist_pt[0]
                hz = -hand_raw[1:42:2] * 0.15 + wrist_pt[2]
                hy = wrist_pt[1] + np.linspace(0.0, 0.08, 21)
                self.ax.scatter(hx, hy, hz, color=h_color, s=8)
                for chain in HAND_FINGER_CHAINS:
                    pts = np.column_stack([hx[chain], hy[chain], hz[chain]])
                    self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=h_color, linewidth=1.0)

        # 4. Extract & Render 3D Objects on Desk
        # Main Box: [156:159] -> (cx, cy, area)
        mcx, mcy, marea = feat[156:159]
        main_detected = (marea > 0.001)
        mx3d = (mcx - 0.5) * 1.6
        my3d = 0.90
        mz3d = -0.10
        msize = [0.55, 0.45, 0.30]

        if main_detected:
            main_faces = make_box_faces(mx3d, my3d, mz3d, msize[0], msize[1], msize[2])
            # Draw semi-transparent open container box
            poly_main = Poly3DCollection(main_faces, alpha=0.20, facecolor='#20a020', edgecolor='#00ff44', linewidth=1.2)
            self.ax.add_collection3d(poly_main)
            self.ax.text(mx3d, my3d, mz3d + msize[2]/2 + 0.08, "MAIN CONTAINER", color='#00ff44', fontsize=8, weight='bold', ha='center')

        # Red Box: [150:153] -> (cx, cy, area)
        rcx, rcy, rarea = feat[150:153]
        red_detected = (rarea > 0.0005)
        rx3d = (rcx - 0.5) * 1.6 if red_detected else -99.0
        ry3d = 0.90
        rz3d = -0.15
        rsize = [0.14, 0.14, 0.14]

        if red_detected:
            red_faces = make_box_faces(rx3d, ry3d, rz3d, rsize[0], rsize[1], rsize[2])
            poly_red = Poly3DCollection(red_faces, alpha=0.85, facecolor='#ff2244', edgecolor='#ffffff', linewidth=1.0)
            self.ax.add_collection3d(poly_red)
            self.ax.text(rx3d, ry3d, rz3d + 0.12, "RED", color='#ff4444', fontsize=9, weight='bold', ha='center')

        # Blue Box: [153:156] -> (cx, cy, area)
        bcx, bcy, barea = feat[153:156]
        blue_detected = (barea > 0.0005)
        bx3d = (bcx - 0.5) * 1.6 if blue_detected else -99.0
        by3d = 0.90
        bz3d = -0.15
        bsize = [0.14, 0.14, 0.14]

        if blue_detected:
            blue_faces = make_box_faces(bx3d, by3d, bz3d, bsize[0], bsize[1], bsize[2])
            poly_blue = Poly3DCollection(blue_faces, alpha=0.85, facecolor='#2266ff', edgecolor='#ffffff', linewidth=1.0)
            self.ax.add_collection3d(poly_blue)
            self.ax.text(bx3d, by3d, bz3d + 0.12, "BLUE", color='#4488ff', fontsize=9, weight='bold', ha='center')

        # 5. Compute Mathematical 3D Containment (Inside vs Outside)
        def is_inside(ox, oy, oz, mx, my, mz, ms):
            if not main_detected or ox < -50:
                return False, "Not Detected"
            dx = abs(ox - mx) <= (ms[0] / 2.0 + 0.05)
            dy = abs(oy - my) <= (ms[1] / 2.0 + 0.05)
            dz = abs(oz - mz) <= (ms[2] / 2.0 + 0.05)
            if dx and dy and dz:
                return True, "INSIDE [v]"
            return False, "OUTSIDE [x]"

        red_in, red_status = is_inside(rx3d, ry3d, rz3d, mx3d, my3d, mz3d, msize)
        blue_in, blue_status = is_inside(bx3d, by3d, bz3d, mx3d, my3d, mz3d, msize)

        # 6. Render Right-Hand Telemetry & Model Inspection Panel
        y_pos = 0.95
        self.ax_info.text(0.05, y_pos, "3D DIGITAL TWIN", color='#00e5ff', fontsize=11, weight='bold')
        y_pos -= 0.07

        self.ax_info.text(0.05, y_pos, f"File: {self.filename}", color='#aaaaaa', fontsize=8)
        y_pos -= 0.05
        self.ax_info.text(0.05, y_pos, f"Ground Truth: {self.label_name}", color='#ffffff', fontsize=9, weight='bold')
        y_pos -= 0.07

        # Model Prediction Box
        is_match = (self.pred_label == self.label_name)
        pred_color = '#00ff66' if is_match else '#ff3344'
        match_icon = "[MATCH]" if is_match else "[MISMATCH]"
        self.ax_info.text(0.05, y_pos, f"AI Model: {self.pred_label}", color=pred_color, fontsize=10, weight='bold')
        y_pos -= 0.04
        self.ax_info.text(0.05, y_pos, f"Confidence: {self.pred_conf*100:.1f}%  {match_icon}", color=pred_color, fontsize=8)
        y_pos -= 0.08

        # 3D Physical Containment Engine Readout
        self.ax_info.text(0.05, y_pos, "--- 3D PHYSICAL STATE ---", color='#888888', fontsize=8)
        y_pos -= 0.05

        r_color = '#ff6666' if red_in else '#aaaaaa'
        self.ax_info.text(0.05, y_pos, f"Red Box: {red_status}", color=r_color, fontsize=9, weight='bold')
        y_pos -= 0.05

        b_color = '#66aaff' if blue_in else '#aaaaaa'
        self.ax_info.text(0.05, y_pos, f"Blue Box: {blue_status}", color=b_color, fontsize=9, weight='bold')
        y_pos -= 0.08

        # Relational Hand Distances (from feature vector)
        wl_r, wl_b, wl_m, wr_r, wr_b, wr_m = feat[159:165]
        self.ax_info.text(0.05, y_pos, "--- HAND DISTANCES ---", color='#888888', fontsize=8)
        y_pos -= 0.05
        self.ax_info.text(0.05, y_pos, f"Right Wrist -> Red:  {wr_r:.2f}", color='#ff8888', fontsize=8)
        y_pos -= 0.04
        self.ax_info.text(0.05, y_pos, f"Right Wrist -> Blue: {wr_b:.2f}", color='#88bbff', fontsize=8)
        y_pos -= 0.04
        self.ax_info.text(0.05, y_pos, f"Right Wrist -> Main: {wr_m:.2f}", color='#88ff88', fontsize=8)
        y_pos -= 0.08

        # Probability Distribution Bar Chart
        self.ax_info.text(0.05, y_pos, "--- CLASS PROBABILITIES ---", color='#888888', fontsize=8)
        y_pos -= 0.05
        for i, name in enumerate(fu.LABELS):
            p = self.probs[i] if len(self.probs) > i else 0.0
            p_bar = "|" * int(p * 15)
            c = '#00ff88' if name == self.label_name else '#777777'
            self.ax_info.text(0.05, y_pos, f"{name[:12]:<12} {p*100:4.1f}% {p_bar}", color=c, fontsize=7, fontfamily='monospace')
            y_pos -= 0.035

        # View Title
        self.ax.set_title(
            f"[{self.current_idx+1}/{len(self.file_list)}] {self.label_name}  |  Frame: {t}/{SEQ_LEN-1}  |  Pred: {self.pred_label} ({self.pred_conf*100:.0f}%)",
            color='#ffffff', fontsize=11, weight='bold'
        )

        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="3D Dataset & Human Motion Digital Twin Visualizer")
    parser.add_argument("--dir", default="dataset_advanced", help="Directory containing .npy dataset files")
    parser.add_argument("--quarantined", action="store_true", help="View samples in dataset_quarantine directory")
    parser.add_argument("--file", default=None, help="Specific .npy file to visualize")
    parser.add_argument("--label", default=None, help="Filter by label name (e.g. pick_red, pick_blue)")
    args = parser.parse_args()

    if args.quarantined:
        args.dir = "dataset_quarantine"

    files = []
    if args.file and os.path.exists(args.file):
        files = [args.file]
    else:
        pattern = os.path.join(args.dir, "label_*.npy")
        files = glob.glob(pattern)
        if args.label:
            target_idx = fu.LABELS.index(args.label) if args.label in fu.LABELS else -1
            if target_idx >= 0:
                files = [f for f in files if f"label_{target_idx}_" in os.path.basename(f)]

    if not files:
        print(f"[ERROR] No matching .npy files found in '{args.dir}'.")
        return

    # Sort files naturally
    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    files.sort(key=natural_sort_key)

    print(f"[3D VISUALIZER] Found {len(files)} dataset files in '{args.dir}'.")
    print("[CONTROLS]")
    print("  Mouse Click & Drag : Rotate 3D camera in any direction")
    print("  [Space]            : Play / Pause 30 FPS animation")
    print("  [Left] / [Right]   : Step frame backward / forward")
    print("  [N] / [P]          : Next / Previous recording")
    print("  [X] / [Delete]     : Quarantine (discard) current sample")
    print("  [R]                : Relabel current sample")
    print("  [K]                : Mark sample as verified clean")
    print("  Close Window       : Exit visualizer\n")

    vis = Visualizer3D(files)
    plt.show()


if __name__ == "__main__":
    main()
