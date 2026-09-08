"""
AI-Powered Autonomous Dataset Cleaner & Health Auditor
Inspects 3D motion/interaction recordings in dataset_advanced across 4 intelligent layers:
  1. Sensor & Tracker Health (MediaPipe pose/hands dropouts, missing YOLO/color boxes, tracking jitter spikes).
  2. Physical Feasibility (Wrist-to-object reach proximity, object movement validation).
  3. AI Confidence & Mislabeled Detection (Data-Centric AI via TARModel cross-checking).
  4. Safe Reversible Quarantine Engine (Never deletes permanently; logs full undo manifest).

Usage:
    # 1. Preview audit without moving any files (Dry-Run):
    python auto_clean_dataset.py --dry-run

    # 2. Automatically quarantine dirty/corrupt samples into dataset_quarantine/:
    python auto_clean_dataset.py --quarantine

    # 3. Undo / Restore quarantined files back into dataset_advanced/:
    python auto_clean_dataset.py --restore

    # 4. Strictness levels: lenient | balanced (default) | strict
    python auto_clean_dataset.py --quarantine --strictness strict
"""

import os
import sys
import glob
import re
import json
import shutil
import time
import csv
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from model_def import TARModel, NUM_CLASSES, FEATURE_DIM, SEQ_LEN
import feature_utils as fu

# Default Directories
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_advanced")
DEFAULT_QUARANTINE_DIR = os.path.join(PROJECT_ROOT, "dataset_quarantine")
DEFAULT_MANIFEST_PATH = os.path.join(DEFAULT_QUARANTINE_DIR, "quarantine_manifest.json")
REPORT_MD_PATH = os.path.join(PROJECT_ROOT, "dataset_clean_report.md")
REPORT_JSON_PATH = os.path.join(PROJECT_ROOT, "dataset_clean_report.json")

# Model checkpoints
MODEL_CHECKPOINTS = ["best_tar_model1.pth", "best_tar_model.pth"]


def load_ai_model():
    """Loads the best available trained model for AI-assisted mislabel detection."""
    model_path = None
    for p in MODEL_CHECKPOINTS:
        full_p = os.path.join(PROJECT_ROOT, p)
        if os.path.exists(full_p):
            model_path = full_p
            break
    if not model_path:
        return None, None

    model = TARModel(input_size=FEATURE_DIM, num_classes=NUM_CLASSES)
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, os.path.basename(model_path)
    except Exception as e:
        print(f"[WARN] Could not load AI model ({e}). Proceeding with heuristic/physics layers only.")
        return None, None


def normalize_window(data):
    """Per-sample Z-score normalization matching training pipeline."""
    arr = np.array(data, dtype=np.float32)
    mean = np.mean(arr)
    std = np.std(arr) + 1e-6
    arr = (arr - mean) / std
    return np.clip(arr, -5.0, 5.0)


class DatasetAuditor:
    def __init__(self, dataset_dir, strictness="balanced"):
        self.dataset_dir = dataset_dir
        self.strictness = strictness
        self.ai_model, self.model_name = load_ai_model()
        
        # Configure calibrated thresholds based on strictness
        if self.strictness == "lenient":
            self.min_hand_ratio = 0.02        # Flags only strictly zero-hand files (<= 1 frame)
            self.min_obj_area = 0.0001
            self.max_reach_dist = 0.50
            self.ai_conf_threshold = 0.95     # Only flag extreme model mismatches
            self.ai_gt_max_prob = 0.05
            self.max_jitter_velocity = 2.8
        elif self.strictness == "strict":
            self.min_hand_ratio = 0.15        # Hands must be detected in >= 15% frames
            self.min_obj_area = 0.0008
            self.max_reach_dist = 0.30
            self.ai_conf_threshold = 0.75
            self.ai_gt_max_prob = 0.25
            self.max_jitter_velocity = 1.8
        else:  # balanced (default)
            self.min_hand_ratio = 0.05        # Hands must appear for >= 3 frames in 48
            self.min_obj_area = 0.0005
            self.max_reach_dist = 0.40
            self.ai_conf_threshold = 0.85
            self.ai_gt_max_prob = 0.15
            self.max_jitter_velocity = 2.2

    def parse_filename(self, filepath):
        fname = os.path.basename(filepath)
        match = re.match(r"^label_(\d+)_\d+\.npy$", fname)
        if match:
            lbl_id = int(match.group(1))
            lbl_name = fu.LABELS[lbl_id] if lbl_id < len(fu.LABELS) else f"label_{lbl_id}"
            return lbl_id, lbl_name
        return None, None

    def audit_sample(self, filepath):
        """
        Audits a single sample file.
        Returns:
            dict with audit results and metrics
        """
        fname = os.path.basename(filepath)
        lbl_id, lbl_name = self.parse_filename(filepath)
        
        result = {
            "file": fname,
            "path": filepath,
            "label_id": lbl_id,
            "label_name": lbl_name,
            "status": "CLEAN",
            "severity": "NONE",
            "reasons": [],
            "model_pred": "N/A",
            "model_conf": 0.0,
            "gt_prob": 0.0,
            "metrics": {}
        }

        # -------------------------------------------------------------
        # LAYER 1: Structural & Tracker Health
        # -------------------------------------------------------------
        try:
            data = np.load(filepath)
        except Exception as e:
            result["status"] = "FLAGGED"
            result["severity"] = "HIGH"
            result["reasons"].append(f"CORRUPT_FILE: Unable to load numpy array ({e})")
            return result

        if data.ndim != 2:
            result["status"] = "FLAGGED"
            result["severity"] = "HIGH"
            result["reasons"].append(f"INVALID_DIMENSION: ndim={data.ndim}, expected 2")
            return result

        T, D = data.shape
        if D != FEATURE_DIM:
            result["status"] = "FLAGGED"
            result["severity"] = "HIGH"
            result["reasons"].append(f"DIMENSION_MISMATCH: {D} dims, expected {FEATURE_DIM}")
            return result

        if np.isnan(data).any() or np.isinf(data).any():
            result["status"] = "FLAGGED"
            result["severity"] = "HIGH"
            result["reasons"].append("CORRUPT_VALUES: Contains NaN or Inf numbers")
            return result

        # Check Frozen/Static Clip
        feat_std = float(np.std(data))
        result["metrics"]["feat_std"] = feat_std
        if feat_std < 1e-4:
            result["status"] = "FLAGGED"
            result["severity"] = "HIGH"
            result["reasons"].append("STATIC_RECORDING: Features show zero variance (frozen frame)")
            return result

        # Check Hand Tracking Presence (Features 66:108 LH, 108:150 RH)
        lh = data[:, 66:108]
        rh = data[:, 108:150]
        either_hand_frames = np.sum(np.any(lh != 0, axis=1) | np.any(rh != 0, axis=1))
        hand_ratio = either_hand_frames / float(T)
        result["metrics"]["hand_ratio"] = hand_ratio

        # For any active action (non-idle), hand tracking is crucial
        if lbl_id is not None and lbl_id != 0:
            if hand_ratio < self.min_hand_ratio:
                result["status"] = "FLAGGED"
                result["severity"] = "HIGH"
                result["reasons"].append(
                    f"MISSING_HAND_TRACKING: Hands detected in only {hand_ratio*100:.0f}% of frames (min {self.min_hand_ratio*100:.0f}%)"
                )

        # Check Target Object Presence
        # Red: [150:153], Blue: [153:156], Main: [156:159]
        red_max_area = float(np.max(data[:, 152]))
        blue_max_area = float(np.max(data[:, 155]))
        main_max_area = float(np.max(data[:, 158]))
        result["metrics"]["red_max_area"] = red_max_area
        result["metrics"]["blue_max_area"] = blue_max_area
        result["metrics"]["main_max_area"] = main_max_area

        if lbl_name in ("pick_red", "place_red_out") and red_max_area < self.min_obj_area:
            result["status"] = "FLAGGED"
            result["severity"] = "HIGH"
            result["reasons"].append(f"MISSING_TARGET_OBJECT: Red box never detected in '{lbl_name}'")

        if lbl_name in ("pick_blue", "place_blue_in") and blue_max_area < self.min_obj_area:
            result["status"] = "FLAGGED"
            result["severity"] = "HIGH"
            result["reasons"].append(f"MISSING_TARGET_OBJECT: Blue box never detected in '{lbl_name}'")

        if lbl_name in ("open_box", "close_box") and main_max_area < self.min_obj_area:
            result["status"] = "FLAGGED"
            result["severity"] = "MEDIUM"
            result["reasons"].append(f"MISSING_CONTAINER: Main container box never detected in '{lbl_name}'")

        # Check Landmark Velocity Spikes (Teleportation Jitter)
        vel_block = np.abs(data[1:, 166:232]) # Pose velocity channels
        max_vel = float(np.max(vel_block)) if vel_block.size > 0 else 0.0
        result["metrics"]["max_pose_velocity"] = max_vel
        if max_vel > self.max_jitter_velocity:
            result["status"] = "FLAGGED"
            sev = "HIGH" if max_vel > 0.8 else "MEDIUM"
            if result["severity"] != "HIGH":
                result["severity"] = sev
            result["reasons"].append(
                f"TRACKER_JITTER_SPIKE: Severe landmark teleportation detected (max delta {max_vel:.2f})"
            )

        # -------------------------------------------------------------
        # LAYER 2: 3D Physical Feasibility & Reach Proximity
        # -------------------------------------------------------------
        # Wrist-to-object distances:
        # [159]=LW->Red, [160]=LW->Blue, [161]=LW->Main
        # [162]=RW->Red, [163]=RW->Blue, [164]=RW->Main
        if lbl_name in ("pick_red", "place_red_out") and red_max_area >= self.min_obj_area:
            min_dist_red = float(min(np.min(data[:, 159]), np.min(data[:, 162])))
            result["metrics"]["min_dist_red"] = min_dist_red
            if min_dist_red > self.max_reach_dist:
                result["status"] = "FLAGGED"
                if result["severity"] != "HIGH":
                    result["severity"] = "MEDIUM"
                result["reasons"].append(
                    f"PHYSICAL_PROXIMITY_FAIL: Hand never came close to Red box (closest {min_dist_red:.2f} > {self.max_reach_dist:.2f})"
                )

        if lbl_name in ("pick_blue", "place_blue_in") and blue_max_area >= self.min_obj_area:
            min_dist_blue = float(min(np.min(data[:, 160]), np.min(data[:, 163])))
            result["metrics"]["min_dist_blue"] = min_dist_blue
            if min_dist_blue > self.max_reach_dist:
                result["status"] = "FLAGGED"
                if result["severity"] != "HIGH":
                    result["severity"] = "MEDIUM"
                result["reasons"].append(
                    f"PHYSICAL_PROXIMITY_FAIL: Hand never came close to Blue box (closest {min_dist_blue:.2f} > {self.max_reach_dist:.2f})"
                )

        # -------------------------------------------------------------
        # LAYER 3: AI Model Confidence & Outlier Mislabeled Audit
        # -------------------------------------------------------------
        if self.ai_model is not None and lbl_id is not None:
            normed = normalize_window(data)
            with torch.no_grad():
                x = torch.tensor(normed, dtype=torch.float32).unsqueeze(0)
                logits = self.ai_model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            pred_idx = int(np.argmax(probs))
            pred_conf = float(probs[pred_idx])
            pred_name = fu.LABELS[pred_idx] if pred_idx < len(fu.LABELS) else f"label_{pred_idx}"
            gt_prob = float(probs[lbl_id]) if lbl_id < len(probs) else 0.0

            result["model_pred"] = pred_name
            result["model_conf"] = pred_conf
            result["gt_prob"] = gt_prob

            # High-confidence discrepancy
            if pred_idx != lbl_id and pred_conf >= self.ai_conf_threshold and gt_prob <= self.ai_gt_max_prob:
                result["status"] = "FLAGGED"
                sev = "HIGH" if pred_conf > 0.88 else "MEDIUM"
                if result["severity"] != "HIGH":
                    result["severity"] = sev
                result["reasons"].append(
                    f"AI_MISLABEL_SUSPECT: Model predicts '{pred_name}' with {pred_conf*100:.1f}% conf, while GT '{lbl_name}' probability is only {gt_prob*100:.1f}%"
                )

        return result

    def scan_dataset(self):
        """Scans and audits all .npy files in the dataset folder."""
        pattern = os.path.join(self.dataset_dir, "label_*.npy")
        files = glob.glob(pattern)
        
        # Natural sort
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        files.sort(key=natural_sort_key)

        print(f"[AUDIT] Scanning {len(files)} files in '{self.dataset_dir}' (Strictness: {self.strictness})...")
        results = []
        for i, f in enumerate(files):
            r = self.audit_sample(f)
            results.append(r)
            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  Processed {i+1}/{len(files)} samples...")

        return results


def quarantine_files(flagged_results, quarantine_dir=DEFAULT_QUARANTINE_DIR, manifest_path=DEFAULT_MANIFEST_PATH):
    """Safely moves flagged files into quarantine_dir and records an undo manifest."""
    os.makedirs(quarantine_dir, exist_ok=True)
    manifest = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as fp:
                manifest = json.load(fp)
        except Exception:
            manifest = []

    moved_count = 0
    for r in flagged_results:
        src = r["path"]
        if not os.path.exists(src):
            continue
        dest = os.path.join(quarantine_dir, r["file"])
        
        # If destination file already exists, create a unique backup name
        if os.path.exists(dest):
            base, ext = os.path.splitext(r["file"])
            dest = os.path.join(quarantine_dir, f"{base}_{int(time.time())}{ext}")

        shutil.move(src, dest)
        manifest.append({
            "original_path": src,
            "quarantined_path": dest,
            "filename": r["file"],
            "label_name": r["label_name"],
            "reasons": r["reasons"],
            "severity": r["severity"],
            "model_pred": r["model_pred"],
            "model_conf": r["model_conf"],
            "timestamp": time.time()
        })
        moved_count += 1

    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    return moved_count


def restore_quarantine(manifest_path=DEFAULT_MANIFEST_PATH):
    """Restores all quarantined files back to their original locations."""
    if not os.path.exists(manifest_path):
        print(f"[RESTORE] No quarantine manifest found at '{manifest_path}'. Nothing to restore.")
        return 0

    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    restored_count = 0
    remaining_manifest = []
    for item in manifest:
        q_path = item["quarantined_path"]
        orig_path = item["original_path"]
        if os.path.exists(q_path):
            os.makedirs(os.path.dirname(orig_path), exist_ok=True)
            shutil.move(q_path, orig_path)
            restored_count += 1
        else:
            remaining_manifest.append(item)

    if remaining_manifest:
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump(remaining_manifest, fp, indent=2)
    else:
        try:
            os.remove(manifest_path)
        except OSError:
            pass

    return restored_count


def sync_metadata_csv(dataset_dir=DEFAULT_DATASET_DIR):
    """Updates or reconstructs metadata.csv to accurately reflect currently active dataset files."""
    meta_path = os.path.join(dataset_dir, "metadata.csv")
    active_files = set(f for f in os.listdir(dataset_dir) if f.endswith(".npy") and f.startswith("label_"))
    
    existing_rows = []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            for row in reader:
                if row and row[0] in active_files:
                    existing_rows.append(row)

    existing_filenames = set(r[0] for r in existing_rows)
    
    for fname in sorted(active_files):
        if fname not in existing_filenames:
            match = re.match(r"^label_(\d+)_\d+\.npy$", fname)
            lbl_id = int(match.group(1)) if match else 0
            lbl_name = fu.LABELS[lbl_id] if lbl_id < len(fu.LABELS) else f"label_{lbl_id}"
            existing_rows.append([fname, str(lbl_id), lbl_name, "1.0", "1.0", "1.0", "48", str(time.time())])

    with open(meta_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerows(existing_rows)

    return len(existing_rows)


def generate_reports(results, dataset_dir, quarantine_dir, model_name):
    """Generates detailed JSON and GitHub Markdown reports."""
    clean_items = [r for r in results if r["status"] == "CLEAN"]
    flagged_items = [r for r in results if r["status"] == "FLAGGED"]

    high_sev = [r for r in flagged_items if r["severity"] == "HIGH"]
    med_sev = [r for r in flagged_items if r["severity"] == "MEDIUM"]

    # Class breakdown
    class_stats = {}
    for r in results:
        lbl = r["label_name"]
        if lbl not in class_stats:
            class_stats[lbl] = {"total": 0, "clean": 0, "flagged": 0}
        class_stats[lbl]["total"] += 1
        if r["status"] == "CLEAN":
            class_stats[lbl]["clean"] += 1
        else:
            class_stats[lbl]["flagged"] += 1

    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(results),
        "clean_samples": len(clean_items),
        "flagged_samples": len(flagged_items),
        "high_severity": len(high_sev),
        "medium_severity": len(med_sev),
        "ai_model_used": model_name or "None",
        "dataset_dir": dataset_dir,
        "class_breakdown": class_stats,
        "flagged_details": flagged_items
    }

    # Write JSON report
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as fp:
        json.dump(summary_data, fp, indent=2)

    # Write Markdown Report
    clean_pct = (len(clean_items) / len(results) * 100) if results else 0
    flagged_pct = (len(flagged_items) / len(results) * 100) if results else 0

    lines = [
        "# Dataset Health & Auto-Clean Audit Report",
        f"\n**Audit Date:** {summary_data['timestamp']}  ",
        f"**Dataset Location:** `{dataset_dir}`  ",
        f"**AI Auditor Model:** `{model_name or 'Heuristics Only'}`\n",
        "## 1. Executive Summary",
        f"- **Total Samples Audited:** {len(results)}",
        f"- **Clean & Verified Samples:** {len(clean_items)} ({clean_pct:.1f}%)",
        f"- **Flagged / Dirty Samples:** {len(flagged_items)} ({flagged_pct:.1f}%)",
        f"  - **High Severity (Critical Flaws):** {len(high_sev)}",
        f"  - **Medium Severity (Physics/AI Ambiguity):** {len(med_sev)}\n",
        "## 2. Per-Class Health Breakdown\n",
        "| Class Label | Total | Clean Samples | Flagged Samples | Clean Rate |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]

    for lbl, stats in class_stats.items():
        c_rate = (stats["clean"] / stats["total"] * 100) if stats["total"] else 0
        lines.append(f"| `{lbl}` | {stats['total']} | {stats['clean']} | {stats['flagged']} | {c_rate:.1f}% |")

    lines.append("\n## 3. Flagged Samples Detailed Ledger\n")
    if not flagged_items:
        lines.append("No samples flagged! The dataset is 100% clean.")
    else:
        lines.append("| File | Label | Severity | Primary Reason | AI Prediction |")
        lines.append("| :--- | :--- | :---: | :--- | :--- |")
        for r in flagged_items:
            reasons_str = "<br>".join(r["reasons"])
            pred_str = f"{r['model_pred']} ({r['model_conf']*100:.0f}%)" if r["model_pred"] != "N/A" else "N/A"
            sev_badge = "HIGH" if r["severity"] == "HIGH" else "MEDIUM"
            lines.append(f"| `{r['file']}` | `{r['label_name']}` | {sev_badge} | {reasons_str} | {pred_str} |")

    lines.append("\n## 4. Next Steps & Recommended Actions")
    lines.append("1. **To isolate dirty samples immediately:**")
    lines.append("   ```powershell")
    lines.append("   python auto_clean_dataset.py --quarantine")
    lines.append("   ```")
    lines.append("2. **To inspect quarantined samples in 3D:**")
    lines.append("   ```powershell")
    lines.append("   python visualize_3d_dataset.py --quarantined")
    lines.append("   ```")
    lines.append("3. **To retrain the model on the clean dataset:**")
    lines.append("   ```powershell")
    lines.append("   python train_tar.py")
    lines.append("   ```")

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

    return REPORT_MD_PATH, REPORT_JSON_PATH


def main():
    parser = argparse.ArgumentParser(description="AI Automated Dataset Cleaner & Health Auditor")
    parser.add_argument("--dir", default=DEFAULT_DATASET_DIR, help="Path to dataset directory")
    parser.add_argument("--dry-run", action="store_true", help="Audit without moving any files (default behavior)")
    parser.add_argument("--quarantine", action="store_true", help="Safely move flagged files to dataset_quarantine/")
    parser.add_argument("--restore", action="store_true", help="Restore all quarantined files back to dataset directory")
    parser.add_argument("--strictness", choices=["lenient", "balanced", "strict"], default="balanced", help="Audit strictness")
    parser.add_argument("--sync-meta", action="store_true", help="Synchronize metadata.csv with existing files")
    args = parser.parse_args()

    # Handle restore operation
    if args.restore:
        print("[RESTORE] Restoring quarantined files back into dataset...")
        restored = restore_quarantine()
        sync_metadata_csv(args.dir)
        print(f"[RESTORE] Successfully restored {restored} files to '{args.dir}'.")
        return

    # Handle metadata sync only
    if args.sync_meta and not args.quarantine and not args.dry_run:
        count = sync_metadata_csv(args.dir)
        print(f"[METADATA] Synchronized metadata.csv with {count} active samples.")
        return

    auditor = DatasetAuditor(dataset_dir=args.dir, strictness=args.strictness)
    results = auditor.scan_dataset()

    flagged = [r for r in results if r["status"] == "FLAGGED"]
    clean = [r for r in results if r["status"] == "CLEAN"]

    md_path, json_path = generate_reports(results, args.dir, DEFAULT_QUARANTINE_DIR, auditor.model_name)

    print("\n" + "=" * 60)
    print("AI DATASET AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total Samples : {len(results)}")
    print(f"Clean Samples : {len(clean)} ({len(clean)/len(results)*100:.1f}%)")
    print(f"Flagged Dirty : {len(flagged)} ({len(flagged)/len(results)*100:.1f}%)")
    print(f"Audit Reports : {md_path}")
    print("=" * 60)

    if args.quarantine:
        print(f"\n[QUARANTINE] Moving {len(flagged)} flagged files to '{DEFAULT_QUARANTINE_DIR}'...")
        moved = quarantine_files(flagged)
        sync_metadata_csv(args.dir)
        print(f"[QUARANTINE] Successfully isolated {moved} dirty samples.")
        print(f"[DATASET] Clean active samples remaining in '{args.dir}': {len(clean)}")
        print("\nYou can now run retraining on clean data:")
        print("  python train_tar.py")
        print("  or")
        print("  python retrain_tar.py --dataset_path dataset_advanced")
    else:
        print("\n[NOTE] Ran in PREVIEW/DRY-RUN mode. No files were moved.")
        print("To automatically isolate the flagged samples into quarantine, run:")
        print("  python auto_clean_dataset.py --quarantine\n")


if __name__ == "__main__":
    main()
