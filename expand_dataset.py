"""
Synthetic Dataset Expander for TAR Pipeline
============================================
Generates realistic synthetic training samples from existing (48, 332) feature
sequences using physics-informed augmentation strategies that go FAR beyond
the basic on-the-fly augmentations in train_tar.py.

Augmentation Strategies:
  1. Temporal Interpolation (Mixup): Blend two same-class samples with random λ
  2. Speed Variation: Time-stretch/compress by 0.7x to 1.4x (wider than training aug)
  3. Hand Mirror Swap: Swap left ↔ right hand features (simulates left-handed operator)
  4. Pose Magnitude Scaling: Scale pose coords to simulate different body sizes/distances
  5. Landmark Jitter: Gaussian noise on pose/hand channels (heavier than train_tar)
  6. Temporal Crop & Resample: Take random sub-windows and resample back to 48 frames
  7. Object Position Perturbation: Shift object cx/cy to simulate different table layouts
  8. Velocity Recomputation: After any spatial transform, recompute velocity from base

Usage:
    # Preview how many samples would be generated (dry run):
    python expand_dataset.py --dry-run

    # Generate synthetic samples (default target: 70 per class):
    python expand_dataset.py

    # Custom target per class:
    python expand_dataset.py --target-per-class 100

    # Only expand specific under-represented classes:
    python expand_dataset.py --classes 2 6
"""

import os
import re
import csv
import argparse
import numpy as np
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DATASET_DIR = "dataset_advanced"
LABELS = ["idle", "open_box", "pick_red", "place_red_out",
          "pick_blue", "place_blue_in", "close_box"]

SEQ_LEN = 48
BASE_DIM = 166  # pose(66) + left_hand(42) + right_hand(42) + objects(9) + distances(6) + lid(1)
TOTAL_DIM = 332  # BASE_DIM + velocity(BASE_DIM)

# Feature layout indices (within the 166-dim base vector)
POSE_SLICE = slice(0, 66)        # 33 landmarks × 2 (x,y)
LEFT_HAND_SLICE = slice(66, 108)  # 21 landmarks × 2
RIGHT_HAND_SLICE = slice(108, 150)  # 21 landmarks × 2
RED_OBJ_SLICE = slice(150, 153)   # cx, cy, area
BLUE_OBJ_SLICE = slice(153, 156)  # cx, cy, area
MAIN_BOX_SLICE = slice(156, 159)  # cx, cy, area
DIST_SLICE = slice(159, 165)      # 6 wrist-to-object distances
LID_SLICE = slice(165, 166)       # lid complexity score

DEFAULT_TARGET_PER_CLASS = 70


# ============================================================
# AUGMENTATION FUNCTIONS
# ============================================================

def decompose(sample):
    """Split a (48, 332) sample into base (48, 166) and recompute velocity."""
    base = sample[:, :BASE_DIM].copy()
    return base


def recompose(base):
    """Recompute velocity from base features and return full (48, 332) sample."""
    velocity = np.zeros_like(base)
    velocity[1:] = base[1:] - base[:-1]
    return np.concatenate([base, velocity], axis=1).astype(np.float32)


def aug_temporal_mixup(sample_a, sample_b, alpha_range=(0.15, 0.45)):
    """
    Blend two same-class samples with random λ.
    Creates a plausible intermediate trajectory between two execution styles.
    """
    lam = np.random.uniform(*alpha_range)
    base_a = decompose(sample_a)
    base_b = decompose(sample_b)
    mixed = lam * base_a + (1.0 - lam) * base_b
    return recompose(mixed)


def aug_speed_variation(sample, speed_range=(0.7, 1.4)):
    """
    Time-stretch/compress the sequence, then resample back to SEQ_LEN.
    Wider range than train_tar.py's 0.85-1.15x.
    """
    base = decompose(sample)
    T, D = base.shape
    speed = np.random.uniform(*speed_range)
    new_len = max(20, min(90, int(round(T * speed))))

    # Interpolate to new length
    orig_idx = np.linspace(0, T - 1, num=new_len)
    stretched = np.zeros((new_len, D), dtype=np.float32)
    for d in range(D):
        stretched[:, d] = np.interp(orig_idx, np.arange(T), base[:, d])

    # Resample back to T=48
    target_idx = np.linspace(0, new_len - 1, num=T)
    resampled = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        resampled[:, d] = np.interp(target_idx, np.arange(new_len), stretched[:, d])

    return recompose(resampled)


def aug_hand_mirror(sample):
    """
    Swap left ↔ right hand features.
    Simulates a left-handed operator performing the same action.
    Also swaps the corresponding distance features (L-wrist ↔ R-wrist distances).
    """
    base = decompose(sample)
    mirrored = base.copy()

    # Swap left and right hand landmark blocks
    left = base[:, LEFT_HAND_SLICE].copy()
    right = base[:, RIGHT_HAND_SLICE].copy()
    mirrored[:, LEFT_HAND_SLICE] = right
    mirrored[:, RIGHT_HAND_SLICE] = left

    # Swap wrist-to-object distance pairs (L↔R for each of red/blue/main)
    # dist layout: [L_to_red, R_to_red, L_to_blue, R_to_blue, L_to_main, R_to_main]
    dists = base[:, DIST_SLICE].copy()
    mirrored[:, 159] = dists[:, 1]  # L_to_red  <- R_to_red
    mirrored[:, 160] = dists[:, 0]  # R_to_red  <- L_to_red
    mirrored[:, 161] = dists[:, 3]  # L_to_blue <- R_to_blue
    mirrored[:, 162] = dists[:, 2]  # R_to_blue <- L_to_blue
    mirrored[:, 163] = dists[:, 5]  # L_to_main <- R_to_main
    mirrored[:, 164] = dists[:, 4]  # R_to_main <- L_to_main

    # Also mirror pose X coordinates (every other value in pose = x,y,x,y...)
    # Negate X displacement from center (the pose is hip-centered)
    for i in range(0, 66, 2):  # x coordinates at even indices
        mirrored[:, i] = -base[:, i]

    return recompose(mirrored)


def aug_pose_scale(sample, scale_range=(0.82, 1.22)):
    """
    Scale pose and hand coordinates to simulate different body sizes or
    camera distances. Object features are NOT scaled (they come from the
    frame, not the body).
    """
    base = decompose(sample)
    scale = np.random.uniform(*scale_range)

    # Scale pose landmarks
    base[:, POSE_SLICE] *= scale
    # Scale hand landmarks
    base[:, LEFT_HAND_SLICE] *= scale
    base[:, RIGHT_HAND_SLICE] *= scale
    # Distances scale proportionally
    base[:, DIST_SLICE] *= scale

    return recompose(base)


def aug_landmark_jitter(sample, sigma=0.015):
    """
    Add Gaussian noise to pose and hand channels.
    Heavier than train_tar.py's 0.008 sigma — simulates noisier MediaPipe tracking.
    """
    base = decompose(sample)

    # Jitter pose landmarks
    base[:, POSE_SLICE] += np.random.normal(0, sigma, base[:, POSE_SLICE].shape).astype(np.float32)
    # Jitter hand landmarks (slightly less noise)
    base[:, LEFT_HAND_SLICE] += np.random.normal(0, sigma * 0.8, base[:, LEFT_HAND_SLICE].shape).astype(np.float32)
    base[:, RIGHT_HAND_SLICE] += np.random.normal(0, sigma * 0.8, base[:, RIGHT_HAND_SLICE].shape).astype(np.float32)

    return recompose(base)


def aug_temporal_crop(sample, crop_fraction=(0.6, 0.85)):
    """
    Take a random contiguous sub-window of the sequence and resample
    back to SEQ_LEN. Simulates capturing the action at a different
    point in the sliding window.
    """
    base = decompose(sample)
    T, D = base.shape
    frac = np.random.uniform(*crop_fraction)
    crop_len = max(16, int(T * frac))
    max_start = T - crop_len
    start = np.random.randint(0, max(1, max_start + 1))
    cropped = base[start:start + crop_len]

    # Resample back to T=48
    target_idx = np.linspace(0, crop_len - 1, num=T)
    resampled = np.zeros((T, D), dtype=np.float32)
    for d in range(D):
        resampled[:, d] = np.interp(target_idx, np.arange(crop_len), cropped[:, d])

    return recompose(resampled)


def aug_object_perturbation(sample, pos_sigma=0.03, area_sigma=0.01):
    """
    Slightly shift object center positions and area to simulate different
    table layouts or camera angles. Small enough to stay realistic.
    """
    base = decompose(sample)

    for obj_slice in [RED_OBJ_SLICE, BLUE_OBJ_SLICE, MAIN_BOX_SLICE]:
        obj = base[:, obj_slice].copy()
        # Only perturb non-zero entries (object was detected)
        mask = np.any(obj != 0, axis=1)
        if mask.any():
            # Perturb cx, cy
            obj[mask, 0] += np.random.normal(0, pos_sigma, mask.sum()).astype(np.float32)
            obj[mask, 1] += np.random.normal(0, pos_sigma, mask.sum()).astype(np.float32)
            # Perturb area
            obj[mask, 2] += np.random.normal(0, area_sigma, mask.sum()).astype(np.float32)
            obj[mask, 2] = np.clip(obj[mask, 2], 0, 1)  # area stays non-negative
        base[:, obj_slice] = obj

    return recompose(base)


def aug_composite(sample, other_sample=None):
    """
    Apply a random chain of 2-4 augmentations to create a richly varied sample.
    This ensures synthetic samples are diverse, not just single-transform variants.
    """
    base = decompose(sample)
    result = recompose(base)

    # Randomly select 2-4 augmentations
    augs = [
        lambda s: aug_speed_variation(s, (0.75, 1.30)),
        lambda s: aug_pose_scale(s, (0.85, 1.18)),
        lambda s: aug_landmark_jitter(s, sigma=np.random.uniform(0.008, 0.020)),
        lambda s: aug_temporal_crop(s, (0.65, 0.90)),
        lambda s: aug_object_perturbation(s, pos_sigma=0.025),
    ]

    n_augs = np.random.randint(2, 5)
    chosen = np.random.choice(len(augs), size=n_augs, replace=False)

    for idx in chosen:
        result = augs[idx](result)

    return result


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_samples(dataset_dir):
    """Find all valid label_X_Y.npy files grouped by class."""
    pattern = re.compile(r"^label_(\d+)_(\d+)\.npy$")
    class_samples = {i: [] for i in range(len(LABELS))}

    for fname in sorted(os.listdir(dataset_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        label_id = int(m.group(1))
        fpath = os.path.join(dataset_dir, fname)

        # Only include synthetic-generation-eligible files (skip already-synthetic ones)
        if label_id < len(LABELS):
            class_samples[label_id].append(fpath)

    return class_samples


def get_next_id(dataset_dir, label_id):
    """Get the next available file index for a label class."""
    pattern = re.compile(rf"^label_{label_id}_(\d+)\.npy$")
    existing = []
    for f in os.listdir(dataset_dir):
        m = pattern.match(f)
        if m:
            existing.append(int(m.group(1)))
    return max(existing, default=-1) + 1


# ============================================================
# MAIN GENERATION LOGIC
# ============================================================

def expand_dataset(dataset_dir=DATASET_DIR,
                   target_per_class=DEFAULT_TARGET_PER_CLASS,
                   target_classes=None,
                   dry_run=False):

    print("=" * 65)
    print("  SYNTHETIC DATASET EXPANDER")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print(f"  Source: {os.path.abspath(dataset_dir)}")
    print(f"  Target per class: {target_per_class}")
    print(f"  Mode: {'DRY RUN (preview only)' if dry_run else 'GENERATE'}")
    print("=" * 65)

    class_samples = discover_samples(dataset_dir)

    total_original = 0
    total_to_generate = 0
    generation_plan = {}

    print(f"\n  {'Class':<20} {'Current':>8} {'Target':>8} {'To Generate':>12}")
    print("  " + "-" * 52)

    for cid in range(len(LABELS)):
        current = len(class_samples[cid])
        total_original += current

        if target_classes and cid not in target_classes:
            needed = 0
        else:
            needed = max(0, target_per_class - current)

        total_to_generate += needed
        generation_plan[cid] = needed
        status = "" if needed == 0 else f"  (+{needed})"
        print(f"  {LABELS[cid]:<20} {current:>8} {target_per_class:>8} {needed:>12}{status}")

    print(f"\n  Total original: {total_original}")
    print(f"  Total to generate: {total_to_generate}")
    print(f"  Final dataset size: {total_original + total_to_generate}")

    if dry_run:
        print("\n  [DRY RUN] No files written. Remove --dry-run to generate.")
        return

    if total_to_generate == 0:
        print("\n  All classes already meet the target. Nothing to generate.")
        return

    # --- Generate synthetic samples ---
    print(f"\n  Generating synthetic samples...")
    generated_log = []

    for cid in range(len(LABELS)):
        needed = generation_plan[cid]
        if needed == 0:
            continue

        originals = class_samples[cid]
        if len(originals) < 2:
            print(f"  [WARN] Class {cid} ({LABELS[cid]}) has <2 samples, skipping mixup variants")

        # Load all original samples for this class
        original_data = [np.load(fp).astype(np.float32) for fp in originals]
        next_id = get_next_id(dataset_dir, cid)
        class_generated = 0

        # Strategy distribution for variety:
        #   30% composite (multi-aug chain)
        #   20% mixup (blend two samples)
        #   15% speed variation
        #   10% hand mirror
        #   10% temporal crop
        #   10% pose scale + jitter
        #   5%  object perturbation

        for i in range(needed):
            roll = np.random.random()
            src_idx = np.random.randint(0, len(original_data))
            src = original_data[src_idx]

            try:
                if roll < 0.30:
                    # Composite (chain of 2-4 augmentations)
                    synthetic = aug_composite(src)
                elif roll < 0.50 and len(original_data) >= 2:
                    # Mixup: blend two different same-class originals
                    other_idx = np.random.randint(0, len(original_data))
                    while other_idx == src_idx and len(original_data) > 1:
                        other_idx = np.random.randint(0, len(original_data))
                    synthetic = aug_temporal_mixup(src, original_data[other_idx])
                elif roll < 0.65:
                    # Speed variation
                    synthetic = aug_speed_variation(src)
                elif roll < 0.75:
                    # Hand mirror swap
                    synthetic = aug_hand_mirror(src)
                elif roll < 0.85:
                    # Temporal crop & resample
                    synthetic = aug_temporal_crop(src)
                elif roll < 0.95:
                    # Pose scale + jitter combo
                    synthetic = aug_pose_scale(aug_landmark_jitter(src))
                else:
                    # Object perturbation + jitter
                    synthetic = aug_object_perturbation(aug_landmark_jitter(src, sigma=0.012))

                # Validate shape
                assert synthetic.shape == (SEQ_LEN, TOTAL_DIM), f"Bad shape: {synthetic.shape}"
                assert not np.any(np.isnan(synthetic)), "NaN detected"
                assert not np.any(np.isinf(synthetic)), "Inf detected"

                # Save
                fname = f"label_{cid}_{next_id}.npy"
                fpath = os.path.join(dataset_dir, fname)
                np.save(fpath, synthetic)
                generated_log.append({"filename": fname, "label_id": cid, "label_name": LABELS[cid]})
                next_id += 1
                class_generated += 1

            except Exception as e:
                print(f"  [WARN] Failed to generate sample for class {cid}: {e}")
                continue

        print(f"  [DONE] {LABELS[cid]:<16}: +{class_generated} synthetic samples generated")

    # --- Update metadata.csv ---
    print(f"\n  Rebuilding metadata.csv...")
    meta_path = os.path.join(dataset_dir, "metadata.csv")
    pattern = re.compile(r"^label_(\d+)_(\d+)\.npy$")
    entries = []
    for f in sorted(os.listdir(dataset_dir)):
        m = pattern.match(f)
        if m:
            label_id = int(m.group(1))
            label_name = LABELS[label_id] if label_id < len(LABELS) else f"class_{label_id}"
            fpath = os.path.join(dataset_dir, f)
            entries.append({
                "filename": f,
                "label_id": label_id,
                "label_name": label_name,
                "shape": f"{SEQ_LEN}x{TOTAL_DIM}",
                "size_bytes": os.path.getsize(fpath)
            })

    with open(meta_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["filename", "label_id", "label_name", "shape", "size_bytes"])
        writer.writeheader()
        writer.writerows(entries)

    # --- Final Report ---
    final_counts = {}
    for e in entries:
        cid = e["label_id"]
        final_counts[cid] = final_counts.get(cid, 0) + 1

    print(f"\n{'=' * 65}")
    print(f"  EXPANSION COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Synthetic samples generated: {len(generated_log)}")
    print(f"  Total dataset size: {len(entries)}")
    print(f"\n  {'Class':<20} {'Original':>10} {'+ Synthetic':>12} {'= Total':>10}")
    print("  " + "-" * 55)
    for cid in range(len(LABELS)):
        orig = len(class_samples[cid])
        total = final_counts.get(cid, 0)
        synth = total - orig
        print(f"  {LABELS[cid]:<20} {orig:>10} {'+' + str(synth):>12} {total:>10}")

    print(f"\n  Dataset ready for training with: python train_tar.py")
    print(f"  Or fine-tune with: python retrain_tar.py")
    print(f"{'=' * 65}")


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand TAR dataset with synthetic augmented samples")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview generation plan without writing files")
    parser.add_argument("--target-per-class", type=int, default=DEFAULT_TARGET_PER_CLASS,
                        help=f"Target number of samples per class (default: {DEFAULT_TARGET_PER_CLASS})")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="Only expand specific class IDs (e.g., --classes 2 6)")
    parser.add_argument("--dataset", type=str, default=DATASET_DIR,
                        help=f"Dataset directory (default: {DATASET_DIR})")

    args = parser.parse_args()

    expand_dataset(
        dataset_dir=args.dataset,
        target_per_class=args.target_per_class,
        target_classes=args.classes,
        dry_run=args.dry_run
    )
