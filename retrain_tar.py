"""
Retrain / Fine-tune an existing TARModel checkpoint on a new or updated dataset.

Features:
  - Loads an existing checkpoint (e.g. best_tar_model.pth).
  - Automatically creates a backup of the base model before overwriting.
  - Runs a pre-retraining baseline evaluation so you can compare before vs after.
  - Fine-tunes with a lower default learning rate (1e-4) to protect learned features.
  - Optional `--combine_with` flag to merge new samples with an older dataset to
    prevent catastrophic forgetting.
  - Optional `--freeze_backbone` flag to only tune the classifier head.

Usage:
    # Run with defaults (configured below):
    python retrain_tar.py

    # Specify custom dataset and epochs:
    python retrain_tar.py --dataset_path dataset_final --epochs 30

    # Combine new dataset with original backup to prevent forgetting:
    python retrain_tar.py --dataset_path dataset_final --combine_with dataset_backup/dataset_advanced

    # Specify custom output path:
    python retrain_tar.py --dataset_path dataset_final --output best_tar_model_v2.pth
"""

import os
import re
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from model_def import TARModel, NUM_CLASSES, FEATURE_DIM, SEQ_LEN
import feature_utils as fu

# ==============================================================================
# DEFAULT CONFIG (used when running without CLI arguments)
# ==============================================================================
DEFAULT_BASE_MODEL = "best_tar_model.pth"
DEFAULT_NEW_DATASET = "dataset_advanced"       # Change this to your new dataset folder
DEFAULT_COMBINE_WITH = None                  # e.g. "dataset_backup/dataset_advanced"
DEFAULT_OUTPUT_MODEL = "best_tar_model1.pth"
DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4                            # Lower LR for fine-tuning
DEFAULT_FREEZE_BACKBONE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# DATA AUGMENTATION FOR HAR SEQUENCES
# ==============================================================================
def augment_sequence(data, is_training=True):
    """
    Realistic sliding-window and temporal augmentations:
      1. Temporal shift (sliding window jitter): Shifts sequence left or right by up to 6 frames.
         Simulates live sliding-window misalignment when action is not perfectly centered.
      2. Time-warping (speed variation): Interpolates time axis by 0.85x to 1.15x.
      3. Landmark Gaussian Jitter: Adds subtle noise (sigma=0.008) to coordinates.
      4. Channel/Dropout masking: Randomly zeroes out a 2-4 frame block (simulates MediaPipe occlusion).
    """
    if not is_training:
        return data

    seq = data.copy()
    T, F = seq.shape

    # 1. Temporal shift (simulate rolling window misalignment)
    if np.random.rand() < 0.75:
        shift = np.random.randint(-6, 7)
        if shift > 0:
            seq = np.vstack([np.repeat(seq[:1], shift, axis=0), seq[:-shift]])
        elif shift < 0:
            shift_abs = abs(shift)
            seq = np.vstack([seq[shift_abs:], np.repeat(seq[-1:], shift_abs, axis=0)])

    # 2. Time-warping / speed scaling (0.85x to 1.15x)
    if np.random.rand() < 0.50:
        speed = np.random.uniform(0.85, 1.15)
        new_len = max(24, min(72, int(round(T * speed))))
        orig_indices = np.linspace(0, T - 1, num=new_len)
        warped = np.zeros((new_len, F), dtype=np.float32)
        for c in range(F):
            warped[:, c] = np.interp(orig_indices, np.arange(T), seq[:, c])
        # Resample back to T=48
        target_indices = np.linspace(0, new_len - 1, num=T)
        resampled = np.zeros((T, F), dtype=np.float32)
        for c in range(F):
            resampled[:, c] = np.interp(target_indices, np.arange(new_len), warped[:, c])
        seq = resampled

    # 3. Subtle Gaussian noise on coordinates
    if np.random.rand() < 0.50:
        noise = np.random.normal(0.0, 0.008, size=seq.shape).astype(np.float32)
        seq += noise

    # 4. Temporal Block Dropout (simulates brief tracker dropouts)
    if np.random.rand() < 0.35:
        drop_len = np.random.randint(2, 5)
        drop_start = np.random.randint(0, max(1, T - drop_len))
        seq[drop_start:drop_start + drop_len] = 0.0

    return seq


# ==============================================================================
# DATASET HELPERS
# ==============================================================================
class FileTARDataset(Dataset):
    """Dataset for individual .npy files (label_X_Y.npy)."""
    def __init__(self, file_label_pairs, is_training=False):
        self.samples = file_label_pairs
        self.is_training = is_training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path).astype(np.float32)

        # Ensure correct shape (SEQ_LEN, FEATURE_DIM)
        if data.shape != (SEQ_LEN, FEATURE_DIM):
            if data.shape[1] == FEATURE_DIM:
                data = fu.resample_sequence(data, target_len=SEQ_LEN)
            else:
                raise ValueError(
                    f"File {file_path} shape {data.shape} incompatible with ({SEQ_LEN}, {FEATURE_DIM})"
                )

        if self.is_training:
            data = augment_sequence(data, is_training=True)

        # Z-score normalization matching train_tar.py
        mean = np.mean(data)
        std = np.std(data) + 1e-6
        data = (data - mean) / std
        data = np.clip(data, -5.0, 5.0)

        return torch.tensor(data, dtype=torch.float32), label


class ArrayTARDataset(Dataset):
    """Dataset for precomputed (X, y) arrays."""
    def __init__(self, x_data, y_data, is_training=False):
        self.x = x_data.astype(np.float32)
        self.y = y_data.astype(np.int64)
        self.is_training = is_training

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        data = self.x[idx]
        label = int(self.y[idx])

        if self.is_training:
            data = augment_sequence(data, is_training=True)

        mean = np.mean(data)
        std = np.std(data) + 1e-6
        data = (data - mean) / std
        data = np.clip(data, -5.0, 5.0)

        return torch.tensor(data, dtype=torch.float32), label


def discover_files(folder_path):
    """Finds all valid label_X_Y.npy files in a folder with FEATURE_DIM columns."""
    if not os.path.exists(folder_path):
        print(f"[WARN] Directory '{folder_path}' does not exist.")
        return []

    valid_samples = []
    skipped = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".npy") or fname in ("X.npy", "y.npy"):
            continue

        match = re.match(r"^label_(\d+)_\d+\.npy$", fname)
        if not match:
            continue

        label = int(match.group(1))
        fpath = os.path.join(folder_path, fname)

        try:
            arr = np.load(fpath)
            if arr.ndim == 2 and arr.shape[1] == FEATURE_DIM:
                valid_samples.append((fpath, label))
            else:
                skipped.append((fname, f"Shape {arr.shape} != (*, {FEATURE_DIM})"))
        except Exception as e:
            skipped.append((fname, str(e)))

    if skipped:
        print(f"[WARN] In '{folder_path}', skipped {len(skipped)} files:")
        for f, reason in skipped[:3]:
            print(f"   - {f}: {reason}")
        if len(skipped) > 3:
            print(f"   ... and {len(skipped) - 3} more.")

    return valid_samples


def load_dataset(dataset_path, combine_with=None):
    """Loads dataset from folder, optionally merging with a base/backup dataset."""
    x_path = os.path.join(dataset_path, "X.npy")
    y_path = os.path.join(dataset_path, "y.npy")

    # If dataset has precomputed X.npy and y.npy and no combine requested
    if os.path.exists(x_path) and os.path.exists(y_path) and not combine_with:
        X = np.load(x_path)
        y = np.load(y_path)
        if X.ndim == 3 and X.shape[1] == SEQ_LEN and X.shape[2] == FEATURE_DIM:
            print(f"[DATASET] Loaded array dataset from '{dataset_path}': X={X.shape}, y={y.shape}")
            return "array", (X, y)

    # Load file pairs from primary dataset
    primary_samples = discover_files(dataset_path)
    print(f"[DATASET] Found {len(primary_samples)} samples in '{dataset_path}'")

    all_samples = list(primary_samples)
    if combine_with and os.path.exists(combine_with):
        combined_samples = discover_files(combine_with)
        print(f"[DATASET] Combining with {len(combined_samples)} samples from '{combine_with}'")
        all_samples.extend(combined_samples)

    return "files", all_samples


# ==============================================================================
# EVALUATION HELPER
# ==============================================================================
def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * len(y)

            p = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(p)
            targets.extend(y.cpu().numpy())

    avg_loss = total_loss / max(len(preds), 1)
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ==============================================================================
# MAIN RETRAINING ROUTINE
# ==============================================================================
def retrain(base_model_path=DEFAULT_BASE_MODEL,
            dataset_path=DEFAULT_NEW_DATASET,
            combine_with=DEFAULT_COMBINE_WITH,
            output_model_path=DEFAULT_OUTPUT_MODEL,
            epochs=DEFAULT_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
            lr=DEFAULT_LR,
            freeze_backbone=DEFAULT_FREEZE_BACKBONE):

    print("=" * 70)
    print(" " * 20 + "TAR MODEL RETRAINING")
    print("=" * 70)
    print(f"  Base Model   : {base_model_path}")
    print(f"  New Dataset  : {dataset_path}")
    if combine_with:
        print(f"  Combine With : {combine_with}")
    print(f"  Output Model : {output_model_path}")
    print(f"  Epochs       : {epochs}")
    print(f"  Learning Rate: {lr}")
    print(f"  Freeze Head  : {freeze_backbone}")
    print(f"  Device       : {DEVICE}")
    print("=" * 70)

    # 1. Check & Load Existing Model
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model '{base_model_path}' not found! Train an initial model first."
        )

    # Backup the original model so it is never lost
    backup_path = f"{os.path.splitext(base_model_path)[0]}_backup.pth"
    if not os.path.exists(backup_path):
        shutil.copyfile(base_model_path, backup_path)
        print(f"[BACKUP] Created backup of base model at '{backup_path}'")

    model = TARModel(input_size=FEATURE_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(base_model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    print(f"[MODEL] Successfully loaded checkpoint weights from '{base_model_path}'")

    # Optional: freeze transformer backbone (train only classification layers)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("[MODEL] Backbone frozen. Only training classification head.")

    # 2. Load Dataset
    mode, data_obj = load_dataset(dataset_path, combine_with=combine_with)

    if mode == "array":
        X, y = data_obj
        labels = y.tolist()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(X, y))

        train_ds = ArrayTARDataset(X[train_idx], y[train_idx], is_training=True)
        val_ds = ArrayTARDataset(X[val_idx], y[val_idx], is_training=False)
    else:
        samples = data_obj
        if len(samples) < NUM_CLASSES * 2:
            print(f"[ERROR] Only {len(samples)} valid samples found.")
            print(f"Need at least {NUM_CLASSES * 2} samples across the 7 classes to train & validate.")
            return

        labels = [s[1] for s in samples]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(samples, labels))

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        train_ds = FileTARDataset(train_samples, is_training=True)
        val_ds = FileTARDataset(val_samples, is_training=False)

    print(f"\n[SPLIT] Train set: {len(train_ds)} samples | Val set: {len(val_ds)} samples")
    print("[CLASSES] Distribution across dataset:")
    for cid in range(NUM_CLASSES):
        cname = fu.LABELS[cid] if cid < len(fu.LABELS) else f"class_{cid}"
        count = sum(1 for l in labels if l == cid)
        print(f"   Class {cid} ({cname}): {count} samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 3. Baseline Validation Before Training
    base_loss, base_acc, base_f1 = evaluate(model, val_loader, criterion)
    print(f"\n[BASELINE] Pre-retraining performance on new validation split:")
    print(f"   --> Val Loss: {base_loss:.4f} | Val Acc: {base_acc:.4f} | Val F1: {base_f1:.4f}\n")

    best_f1 = base_f1
    saved_best = False

    # 4. Retraining Loop
    print(f"[TRAIN] Starting fine-tuning for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)

        scheduler.step()
        train_loss = total_loss / len(train_ds)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:02d}/{epochs:02d} [lr={current_lr:.2e}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}", end="")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), output_model_path)
            saved_best = True
            print(f"  --> [*] New best! Saved to '{output_model_path}' (F1: {best_f1:.4f})")
        else:
            print("")

    # If the model didn't beat the baseline, save the final model or keep baseline
    if not saved_best:
        torch.save(model.state_dict(), output_model_path)
        print(f"\n[NOTE] Baseline was already strong. Saved final epoch checkpoint to '{output_model_path}' (F1: {val_f1:.4f})")
    else:
        print(f"\n[DONE] Retraining complete! Best model saved to '{output_model_path}' with F1: {best_f1:.4f}")
        print(f"(Baseline was F1: {base_f1:.4f} -> Improved to F1: {best_f1:.4f})")


# ==============================================================================
# CLI PARSER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain / fine-tune an existing TARModel checkpoint.")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL,
                        help="Path to initial .pth checkpoint (default: best_tar_model.pth)")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_NEW_DATASET,
                        help="Path to new dataset directory (default: dataset_advanced)")
    parser.add_argument("--combine_with", type=str, default=DEFAULT_COMBINE_WITH,
                        help="Optional older dataset directory to merge with (prevents forgetting)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_MODEL,
                        help="Path to save retrained model (default: best_tar_model.pth)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Number of fine-tuning epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--freeze_backbone", action="store_true", default=DEFAULT_FREEZE_BACKBONE,
                        help="Freeze transformer and only train classifier head")

    args = parser.parse_args()

    retrain(
        base_model_path=args.base_model,
        dataset_path=args.dataset_path,
        combine_with=args.combine_with,
        output_model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone
    )
