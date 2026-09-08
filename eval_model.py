"""
Evaluates a trained TAR checkpoint with a full per-class breakdown -
precision/recall/F1 per action and a confusion matrix.

Usage:
    python eval_model.py
    python eval_model.py --dataset_path dataset_advanced
    python eval_model.py --dataset_path dataset
"""

import os
import re
import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from model_def import TARModel, NUM_CLASSES, FEATURE_DIM, SEQ_LEN
import feature_utils as fu
from train_tar import load_valid_dataset, FileTARDataset, ArrayTARDataset, DATASET_PATH as TRAIN_DATASET_PATH

DEFAULT_MODEL_PATH = "best_tar_model1.pth" if os.path.exists("best_tar_model1.pth") else "best_tar_model.pth"


def load_model(path):
    model = TARModel(input_size=FEATURE_DIM, num_classes=NUM_CLASSES)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate_dataset(model, ds):
    preds, targets, confidences = [], [], []

    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            out = model(x.unsqueeze(0))
            probs = torch.softmax(out, dim=1)[0]
            pred = int(torch.argmax(probs))
            preds.append(pred)
            targets.append(int(y))
            confidences.append(float(probs[pred]))

    label_ids = sorted(set(targets) | set(preds))
    target_names = [fu.LABELS[i] if i < len(fu.LABELS) else f"class_{i}" for i in label_ids]

    print("=== Classification Report ===")
    print(classification_report(targets, preds, labels=label_ids, target_names=target_names, zero_division=0))

    print("=== Confusion Matrix (rows=true, cols=predicted) ===")
    print("Labels Order:", target_names)
    print(confusion_matrix(targets, preds, labels=label_ids))

    acc = np.mean(np.array(preds) == np.array(targets)) * 100
    print(f"\nOverall Accuracy: {acc:.2f}%\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--dataset_path", default=None,
                        help="Path to dataset directory. Defaults to training dataset.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model file '{args.model_path}' not found. Please train model first.")
        return

    model = load_model(args.model_path)
    target_dir = args.dataset_path or TRAIN_DATASET_PATH

    print(f"[INFO] Evaluating model '{args.model_path}' on '{target_dir}'...")

    mode, data_obj = load_valid_dataset(target_dir)
    if mode == "array":
        X, y = data_obj
        if args.dataset_path is None:
            # Evaluate on held-out validation fold (same split as train_tar.py)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            _, val_idx = next(skf.split(X, y))
            val_ds = ArrayTARDataset(X[val_idx], y[val_idx])
            print(f"[INFO] Evaluated on held-out validation fold ({len(val_ds)} samples):\n")
            evaluate_dataset(model, val_ds)
        else:
            # If explicit external dataset provided, evaluate on all samples
            ds = ArrayTARDataset(X, y)
            print(f"[INFO] Evaluated on full external dataset ({len(ds)} samples):\n")
            evaluate_dataset(model, ds)

    elif mode == "files":
        samples = data_obj
        labels = [s[1] for s in samples]
        if args.dataset_path is None:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            _, val_idx = next(skf.split(samples, labels))
            val_samples = [samples[i] for i in val_idx]
            val_ds = FileTARDataset(val_samples)
            print(f"[INFO] Evaluated on held-out validation fold ({len(val_ds)} samples):\n")
            evaluate_dataset(model, val_ds)
        else:
            ds = FileTARDataset(samples)
            print(f"[INFO] Evaluated on full external dataset ({len(ds)} samples):\n")
            evaluate_dataset(model, ds)
    else:
        print(f"[ERROR] Could not load valid dataset from '{target_dir}'.")


if __name__ == "__main__":
    main()