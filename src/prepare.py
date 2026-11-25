import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from load_data import load_raw, CHANNELS

# Fixed validation volunteers so all splits are strictly person-disjoint
VAL_SUBJECTS = [3, 15, 25, 30]

EXPECTED_TRAIN_COUNTS = [1226, 1073, 986, 1286, 1374, 1407]
EXPECTED_TEST_COUNTS = [496, 471, 420, 491, 532, 537]


def compute_channel_stats_manual(X_train):
    """Compute per-channel mean and std across all training windows and 128 steps."""
    num_windows, seq_len, num_channels = X_train.shape
    total_points = float(num_windows * seq_len)

    means = np.zeros(num_channels, dtype=np.float32)
    stds = np.zeros(num_channels, dtype=np.float32)

    for c in range(num_channels):
        channel_vals = X_train[:, :, c]
        m = float(np.sum(channel_vals, dtype=np.float64) / total_points)
        diff = channel_vals.astype(np.float64) - m
        var = float(np.sum(diff * diff) / total_points)
        means[c] = np.float32(m)
        stds[c] = np.float32(var ** 0.5)

    return means, stds


def standardise_windows(X, mean, std):
    """Subtract training channel mean and divide by training channel std."""
    out = np.empty_like(X, dtype=np.float32)
    for c in range(X.shape[2]):
        out[:, :, c] = (X[:, :, c] - mean[c]) / std[c]
    return out


def prepare(data_dir="data"):
    """Load raw HAR signals, shift labels to 0..5, split by subject, standardise, and convert to tensors."""
    raw = load_raw(data_dir)
    names = raw["activity_names"]
    train = raw["train"]
    test = raw["test"]

    # Convert activity labels from 1..6 to 0..5
    y_train_full = train["y"] - 1
    y_test = test["y"] - 1

    # Validation split strictly by volunteer ID
    is_val = np.isin(train["subject"], VAL_SUBJECTS)
    is_tr = ~is_val

    X_tr_raw = train["X"][is_tr]
    y_tr = y_train_full[is_tr]
    subj_tr = train["subject"][is_tr]

    X_va_raw = train["X"][is_val]
    y_va = y_train_full[is_val]
    subj_va = train["subject"][is_val]

    X_te_raw = test["X"]
    subj_te = test["subject"]

    # Compute mean and standard deviation using ONLY training windows
    mean, std = compute_channel_stats_manual(X_tr_raw)

    X_tr = standardise_windows(X_tr_raw, mean, std)
    X_va = standardise_windows(X_va_raw, mean, std)
    X_te = standardise_windows(X_te_raw, mean, std)

    # Convert signals to float32 tensors (N, 128, 9) and labels to long tensors (N,)
    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.long)

    X_va_tensor = torch.tensor(X_va, dtype=torch.float32)
    y_va_tensor = torch.tensor(y_va, dtype=torch.long)

    X_te_tensor = torch.tensor(X_te, dtype=torch.float32)
    y_te_tensor = torch.tensor(y_test, dtype=torch.long)

    return {
        "train": (X_tr_tensor, y_tr_tensor),
        "val": (X_va_tensor, y_va_tensor),
        "test": (X_te_tensor, y_te_tensor),
        "mean": mean,
        "std": std,
        "names": names,
        "subjects": {
            "train": sorted(list(set(subj_tr.tolist()))),
            "val": sorted(list(set(subj_va.tolist()))),
            "test": sorted(list(set(subj_te.tolist()))),
        },
        "raw_label_counts": {
            "train_full": [int(np.sum(y_train_full == k)) for k in range(6)],
            "test": [int(np.sum(y_test == k)) for k in range(6)],
        },
    }


if __name__ == "__main__":
    data = prepare()
    names = data["names"]

    print("=== Activity Window Counts (Labels 0..5) ===")
    print(f"{'Idx':<4} {'Activity':<20} {'Train+Val':>10} {'Expected':>10} {'Test':>8} {'Expected':>10}")
    for k in range(6):
        tr_c = data["raw_label_counts"]["train_full"][k]
        te_c = data["raw_label_counts"]["test"][k]
        print(
            f"{k:<4} {names[k]:<20} {tr_c:>10} {EXPECTED_TRAIN_COUNTS[k]:>10} "
            f"{te_c:>8} {EXPECTED_TEST_COUNTS[k]:>10}"
        )
    print(
        f"{'Tot':<4} {'ALL ACTIVITIES':<20} "
        f"{sum(data['raw_label_counts']['train_full']):>10} {sum(EXPECTED_TRAIN_COUNTS):>10} "
        f"{sum(data['raw_label_counts']['test']):>8} {sum(EXPECTED_TEST_COUNTS):>10}\n"
    )

    print("=== Person-Disjoint Split Summary ===")
    for split_key in ["train", "val", "test"]:
        X_t, y_t = data[split_key]
        vols = data["subjects"][split_key]
        print(
            f"{split_key:5s} -> X: {tuple(X_t.shape)} ({X_t.dtype}), "
            f"y: {tuple(y_t.shape)} ({y_t.dtype}), "
            f"volunteers ({len(vols)}): {vols}"
        )

    s_tr = set(data["subjects"]["train"])
    s_va = set(data["subjects"]["val"])
    s_te = set(data["subjects"]["test"])
    disjoint = (len(s_tr & s_va) == 0) and (len(s_tr & s_te) == 0) and (len(s_va & s_te) == 0)
    print("Zero volunteer overlap across splits:", disjoint)

    os.makedirs("results", exist_ok=True)
    with open("results/channel_stats.csv", "w", encoding="utf-8") as f:
        f.write("channel,mean,std\n")
        for idx, ch in enumerate(CHANNELS):
            f.write(f"{ch},{data['mean'][idx]:.6f},{data['std'][idx]:.6f}\n")

    print("\n=== Training Channel Standardisation Statistics ===")
    for idx, ch in enumerate(CHANNELS):
        print(f"  {ch:<14s}  mean = {data['mean'][idx]: .4f}   std = {data['std'][idx]: .4f}")
    print("Saved results/channel_stats.csv")
