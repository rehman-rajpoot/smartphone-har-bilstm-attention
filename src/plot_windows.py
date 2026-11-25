import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load_data import load_raw, CHANNELS

FS = 50  # 50 Hz sampling rate (128 samples = 2.56 seconds)


def plot_one_per_activity(split, names, out_path="results/example_windows.png"):
    """Plot the first window of each activity: body acceleration on top, gyroscope below."""
    t = np.arange(128, dtype=np.float32) / FS
    fig, axes = plt.subplots(2, 6, figsize=(20, 5.2), sharex=True)

    for k in range(len(names)):
        name = names[k]
        # Find first window index belonging to activity label (k + 1)
        first_idx = 0
        for idx in range(len(split["y"])):
            if int(split["y"][idx]) == (k + 1):
                first_idx = idx
                break

        for c in range(3):
            axes[0, k].plot(t, split["X"][first_idx, :, c], label=CHANNELS[c], linewidth=1.1)
            axes[1, k].plot(t, split["X"][first_idx, :, 3 + c], label=CHANNELS[3 + c], linewidth=1.1)

        axes[0, k].set_title(name, fontsize=9, fontweight="bold")
        axes[1, k].set_xlabel("time (s)")
        axes[0, k].grid(alpha=0.25)
        axes[1, k].grid(alpha=0.25)

    axes[0, 0].set_ylabel("body acc (g)")
    axes[1, 0].set_ylabel("gyro (rad/s)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    axes[1, 0].legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    raw = load_raw()
    plot_one_per_activity(raw["train"], raw["activity_names"])
