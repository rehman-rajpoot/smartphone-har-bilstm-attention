from pathlib import Path
import numpy as np

# Raw inertial signal file prefixes in the fixed 9-channel order
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _find_root(data_dir):
    """Locate the unzipped UCI HAR dataset folder containing train/ and test/."""
    base = Path(data_dir)
    for folder in sorted(base.rglob("Inertial Signals")):
        if folder.parent.name == "train":
            return folder.parent.parent
    raise FileNotFoundError(
        f"Could not find 'train/Inertial Signals' inside '{data_dir}'. "
        "Please unzip the UCI HAR archive (and any nested zip) into data/."
    )


def load_raw(data_dir="data"):
    """Load raw 128-step x 9-channel inertial windows, labels (1..6), and subjects."""
    root = _find_root(data_dir)
    dataset = {}

    for split in ["train", "test"]:
        sig_dir = root / split / "Inertial Signals"
        channel_arrays = []
        for ch_name in CHANNELS:
            file_path = sig_dir / f"{ch_name}_{split}.txt"
            arr = np.loadtxt(file_path, dtype=np.float32)
            channel_arrays.append(arr)

        X = np.stack(channel_arrays, axis=-1)
        y = np.loadtxt(root / split / f"y_{split}.txt", dtype=int)
        subject = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
        dataset[split] = {"X": X, "y": y, "subject": subject}

    label_lines = (root / "activity_labels.txt").read_text().splitlines()
    activity_names = []
    for line in label_lines:
        if line.strip():
            activity_names.append(line.split()[1])

    dataset["activity_names"] = activity_names
    return dataset


if __name__ == "__main__":
    raw = load_raw()
    for split_name in ["train", "test"]:
        part = raw[split_name]
        unique_subjects = sorted(list(set(part["subject"].tolist())))
        print(
            f"{split_name}: X {part['X'].shape}, "
            f"y {part['y'].shape}, "
            f"subject {part['subject'].shape}, "
            f"volunteers ({len(unique_subjects)}): {unique_subjects}"
        )
    print("activities:", raw["activity_names"])
