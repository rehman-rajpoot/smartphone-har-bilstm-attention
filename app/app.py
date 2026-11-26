import io
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from load_data import load_raw, CHANNELS
from prepare import prepare
from model import HARNet

FS = 50  # 50 Hz -> 128 samples = 2.56 seconds
DISPLAY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]
SHORT_LABELS = ["Walk", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]


@st.cache_resource
def load_pipeline():
    """Load raw test windows, training standardisation stats, and trained HARNet weights."""
    data_dir = ROOT_DIR / "data"
    ckpt_path = ROOT_DIR / "results" / "best_model.pt"

    raw = load_raw(str(data_dir))
    prep = prepare(str(data_dir))

    model = HARNet(input_size=9, hidden_size=64, num_classes=6)
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "raw_test_X": raw["test"]["X"],          # (2947, 128, 9) unscaled physical units
        "raw_test_subj": raw["test"]["subject"], # (2947,) volunteer IDs
        "y_test": prep["test"][1].numpy(),       # (2947,) labels 0..5
        "mean": prep["mean"],                    # (9,) training channel mean
        "std": prep["std"],                      # (9,) training channel std
        "names": prep["names"],
        "model": model,
    }


def run_single_window(model, raw_window_128x9, mean, std):
    """Standardise a single (128, 9) raw window using training stats and run HARNet."""
    scaled = np.empty((128, 9), dtype=np.float32)
    for c in range(9):
        scaled[:, c] = (raw_window_128x9[:, c] - mean[c]) / std[c]

    x_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, 128, 9)
    with torch.no_grad():
        logits, alpha = model(x_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        att = alpha.squeeze(0).numpy()

    pred_idx = int(np.argmax(probs))
    return pred_idx, probs, att


def parse_uploaded_csv(uploaded_file):
    """Parse an uploaded CSV file into a float32 array of shape (128, 9)."""
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Uploaded CSV file is empty.")

    # Check if first line is a text header
    first_tokens = [tok.strip() for tok in lines[0].split(",")]
    start_row = 0
    try:
        float(first_tokens[0])
    except ValueError:
        start_row = 1

    rows = []
    for r_idx, line in enumerate(lines[start_row:], start=start_row + 1):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 9:
            raise ValueError(f"Row {r_idx} has {len(parts)} columns; expected 9 sensor channels.")
        rows.append([float(v) for v in parts])

    arr = np.array(rows, dtype=np.float32)
    if arr.shape != (128, 9):
        raise ValueError(f"CSV has shape {arr.shape}; expected (128, 9).")
    return arr


def build_signals_and_attention_figure(raw_win, att_weights, title_suffix=""):
    """Plot 3 sensor groups + temporal attention alpha_t on a shared 2.56s time axis."""
    t = np.arange(128, dtype=np.float32) / FS
    flat_level = 1.0 / 128.0
    peak_t = float(t[int(np.argmax(att_weights))])
    peak_val = float(np.max(att_weights))

    fig, axes = plt.subplots(
        4, 1, figsize=(8.2, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 1.15]}
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    groups = [
        (0, "Body Acc (g)", ["body_acc_x", "body_acc_y", "body_acc_z"]),
        (3, "Gyroscope (rad/s)", ["body_gyro_x", "body_gyro_y", "body_gyro_z"]),
        (6, "Total Acc (g)", ["total_acc_x", "total_acc_y", "total_acc_z"]),
    ]

    for g_idx, (offset, ylabel, labels) in enumerate(groups):
        ax = axes[g_idx]
        for c in range(3):
            ax.plot(t, raw_win[:, offset + c], color=colors[c], linewidth=1.25, label=labels[c])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.85)
        ax.grid(alpha=0.25)

    axes[0].set_title(f"2.56-Second Sensor Window & Temporal Attention {title_suffix}", fontsize=10.5, fontweight="bold")

    ax_att = axes[3]
    ax_att.plot(t, att_weights, color="#111111", linewidth=1.6, label="Attention alpha_t")
    ax_att.fill_between(t, 0.0, att_weights, color="#4a90e2", alpha=0.25)
    ax_att.axhline(flat_level, color="#d62728", linestyle="--", linewidth=1.0, label="Uniform (1/128 = 0.0078)")
    ax_att.scatter([peak_t], [peak_val], color="#d62728", zorder=5, s=28, label=f"Peak ({peak_val:.3f} @ {peak_t:.2f}s)")
    ax_att.set_ylabel("Attention alpha_t", fontsize=9)
    ax_att.set_xlabel("Time (seconds)", fontsize=9.5)
    ax_att.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.85)
    ax_att.grid(alpha=0.25)

    fig.tight_layout()
    return fig


def build_probability_bar_figure(probs, pred_idx):
    """Plot horizontal bar chart of all 6 activity softmax probabilities."""
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    y_pos = np.arange(len(SHORT_LABELS))
    bar_colors = ["#1f77b4" if i == pred_idx else "#b0c4de" for i in range(6)]

    bars = ax.barh(y_pos, probs * 100.0, color=bar_colors, edgecolor="#222222", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(SHORT_LABELS, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Predicted Probability (%)", fontsize=9.5)
    ax.set_title("Six-Activity Class Probabilities", fontsize=10.5, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)

    for idx, bar in enumerate(bars):
        pct = probs[idx] * 100.0
        ax.text(
            min(pct + 1.5, 101.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{pct:.2f}%",
            va="center",
            fontsize=8.5,
            fontweight="bold" if idx == pred_idx else "normal",
        )

    fig.tight_layout()
    return fig


def window_to_csv_bytes(raw_win):
    """Export a (128, 9) window to CSV bytes for convenient user download/testing."""
    buf = io.StringIO()
    buf.write(",".join(CHANNELS) + "\n")
    for r in range(128):
        buf.write(",".join(f"{raw_win[r, c]:.6f}" for c in range(9)) + "\n")
    return buf.getvalue().encode("utf-8")


def main():
    st.set_page_config(
        page_title="Smartphone HAR - Custom BiLSTM + Attention",
        layout="wide",
    )

    st.title("Recognizing Human Activity from Phone Motion Sensors")
    st.caption(
        "Custom PyTorch BiLSTM + Additive Temporal Attention Network (46,982 parameters) "
        "| 128 time steps @ 50 Hz (2.56 s) x 9 inertial channels"
    )

    bundle = load_pipeline()
    raw_X = bundle["raw_test_X"]
    y_test = bundle["y_test"]
    subjects = bundle["raw_test_subj"]
    names = bundle["names"]
    model = bundle["model"]

    st.sidebar.header("Window Selection")
    mode = st.sidebar.radio(
        "Input Source",
        options=["Pick a Test Set Window", "Upload CSV (128 x 9)"],
        index=0,
    )

    true_label_str = "Custom CSV Upload"
    subject_str = "N/A"
    win_tag = "(Uploaded CSV)"

    if mode == "Pick a Test Set Window":
        filter_choice = st.sidebar.selectbox(
            "Filter by True Activity",
            options=["ALL (2,947 windows)"] + names,
            index=1,
        )
        if filter_choice == "ALL (2,947 windows)":
            valid_indices = list(range(len(y_test)))
        else:
            target_cls = names.index(filter_choice)
            valid_indices = [i for i in range(len(y_test)) if int(y_test[i]) == target_cls]

        chosen_pos = st.sidebar.slider(
            f"Select Window ({len(valid_indices)} available)",
            min_value=0,
            max_value=len(valid_indices) - 1,
            value=0,
        )
        win_idx = valid_indices[chosen_pos]
        current_win = raw_X[win_idx]
        true_cls = int(y_test[win_idx])
        true_label_str = names[true_cls]
        subject_str = f"Volunteer #{int(subjects[win_idx])}"
        win_tag = f"(Test Window #{win_idx} | {subject_str})"

        st.sidebar.download_button(
            label="Download Current Window as CSV (128x9)",
            data=window_to_csv_bytes(current_win),
            file_name=f"test_window_{win_idx}_{true_label_str.lower()}.csv",
            mime="text/csv",
        )
    else:
        st.sidebar.markdown(
            "Upload a `.csv` file with **128 rows** and **9 columns** in standard channel order:\n"
            "`body_acc_x..z`, `body_gyro_x..z`, `total_acc_x..z`."
        )
        st.sidebar.download_button(
            label="Download Sample 128x9 CSV Template",
            data=window_to_csv_bytes(raw_X[0]),
            file_name="sample_har_window_128x9.csv",
            mime="text/csv",
        )
        uploaded = st.sidebar.file_uploader("Choose 128x9 CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a 128 x 9 CSV file in the sidebar (or download the sample CSV template first).")
            return
        try:
            current_win = parse_uploaded_csv(uploaded)
        except ValueError as err:
            st.error(f"Invalid CSV format: {err}")
            return

    pred_idx, probs, att_weights = run_single_window(
        model, current_win, bundle["mean"], bundle["std"]
    )
    pred_name = names[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0
    max_att = float(np.max(att_weights))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Activity", pred_name)
    c2.metric("Confidence", f"{confidence:.2f}%")
    c3.metric("True Activity", true_label_str)
    c4.metric("Peak Attention (max alpha_t)", f"{max_att:.4f}", f"{max_att / (1.0/128.0):.1f}x uniform")

    left_col, right_col = st.columns([1.45, 1.0])

    with left_col:
        fig_sig = build_signals_and_attention_figure(current_win, att_weights, title_suffix=win_tag)
        st.pyplot(fig_sig, width="stretch")
        plt.close(fig_sig)

    with right_col:
        fig_prob = build_probability_bar_figure(probs, pred_idx)
        st.pyplot(fig_prob, width="stretch")
        plt.close(fig_prob)

        st.markdown("#### Class Probability Breakdown")
        prob_rows = []
        for idx, nm in enumerate(names):
            prob_rows.append({
                "Activity": nm,
                "Probability (%)": f"{probs[idx] * 100.0:.2f}%",
            })
        st.table(prob_rows)


if __name__ == "__main__":
    main()
