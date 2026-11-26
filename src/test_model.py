import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from load_data import load_raw, CHANNELS
from prepare import prepare
from model import HARNet
from metrics import accuracy, confusion_matrix, per_class_accuracy

FS = 50
SHORT_NAMES = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]


@torch.no_grad()
def predict_test_windows(model, X_test, batch_size=256):
    """Run trained HARNet once across all test windows, returning predictions, probabilities, and alpha."""
    model.eval()
    preds_list = []
    probs_list = []
    alphas_list = []

    n_total = X_test.shape[0]
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        xb = X_test[start:end]
        logits, alpha = model(xb)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        preds_list.append(preds.cpu())
        probs_list.append(probs.cpu())
        alphas_list.append(alpha.cpu())

    return torch.cat(preds_list, dim=0), torch.cat(probs_list, dim=0), torch.cat(alphas_list, dim=0)


def save_confusion_heatmap(C_tensor, out_path="results/confusion_matrix.png"):
    """Plot and save the 6x6 test confusion matrix with activity names on both axes."""
    C_np = C_tensor.numpy()
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(C_np, cmap="Blues")

    ax.set_xticks(range(6))
    ax.set_xticklabels(SHORT_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(6))
    ax.set_yticklabels(SHORT_NAMES)
    ax.set_xlabel("Predicted Activity")
    ax.set_ylabel("True Activity")
    ax.set_title("Test Set Confusion Matrix (2,947 Windows)")

    threshold = float(C_np.max()) / 2.0
    for j in range(6):
        for k in range(6):
            text_color = "white" if C_np[j, k] > threshold else "black"
            ax.text(k, j, str(int(C_np[j, k])), ha="center", va="center", color=text_color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


def analyse_and_plot_attention(raw_test_X, names, y_true_np, y_pred_np, alpha_np):
    """Generate attention example windows plot, average attention curves, and peakiness statistics."""
    t = np.arange(128, dtype=np.float32) / FS
    flat_ref = 1.0 / 128.0

    # 1. One correctly classified test window per activity: body_acc on top, alpha_t underneath
    fig, axes = plt.subplots(2, 6, figsize=(20, 5.4), sharex=True)
    for k in range(6):
        chosen_idx = 0
        for idx in range(len(y_true_np)):
            if int(y_true_np[idx]) == k and int(y_pred_np[idx]) == k:
                chosen_idx = idx
                break

        for c in range(3):
            axes[0, k].plot(t, raw_test_X[chosen_idx, :, c], label=CHANNELS[c], linewidth=1.1)
        axes[0, k].set_title(names[k], fontsize=9, fontweight="bold")
        axes[0, k].grid(alpha=0.25)

        axes[1, k].plot(t, alpha_np[chosen_idx], color="black", linewidth=1.3, label="alpha_t")
        axes[1, k].axhline(flat_ref, color="crimson", linestyle="--", linewidth=0.9, label="1/128")
        axes[1, k].set_xlabel("time (s)")
        axes[1, k].grid(alpha=0.25)

    axes[0, 0].set_ylabel("body acc (g)")
    axes[1, 0].set_ylabel("attention weight alpha_t")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    axes[1, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig("results/attention_examples.png", dpi=140)
    plt.close(fig)
    print("Saved results/attention_examples.png")

    # 2. Average attention curve over all test windows of each activity
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for k in range(6):
        mask = (y_true_np == k)
        mean_curve = np.mean(alpha_np[mask], axis=0)
        ax.plot(t, mean_curve, linewidth=1.5, label=names[k])

    ax.axhline(flat_ref, color="dimgray", linestyle="--", linewidth=1.0, label="Uniform (1/128 = 0.0078)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Mean attention weight alpha_t")
    ax.set_title("Average Temporal Attention Profile per Activity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig("results/attention_average.png", dpi=140)
    plt.close(fig)
    print("Saved results/attention_average.png")

    # 3. Mean of max_t alpha_t (peakiness) per activity
    peakiness = []
    print(f"\n=== Attention Peakiness: Mean of max_t(alpha_t) [Flat = {flat_ref:.4f}] ===")
    for k in range(6):
        mask = (y_true_np == k)
        class_alphas = alpha_np[mask]
        max_per_window = np.max(class_alphas, axis=1)
        mean_max = float(np.mean(max_per_window))
        peakiness.append(mean_max)
        print(f"  {names[k]:<20s}  {mean_max:.4f}  ({mean_max / flat_ref:5.1f}x flat)")

    with open("results/attention_peakiness.csv", "w", encoding="utf-8") as f:
        f.write(",".join(names) + "\n")
        f.write(",".join(f"{val:.6f}" for val in peakiness) + "\n")
    print("Saved results/attention_peakiness.csv")
    return peakiness


def write_results_tables_md(data, test_acc, per_cls_acc, C_tensor, peakiness):
    """Write formatted Results Tables 1, 2, and 3 into results/tables.md."""
    lines = ["# Results Tables\n"]

    lines.append("## Table 1 - Data\n")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append(
        f"| Train / validation / test windows | "
        f"{len(data['train'][1])} / {len(data['val'][1])} / {len(data['test'][1])} |"
    )
    lines.append(
        f"| Train / validation / test volunteers | "
        f"{len(data['subjects']['train'])} / {len(data['subjects']['val'])} / {len(data['subjects']['test'])} |\n"
    )
    lines.append("Mean and standard deviation of each channel (training windows only):\n")
    lines.append("| Channel | Mean | Std |")
    lines.append("|---|---|---|")
    for idx, ch in enumerate(CHANNELS):
        lines.append(f"| {ch} | {data['mean'][idx]:.4f} | {data['std'][idx]:.4f} |")

    if os.path.exists("results/history.json"):
        with open("results/history.json", "r", encoding="utf-8") as f:
            summary = json.load(f)["summary"]
        lines.append("\n## Table 2 - Network and training\n")
        lines.append("| Item | Value |")
        lines.append("|---|---|")
        lines.append(f"| Trainable parameters | {summary['parameters']} |")
        lines.append(f"| Initial loss (before training) | {summary['initial_loss']:.4f} |")
        lines.append(f"| Best epoch (lowest validation loss) | {summary['best_epoch']} |")
        lines.append(
            f"| Training / validation loss at best epoch | "
            f"{summary['train_loss_at_best']:.4f} / {summary['val_loss_at_best']:.4f} |"
        )
        lines.append(f"| Validation accuracy at best epoch | {summary['val_acc_at_best']:.4f} |")
        lines.append(
            f"| Training time and hardware | "
            f"{summary['training_minutes']:.1f} min, {summary['hardware']} |"
        )

    lines.append("\n## Table 3 - Test results\n")
    lines.append("| Metric | " + " | ".join(SHORT_NAMES) + " |")
    lines.append("|---|" + "---|" * 6)
    lines.append("| Per-class accuracy | " + " | ".join(f"{float(v):.4f}" for v in per_cls_acc) + " |")
    lines.append("| Mean of max_t alpha_t | " + " | ".join(f"{v:.4f}" for v in peakiness) + " |\n")
    lines.append(f"**Overall Test Accuracy: {test_acc:.4f}** ({test_acc * 100.0:.2f}%)\n")

    lines.append("### Full 6 x 6 Test Confusion Matrix (rows = true activity, columns = predicted)\n")
    lines.append("| True \\ Predicted | " + " | ".join(SHORT_NAMES) + " |")
    lines.append("|---|" + "---|" * 6)
    for j in range(6):
        row_str = " | ".join(str(int(C_tensor[j, k].item())) for k in range(6))
        lines.append(f"| {data['names'][j]} | {row_str} |")

    with open("results/tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved results/tables.md")


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = load_raw()
    data = prepare()
    names = data["names"]
    X_te, y_te = data["test"]

    model = HARNet(input_size=9, hidden_size=64, num_classes=6).to(device)
    state_dict = torch.load("results/best_model.pt", map_location=device)
    model.load_state_dict(state_dict)

    y_pred, probs, alpha = predict_test_windows(model, X_te.to(device))

    # Verify attention weights are positive and sum to 1 for every window
    alpha_sums = alpha.sum(dim=1)
    print(
        f"Attention check -> all positive: {bool((alpha > 0).all().item())}, "
        f"max |sum - 1|: {float((alpha_sums - 1.0).abs().max().item()):.2e}"
    )

    test_acc = accuracy(y_te, y_pred)
    C = confusion_matrix(y_te, y_pred, num_classes=6)
    per_cls = per_class_accuracy(C)

    print(f"\n=== Test Set Evaluation (2,947 Unseen Windows) ===")
    print(f"Overall Test Accuracy: {test_acc:.4f} ({test_acc * 100.0:.2f}%)")
    print("Per-Class Accuracy:")
    for idx in range(6):
        print(f"  {names[idx]:<20s}  {float(per_cls[idx].item()):.4f}")

    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(C)
    print(f"Total entries in C: {int(C.sum().item())} (expected 2947)")
    print(f"Row sums: {C.sum(dim=1).tolist()} (expected [496, 471, 420, 491, 532, 537])")

    torch.save(
        {"y_true": y_te, "y_pred": y_pred, "probs": probs, "alpha": alpha},
        "results/test_outputs.pt",
    )

    with open("results/confusion_matrix.csv", "w", encoding="utf-8") as f:
        f.write(",".join(names) + "\n")
        for j in range(6):
            f.write(",".join(str(int(C[j, k].item())) for k in range(6)) + "\n")

    save_confusion_heatmap(C)
    peakiness = analyse_and_plot_attention(
        raw["test"]["X"], names, y_te.numpy(), y_pred.numpy(), alpha.numpy()
    )
    write_results_tables_md(data, test_acc, per_cls, C, peakiness)


if __name__ == "__main__":
    main()
