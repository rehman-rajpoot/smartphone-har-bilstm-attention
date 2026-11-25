import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from prepare import prepare
from model import HARNet, count_parameters

# Fixed training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 30
GRAD_CLIP_NORM = 1.0


@torch.no_grad()
def evaluate_split(model, X_data, y_data, batch_size=256):
    """Compute mean cross-entropy loss and classification accuracy over X_data."""
    model.eval()
    n_samples = X_data.shape[0]
    loss_sum = 0.0
    correct_count = 0

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        xb = X_data[start:end]
        yb = y_data[start:end]

        logits, _ = model(xb)
        batch_loss = F.cross_entropy(logits, yb, reduction="sum")
        loss_sum += float(batch_loss.item())

        preds = torch.argmax(logits, dim=1)
        correct_count += int(torch.sum(preds == yb).item())

    return loss_sum / float(n_samples), correct_count / float(n_samples)


def plot_training_history(history, best_epoch, out_path="results/training_curves.png"):
    """Plot training/validation cross-entropy loss and validation accuracy vs epoch."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    axes[0].plot(epochs, history["train_loss"], marker="o", markersize=3, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", markersize=3, label="Validation Loss")
    axes[0].axvline(best_epoch, color="black", linestyle="--", linewidth=1.0, label=f"Best Epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Cross-Entropy Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], color="darkgreen", marker="o", markersize=3, label="Validation Accuracy")
    axes[1].axvline(best_epoch, color="black", linestyle="--", linewidth=1.0, label=f"Best Epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy Across Epochs")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare()
    X_tr, y_tr = data["train"][0].to(device), data["train"][1].to(device)
    X_va, y_va = data["val"][0].to(device), data["val"][1].to(device)

    # Seed right before creating HARNet
    torch.manual_seed(0)
    model = HARNet(input_size=9, hidden_size=64, num_classes=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_params = count_parameters(model)
    init_loss, init_acc = evaluate_split(model, X_tr, y_tr)
    print(f"Device: {device} | Trainable parameters: {n_params}")
    print(f"Initial loss before training: {init_loss:.4f} (ln(6) = 1.7918), Initial train acc: {init_acc:.4f}")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    lowest_val_loss = float("inf")
    best_epoch = 0
    t_start = time.time()

    num_train = X_tr.shape[0]

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        # Shuffle training windows at the start of every epoch
        perm = torch.randperm(num_train, device=device)
        epoch_loss_sum = 0.0

        for start in range(0, num_train, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_train)
            batch_idx = perm[start:end]
            xb = X_tr[batch_idx]
            yb = y_tr[batch_idx]

            logits, _ = model(xb)
            loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss_sum += float(loss.item()) * int(batch_idx.shape[0])

        train_loss = epoch_loss_sum / float(num_train)
        val_loss, val_acc = evaluate_split(model, X_va, y_va)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), "results/best_model.pt")
            marker = "  <-- saved best"

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}{marker}"
        )

    elapsed_min = (time.time() - t_start) / 60.0
    hw_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    idx = best_epoch - 1

    summary = {
        "parameters": n_params,
        "initial_loss": init_loss,
        "best_epoch": best_epoch,
        "train_loss_at_best": history["train_loss"][idx],
        "val_loss_at_best": history["val_loss"][idx],
        "val_acc_at_best": history["val_acc"][idx],
        "training_minutes": elapsed_min,
        "hardware": hw_name,
    }

    with open("results/history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "summary": summary}, f, indent=2)

    plot_training_history(history, best_epoch)
    print("\nTraining Summary (Table 2):")
    for key, val in summary.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
