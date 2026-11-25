import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import BiLSTM, Attention


class HARNet(nn.Module):
    """Full activity recognition network: BiLSTM(9 -> 2x64) -> Attention -> Linear(128, 6)."""

    def __init__(self, input_size=9, hidden_size=64, num_classes=6):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.bilstm = BiLSTM(input_size, hidden_size)
        self.attention = Attention(2 * hidden_size, att_size=64)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

        bound = 1.0 / (hidden_size ** 0.5)
        with torch.no_grad():
            self.fc.weight.uniform_(-bound, bound)
            self.fc.bias.uniform_(-bound, bound)

    def forward(self, x):
        # x: (B, 128, 9)
        seq_features = self.bilstm(x)                 # (B, 128, 128)
        context, alpha = self.attention(seq_features)  # (B, 128), (B, 128)
        logits = self.fc(context)                     # (B, 6)
        return logits, alpha


def count_parameters(model):
    """Sum the number of trainable scalar parameters across the model."""
    total = 0
    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()
    return total


def print_hand_parameter_calculation():
    """Display the exact analytical parameter breakdown for D=9, H=64, A=64, C=6."""
    D, H, A, C = 9, 64, 64, 6
    lstm_w = (4 * H) * D
    lstm_u = (4 * H) * H
    lstm_b = 4 * H
    one_lstm = lstm_w + lstm_u + lstm_b
    bilstm_total = 2 * one_lstm

    att_wa = A * (2 * H)
    att_ba = A
    att_v = A
    att_total = att_wa + att_ba + att_v

    fc_w = C * (2 * H)
    fc_b = C
    fc_total = fc_w + fc_b

    grand_total = bilstm_total + att_total + fc_total

    print("=== Hand Calculation of Trainable Parameters ===")
    print(f"1. Forward LSTMCell  : W({4*H}x{D}={lstm_w}) + U({4*H}x{H}={lstm_u}) + b({4*H}={lstm_b}) = {one_lstm}")
    print(f"2. Backward LSTMCell : W({4*H}x{D}={lstm_w}) + U({4*H}x{H}={lstm_u}) + b({4*H}={lstm_b}) = {one_lstm}")
    print(f"   -> BiLSTM Subtotal: 2 x {one_lstm} = {bilstm_total}")
    print(f"3. Attention Layer   : Wa({A}x{2*H}={att_wa}) + ba({att_ba}) + v({att_v}) = {att_total}")
    print(f"4. Classifier Linear : weight({C}x{2*H}={fc_w}) + bias({fc_b}) = {fc_total}")
    print(f"------------------------------------------------------------")
    print(f"   Hand-Calculated Total = {bilstm_total} + {att_total} + {fc_total} = {grand_total}")


if __name__ == "__main__":
    torch.manual_seed(0)
    net = HARNet(input_size=9, hidden_size=64, num_classes=6)

    counted = count_parameters(net)
    print_hand_parameter_calculation()
    print(f"   PyTorch Counted Total = {counted}")

    # Sanity check initial loss ~ log(6) = 1.7918 and attention normalization
    sample_x = torch.randn(16, 128, 9)
    sample_y = torch.randint(0, 6, (16,))
    with torch.no_grad():
        logits, alpha = net(sample_x)
        init_loss = F.cross_entropy(logits, sample_y).item()
    print(f"\nSanity Check -> logits: {tuple(logits.shape)}, alpha: {tuple(alpha.shape)}")
    print(f"Untrained random-batch cross-entropy: {init_loss:.4f} (ln(6) = 1.7918)")
    print(f"Attention min={alpha.min().item():.6f}, max={alpha.max().item():.6f}, row_sum={alpha.sum(dim=1).mean().item():.6f}")
