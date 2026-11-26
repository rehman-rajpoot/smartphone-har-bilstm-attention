import torch


def accuracy(y_true, y_pred):
    """Overall test accuracy: (1 / m) * sum_i 1[y_hat^(i) == y^(i)]."""
    m = int(y_true.shape[0])
    if m == 0:
        return 0.0
    correct = 0
    for i in range(m):
        if int(y_pred[i].item()) == int(y_true[i].item()):
            correct += 1
    return float(correct) / float(m)


def confusion_matrix(y_true, y_pred, num_classes=6):
    """6x6 confusion matrix where C[j, k] counts true class j predicted as class k."""
    C = torch.zeros((num_classes, num_classes), dtype=torch.long)
    m = int(y_true.shape[0])
    for i in range(m):
        true_cls = int(y_true[i].item())
        pred_cls = int(y_pred[i].item())
        C[true_cls, pred_cls] += 1
    return C


def per_class_accuracy(C):
    """Per-class recall C[j, j] / sum_k C[j, k] for each activity j."""
    num_classes = C.shape[0]
    rates = torch.zeros(num_classes, dtype=torch.float32)
    for j in range(num_classes):
        row_total = 0
        for k in range(num_classes):
            row_total += int(C[j, k].item())
        if row_total > 0:
            rates[j] = float(C[j, j].item()) / float(row_total)
    return rates


if __name__ == "__main__":
    yt = torch.tensor([0, 0, 1, 2, 2, 2, 3, 4, 5, 5])
    yp = torch.tensor([0, 1, 1, 2, 2, 0, 3, 3, 5, 5])
    cm = confusion_matrix(yt, yp, num_classes=6)
    print("Sample accuracy:", accuracy(yt, yp))
    print("Sample confusion matrix:\n", cm)
    print("Sample per-class accuracy:", per_class_accuracy(cm).tolist())
