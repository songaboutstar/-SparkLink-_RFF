import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
def get_known_predictions(netA, netC, dataloader, device, known_classes=10):
    netA.eval()
    netC.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).long()

            # 仅保留 Known 样本
            known_mask = (y < known_classes)
            if known_mask.sum() == 0:
                continue

            x_known = x[known_mask]
            y_known = y[known_mask]

            # 特征 + 分类
            feat, _ = netA(x_known)
            logits = netC(feat)
            pred = logits.argmax(dim=1)

            y_true.append(y_known.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return y_true, y_pred
def plot_known_confusion_matrix(
    y_true,
    y_pred,
    known_classes,
    save_path,
    normalize=True
):
    """
    normalize=True → 显示百分比（论文推荐）
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(known_classes))
    )

    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(7.5, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=range(known_classes),
        yticklabels=range(known_classes),
        cbar=True
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.title("Confusion Matrix on Known Classes")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.close()
