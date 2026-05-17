import torch
from torch.utils.data import DataLoader, TensorDataset
from get_dataset_10label import *
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from test3 import TestDataset_prepared, Data_prepared
import matplotlib.pyplot as plt
import os
def plot_rmd_distribution(rmd_scores, labels, known_classes, save_path):
    """
    rmd_scores: np.ndarray, shape [N]
    labels: np.ndarray, 原始标签 0~15
    known_classes: 已知类别数，例如 10
    """

    # 分离 Known / Unknown
    known_scores = rmd_scores[labels < known_classes]
    unknown_scores = rmd_scores[labels >= known_classes]

    plt.figure(figsize=(7, 5))

    sns.kdeplot(
        known_scores,
        label="Known classes",
        linewidth=2.5,
        fill=True,
        alpha=0.35
    )

    sns.kdeplot(
        unknown_scores,
        label="Unknown classes",
        linewidth=2.5,
        fill=True,
        alpha=0.35
    )

    plt.xlabel("Relative Mahalanobis Distance (RMD)")
    plt.ylabel("Density")
    plt.title("RMD Score Distribution (Known vs Unknown)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_roc_curve(fpr, tpr, auroc, save_path):
    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr, tpr,
        linewidth=2.5,
        label=f"RMD (AUROC = {auroc:.4f})"
    )

    # 随机猜测参考线
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, label="Random Guess")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve for Unknown Device Detection", fontsize=13)

    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()


def test(netA, test_dataloader, device, known_classes=10):
    netA.eval()
    correct_known = 0
    total_known = 0

    all_mahal_scores = []  # Relative Mahalanobis score（越大越可能是 unknown）
    all_labels = []        # 原始标签 0~15

    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            target = target.to(device).long()

            features, _ = netA(data)  # [B, 128]

            # 计算 Relative Mahalanobis Distance
           # diffs = features - class_means  # [B, 10, 128]
            diffs = features.unsqueeze(1) - class_means.unsqueeze(0)
            mahal = torch.sum(diffs @ precision * diffs, dim=-1)  # [B, 10]
            min_mahal = mahal.min(dim=1)[0]  # 最小距离
            rmd_score = min_mahal - background_mahal  # Relative score

            # Known 类统计（仅样本数）
            known_mask = (target < known_classes)
            total_known += known_mask.sum().item()

            all_mahal_scores.append(rmd_score.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    # 合并
    rmd_scores = np.concatenate(all_mahal_scores)
    labels = np.concatenate(all_labels)

    # Known 类准确率（这里不计算分类准确率，只统计样本数；如需加 netC 告诉我）
    known_acc = "N/A (no classifier used)"

    # Unknown 检测指标
    binary_labels = (labels >= known_classes).astype(int)  # unknown = 1
    auroc = roc_auc_score(binary_labels, rmd_scores)  # RMD 越大越 unknown

    fpr, tpr, _ = roc_curve(binary_labels, rmd_scores)
    fpr95_idx = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[fpr95_idx[0]] if len(fpr95_idx) > 0 else 1.0
    # =========================
    # 绘制 ROC 曲线
    # =========================
    os.makedirs("Visualization", exist_ok=True)

    roc_save_path = "Visualization/RMD_ROC_curve.png"
    plot_roc_curve(fpr, tpr, auroc, roc_save_path)

    print(f"ROC curve saved to {roc_save_path}")
    # ===== RMD Distribution Plot =====
    rmd_dist_path = "Visualization/RMD_Distribution_Known_vs_Unknown.png"

    plot_rmd_distribution(
        rmd_scores,
        labels,
        known_classes=known_classes,
        save_path=rmd_dist_path
    )

    print(f"RMD distribution saved to {rmd_dist_path}")

    print('\n' + '=' * 60)
    print("【开集识别测试结果 - Relative Mahalanobis Distance】 Known classes: 0 ~ {}".format(known_classes - 1))
    print(f"Known 类准确率: {known_acc}")
    print(f"Unknown 检测 AUROC (RMD): {auroc:.4f}")
    print(f"FPR@95%TPR (RMD): {fpr95:.4f}")
    print('=' * 60)

# 计算 known 类均值 + 共享协方差 + background noise mahal
def compute_rmd_stats(netA, val_dataloader, device, known_classes=10):
    netA.eval()
    features_list = []

    with torch.no_grad():
        for data, _ in val_dataloader:
            data = data.to(device)
            features, _ = netA(data)
            features_list.append(features)

    all_features = torch.cat(features_list)  # [N_known, 128]
    class_means = []
    for i in range(known_classes):
        # 假设你有 val 标签，或用聚类近似；这里用全局均值简化
        class_means.append(all_features.mean(dim=0))
    class_means = torch.stack(class_means)

    # 共享协方差
    cov = torch.cov(all_features.T) + 1e-6 * torch.eye(all_features.shape[1], device=device)
    precision = torch.inverse(cov)

    # Background noise mahal (全局均值距离)
    global_mean = all_features.mean(dim=0)
    background_mahal = torch.sum((all_features - global_mean) @ precision * (all_features - global_mean), dim=-1).mean()

    return class_means, precision, background_mahal

# 保持你的 Data_prepared 和 TestDataset_prepared 不变
# ...

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rand_num = 30
    known_classes = 10

    # 加载 test 数据 (全部 16 类)
    X_test, Y_test = TestDataset_prepared(16, rand_num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 加载 known val 数据计算统计量
    X_val_known, Y_val_known = TestDataset(known_classes)  # 或 ValDataset
    max_value, min_value = Data_prepared(known_classes, rand_num)
    X_val_known = (X_val_known - min_value) / (max_value - min_value + 1e-8)
    val_dataset = TensorDataset(torch.Tensor(X_val_known), torch.Tensor(Y_val_known))
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 加载模型
    netA_path = "model_weight/netA_n_classes_10_label10_unlabel90_rand30.pth"   # 你的 10 类 netA
    print("Loading netA...")
    netA = torch.load(netA_path, map_location=device)
    netA = netA.to(device)

    # 计算 RMD 统计量
    print("Computing Relative Mahalanobis statistics...")
    global class_means, precision, background_mahal
    class_means, precision, background_mahal = compute_rmd_stats(netA, val_dataloader, device, known_classes)

    # 开集测试
    test(netA, test_dataloader, device, known_classes=known_classes)


if __name__ == '__main__':
    main()
    #ok