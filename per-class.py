import torch
from torch.utils.data import DataLoader, TensorDataset
from get_dataset_10label import *
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import os

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# === 修复中文显示（根据你的环境选一种） ===
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 黑体
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 微软雅黑
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] # Linux / 跨平台
plt.rcParams['axes.unicode_minus'] = False


def plot_open_set_evaluation(mahal_distances, binary_labels, known_classes=10, save_dir="figures"):
    """
    绘制开集性能评估图：
    1. ROC 曲线 + AUROC
    2. PR 曲线 + AUPRC
    3. Mahalanobis 距离分布（known vs unknown）
    """
    os.makedirs(save_dir, exist_ok=True)

    # binary_labels: unknown=1, known=0
    unknown_mask = (binary_labels == 1)
    known_mask = (binary_labels == 0)

    scores_known = mahal_distances[known_mask]
    scores_unknown = mahal_distances[unknown_mask]

    print(f"Known samples: {len(scores_known)}, Unknown samples: {len(scores_unknown)}")

    # 1. ROC 曲线
    fpr, tpr, _ = roc_curve(binary_labels, mahal_distances)
    auroc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR / Recall)')
    plt.title(f'ROC - Open-Set Unknown Detection\n(Known: 0~{known_classes - 1}, Unknown: {known_classes}~15)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.show()
    plt.close()

    # 找到 FPR@95%TPR
    idx_95 = np.where(tpr >= 0.95)[0]
    if len(idx_95) > 0:
        fpr95 = fpr[idx_95[0]]
        print(f"FPR @ 95% TPR: {fpr95:.4f}")

    # 2. Precision-Recall 曲线
    precision, recall, _ = precision_recall_curve(binary_labels, mahal_distances)
    auprc = auc(recall, precision)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'PR curve (AUPRC = {auprc:.4f})')
    plt.xlabel('Recall (TPR)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall - Open-Set Detection\n(Known: 0~{known_classes - 1})')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=300)
    plt.show()
    plt.close()

    # 3. 分数（Mahalanobis 距离）分布直方图
    plt.figure(figsize=(9, 5))

    plt.hist(scores_known, bins=60, alpha=0.6, color='blue', label='Known classes', density=True)
    plt.hist(scores_unknown, bins=60, alpha=0.6, color='red', label='Unknown classes', density=True)

    plt.xlabel('Mahalanobis Distance (min to known class means)')
    plt.ylabel('Density')
    plt.title('Mahalanobis Distance Distribution\n(Known vs Unknown)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "distance_distribution.png"), dpi=300)
    plt.show()
    plt.close()


def test(netA, test_dataloader, device, known_classes=10):
    netA.eval()
    correct_known = 0
    total_known = 0

    all_preds = []
    all_mahal_distances = []  # Mahalanobis 距离（越大越可能是 unknown）
    all_labels = []  # 原始标签 0~15

    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            target = target.to(device).long()

            features, _ = netA(data)  # [B, 128]

            # 计算 Mahalanobis 距离（使用 per-class precision）
            diffs = features.unsqueeze(1) - class_means.unsqueeze(0)  # [B, 10, 128]
            mahal_per_class = []
            for i in range(known_classes):
                diff = diffs[:, i, :]  # [B, 128]
                mahal_i = torch.sqrt(torch.sum(diff @ class_precisions[i] * diff, dim=-1))  # [B]
                mahal_per_class.append(mahal_i)
            mahal = torch.stack(mahal_per_class, dim=1)  # [B, 10]
            min_mahal = mahal.min(dim=1)[0]  # [B]

            # Known 类统计（仅用于打印准确率，可选）
            known_mask = (target < known_classes)
            # 如果你有 netC，可以用 pred = netC(features).argmax(1)
            # 这里假设不使用分类器，仅统计样本数
            total_known += known_mask.sum().item()

            all_mahal_distances.append(min_mahal.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    # 合并
    mahal_distances = np.concatenate(all_mahal_distances)
    labels = np.concatenate(all_labels)

    # Known 类准确率（如果没有 netC，这里无法计算分类准确率，只统计样本数）
    # 如果你想加 netC 预测 known 类准确率，请告诉我，我帮你加

    # Unknown 检测指标
    binary_labels = (labels >= known_classes).astype(int)  # unknown = 1
    auroc = roc_auc_score(binary_labels, mahal_distances)  # 距离越大越 unknown

    fpr, tpr, _ = roc_curve(binary_labels, mahal_distances)
    fpr95_idx = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[fpr95_idx[0]] if len(fpr95_idx) > 0 else 1.0

    print('\n' + '=' * 60)
    print("【开集识别测试结果 - Mahalanobis Distance (Per-Class Covariance)】 Known classes: 0 ~ {}".format(known_classes - 1))
    print(f"Known 样本数: {total_known}, Unknown 样本数: {len(labels) - total_known}")
    print(f"Unknown 检测 AUROC (Mahalanobis): {auroc:.4f}")
    print(f"FPR@95%TPR (Mahalanobis): {fpr95:.4f}")
    print('=' * 60)

    return mahal_distances, binary_labels, auroc, fpr95


# 计算 known 类统计（类均值 + per-class 协方差）
def compute_mahalanobis_stats(netA, val_dataloader, device, known_classes=10):
    netA.eval()
    features_per_class = [[] for _ in range(known_classes)]

    with torch.no_grad():
        for data, target in val_dataloader:
            data = data.to(device)
            target = target.to(device).long()

            features, _ = netA(data)

            for i in range(known_classes):
                mask = (target == i)
                if mask.any():
                    features_per_class[i].append(features[mask])

    # 计算每类均值和协方差
    class_means = []
    class_precisions = []  # 每个类的逆协方差矩阵
    for i in range(known_classes):
        feats = torch.cat(features_per_class[i])
        mean = feats.mean(dim=0)
        class_means.append(mean)

        # 计算 per-class 协方差（加正则防奇异）
        cov = torch.cov(feats.T) + 1e-6 * torch.eye(feats.shape[1], device=device)
        precision = torch.inverse(cov)
        class_precisions.append(precision)

    class_means = torch.stack(class_means)  # [10, 128]

    return class_means, class_precisions


def Data_prepared(n_classes, rand_num):
    X_train_labeled, X_train_unlabeled, X_train, X_val, Y_train_labeled, Y_train_unlabeled, Y_train, Y_val = TrainDataset(
        n_classes, rand_num)

    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value


def TestDataset_prepared(n_classes, rand_num):
    X_test, Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes, rand_num)

    X_test = (X_test - min_value) / (max_value - min_value)

    return X_test, Y_test


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rand_num = 30
    known_classes = 10

    # 加载 test 数据 (全部 16 类)
    X_test, Y_test = TestDataset_prepared(16, rand_num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 加载 val 数据 (只 known 10 类，用于计算 Mahalanobis 统计)
    X_val_known, Y_val_known = TestDataset(known_classes)  # 或者用 ValDataset
    # 归一化用同样的 min/max
    max_value, min_value = Data_prepared(known_classes, rand_num)
    X_val_known = (X_val_known - min_value) / (max_value - min_value + 1e-8)
    val_dataset = TensorDataset(torch.Tensor(X_val_known), torch.Tensor(Y_val_known))
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 加载模型
    netA_path = "model_weight/netA_n_classes_10_label10_unlabel90_rand30.pth"  # 你的 10 类 netA
    # netC_path = "model_weight/netC_10class_best.pth"  # 开集测试可以不加载 netC

    print("Loading netA...")
    netA = torch.load(netA_path, map_location=device)
    netA = netA.to(device)

    # 计算 Mahalanobis 统计量（per-class）
    print("Computing Mahalanobis statistics from known val data...")
    global class_means, class_precisions
    class_means, class_precisions = compute_mahalanobis_stats(netA, val_dataloader, device, known_classes)

    # 开集测试（只用 netA 特征 + Mahalanobis）
    print("Running open-set test...")
    mahal_distances, binary_labels, auroc, fpr95 = test(netA, test_dataloader, device, known_classes=known_classes)

    # 绘制评估图
    print("\n生成开集性能可视化图表...")
    plot_open_set_evaluation(mahal_distances, binary_labels, known_classes=known_classes)


if __name__ == '__main__':
    main()