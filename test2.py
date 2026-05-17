import torch
from torch.utils.data import DataLoader, TensorDataset
from get_dataset_10label import *
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import os

def test(netA, netC, test_dataloader, device, known_classes=10):
    netA.eval()
    netC.eval()
    correct_known = 0
    total_known = 0

    all_preds = []        # known 类预测结果
    all_energies = []     # Energy score，用于 unknown 检测
    all_labels = []       # 原始真实标签 (0~15)

    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            data = data.to(device)
            target = target.to(device)

            output_of_netA = netA(data)
            features = output_of_netA[0]

            output_of_netC = netC(features)
            logits = output_of_netC  # [B, n_classes]

            # 预测类别
            pred = logits.argmax(dim=1)

            # Energy Score：值越低越可能是 unknown
            energy = -torch.logsumexp(logits, dim=1)

            # Known 类准确率统计
            known_mask = (target < known_classes)
            correct_known += pred[known_mask].eq(target[known_mask]).sum().item()
            total_known += known_mask.sum().item()

            # 收集用于后续指标计算
            all_preds.append(pred.cpu().numpy())
            all_energies.append(energy.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    # 合并结果
    preds = np.concatenate(all_preds)
    energies = np.concatenate(all_energies)
    labels = np.concatenate(all_labels)

    # Known 类准确率
    known_acc = 100.0 * correct_known / total_known if total_known > 0 else 0.0

    # Unknown 检测指标
    binary_labels = (labels >= known_classes).astype(int)  # unknown = 1, known = 0
    auroc = roc_auc_score(binary_labels, -energies)  # energy 越低越 unknown，取负

    fpr, tpr, _ = roc_curve(binary_labels, -energies)
    fpr95_idx = np.where(tpr >= 0.95)[0]
    fpr95 = fpr[fpr95_idx[0]] if len(fpr95_idx) > 0 else 1.0

    print('\n' + '=' * 60)
    print("【开集识别测试结果】 Known classes: 0 ~ {}".format(known_classes - 1))
    print(f"Known 类准确率: {known_acc:.2f}%")
    print(f"Unknown 检测 AUROC (Energy Score): {auroc:.4f}")
    print(f"FPR@95%TPR (Energy): {fpr95:.4f}")
    print('=' * 60)

def Data_prepared(n_classes, rand_num):
    X_train_labeled, X_train_unlabeled, X_train, X_val, Y_train_labeled, Y_train_unlabeled, Y_train, Y_val = TrainDataset(n_classes, rand_num)

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

    # 保持你原来的注释状态（不 transpose）
    # X_test = X_test.transpose(0, 2, 1)

    return X_test, Y_test

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rand_num = 30
    n_classes_test = 16  # 测试时加载全部 16 类数据

    X_test, Y_test = TestDataset_prepared(n_classes_test, rand_num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # 加载你的 10 类 known 模型权重（请确保这是 10 类训练的模型）
    netA_path = "model_weight/netA_n_classes_10_label10_unlabel90_rand30.pth"   # 替换为你的 10 类 netA 路径
    netC_path = "model_weight/netC_n_classes_10_label10_unlabel90_rand30.pth"   # 替换为你的 10 类 netC 路径

    print("Loading models...")
    netA = torch.load(netA_path, map_location=device)
    netC = torch.load(netC_path, map_location=device)

    netA = netA.to(device)
    netC = netC.to(device)

    print("Models loaded and moved to device successfully.")

    # 开集测试（known_classes=10）
    test(netA, netC, test_dataloader, device, known_classes=10)

if __name__ == '__main__':
    main()