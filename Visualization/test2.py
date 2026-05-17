import torch
from torch.utils.data import DataLoader, TensorDataset
from get_dataset_10label import *  # 假设你的数据加载脚本支持任意 n_classes
from torch.nn import functional as F
import numpy as np

def test(netA, netC, test_dataloader, device):
    netA.eval()
    netC.eval()
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            target = target.to(device).long()

            # AutoEncoder 提取特征
            features, _ = netA(data)  # output_of_netA[0] 是 features

            # Classifier 预测
            logits = netC(features)
            pred = logits.argmax(dim=1)

            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # 高效收集预测和真实标签（用于后续混淆矩阵等）
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = 100.0 * correct / total
    print(f'\nTest set: Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')

    return np.array(all_preds), np.array(all_labels), accuracy

def TestDataset_prepared(n_classes, rand_num):
    X_test, Y_test = TestDataset(n_classes)

    # 直接用测试集自身 min/max 归一化（推荐！简单可靠）
    min_value = X_test.min()
    max_value = X_test.max()
    X_test = (X_test - min_value) / (max_value - min_value + 1e-8)

    print(f"Test set normalized with own min/max: {min_value:.4f} ~ {max_value:.4f}")

    return X_test, Y_test

def main():
    rand_num = 30
    n_classes = 16  # 你的模型是 16 类

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据准备
    X_test, Y_test = TestDataset_prepared(n_classes, rand_num)

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=128,      # 加大 batch，测试更快
        shuffle=False,       # 测试不需要 shuffle
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 加载模型（关键：指定 map_location 并移到 device）
    netA_path = "model_weight/netA_n_classes_10_label10_unlabel90_rand30.pth"  # 你说实际是16类权重
    netC_path = "model_weight/netC_n_classes_10_label10_unlabel90_rand30.pth"

    print("Loading models...")
    netA = torch.load(netA_path, map_location=device)
    netC = torch.load(netC_path, map_location=device)

    netA = netA.to(device)
    netC = netC.to(device)

    print("Models loaded successfully.")

    # 测试
    preds, labels, acc = test(netA, netC, test_dataloader, device)

    # 可选：打印混淆矩阵或保存结果
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:\n", cm)

if __name__ == '__main__':
    main()