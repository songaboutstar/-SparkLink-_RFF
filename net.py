import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from get_dataset_10label import *
import torch

def test(netA, test_dataloader, save_fig_path="confusion_matrix.png"):
    """
    只使用 netA 进行分类预测，计算准确率和混淆矩阵
    假设 netA(data) 返回 logits（[B, num_classes]）或 (features, logits)
    """
    netA.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    y_true_list = []
    y_pred_list = []
    correct = 0

    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            target = target.to(device).long()

            # 只使用 netA
            output = netA(data)

            # 根据 netA 的实际输出格式处理
            # 情况1：netA 直接输出 logits [B, num_classes]
            if isinstance(output, torch.Tensor) and output.dim() == 2:
                logits = output
            # 情况2：netA 返回元组 (features, logits)
            elif isinstance(output, tuple) and len(output) >= 2:
                logits = output[1] if output[1].dim() == 2 else output[0]
            else:
                raise ValueError("netA 输出格式不支持，请检查模型 forward 返回值")

            pred = logits.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            y_pred_list.extend(pred.squeeze().cpu().numpy().tolist())
            y_true_list.extend(target.cpu().numpy().tolist())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    total = len(y_true)
    acc = 100.0 * correct / total if total > 0 else 0
    print(f'\nTest set: Accuracy: {correct}/{total} ({acc:.2f}%)\n')

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 打印混淆矩阵（文字版）
    print("Confusion Matrix:")
    print(cm)

    # 绘制热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(16), yticklabels=range(16))
    plt.title(f'Confusion Matrix (using netA only)\nAccuracy: {acc:.2f}%')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵图已保存至: {save_fig_path}")

    # 可选：在 jupyter 等环境中显示
    # plt.show()

    return cm, y_true, y_pred


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

    # 如果你的数据需要通道转置，请根据实际情况决定是否保留
    # X_test = X_test.transpose(0, 2, 1)

    return X_test, Y_test


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rand_num = 30
    n_classes = 16   # 测试集有 16 类

    X_test, Y_test = TestDataset_prepared(n_classes, rand_num)
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 只加载 netA（不再加载 netC）
    netA_path = "model_weight/netA_n_classes_10_label10_unlabel90_rand30.pth"

    print("Loading netA model...")
    netA = torch.load(netA_path, map_location=device)

    # 如果加载的是 state_dict，请替换为：
    # from your_model import NetA  # 导入你的模型定义
    # netA = NetA()  # 或 NetA(num_classes=16) 等
    # netA.load_state_dict(torch.load(netA_path, map_location=device))

    if isinstance(netA, torch.nn.Module):
        netA = netA.to(device)

    print("netA loaded and moved to device successfully.")

    # 执行测试（只用 netA）
    test(netA, test_dataloader, save_fig_path="Visualization/confusion_matrix_16classes_netA_only.png")


if __name__ == '__main__':
    main()