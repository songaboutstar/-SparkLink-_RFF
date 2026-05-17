import os
import numpy as np

DATA_ROOT = "D:/xingshanorig"      # 原始 16 个设备文件夹
OUT_ROOT  = "C:/kaiji"          # 输出数据集

L = 2048
STRIDE = 1024

# 开集设置：前 10 个设备作为 known classes (dev00 ~ dev09)
KNOWN_NUM = 10

def load_iq(file_path):
    raw = np.fromfile(file_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]
    return iq

def normalize(seg):
    return seg / (np.sqrt(np.mean(np.abs(seg)**2)) + 1e-8)

def process_file(file_path, out_dir, label):
    iq = load_iq(file_path)
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for i in range(0, len(iq) - L, STRIDE):
        seg = iq[i:i+L]
        seg = normalize(seg)
        seg = np.stack([seg.real, seg.imag], axis=0)  # (2, L)

        np.save(os.path.join(out_dir, f"{label}_{count:06d}.npy"), seg)
        count += 1
    return count

devices = sorted(os.listdir(DATA_ROOT))

print(f"【开集实验专用】")
print(f"train/val：只使用前 {KNOWN_NUM} 个设备 (dev00 ~ dev{KNOWN_NUM-1:02d}) 作为 known classes")
print(f"test：使用全部 16 个设备（包含 unknown classes）")

for dev_id, dev in enumerate(devices):
    dev_path = os.path.join(DATA_ROOT, dev)
    files = sorted(os.listdir(dev_path))

    if len(files) < 5:
        print(f"警告：{dev} 文件不足5个，跳过")
        continue

    train_files = files[:3]
    val_files   = files[3:4]
    test_files  = files[4:5]

    label = dev_id  # 全局标签 0~15

    # train 和 val 只处理 known 设备
    if dev_id < KNOWN_NUM:
        for f in train_files:
            process_file(os.path.join(dev_path, f),
                         os.path.join(OUT_ROOT, "train", f"dev{dev_id:02d}"),
                         label)

        for f in val_files:
            process_file(os.path.join(dev_path, f),
                         os.path.join(OUT_ROOT, "val", f"dev{dev_id:02d}"),
                         label)

    # test 处理所有设备（必须包含 unknown）
    for f in test_files:
        process_file(os.path.join(dev_path, f),
                     os.path.join(OUT_ROOT, "test", f"dev{dev_id:02d}"),
                     label)

print("✅ 开集实验数据预处理与划分完成")
print(f"train/val 文件夹：仅 dev00 ~ dev09")
print(f"test 文件夹：dev00 ~ dev15（包含 unknown）")