import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# 关键两行：强制使用支持中文的字体，并解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']          # Windows 常用（黑体）
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 备选：微软雅黑（更现代）
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac 常用
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # Linux / 跨平台推荐（需先安装字体）

plt.rcParams['axes.unicode_minus'] = False   # 解决负号 - 显示为方块的问题
# ---------------- 数据（你可以替换成你真实的实验结果） ----------------
ratios = [10, 15, 20]          # 标注设备占比 (%)
accuracies = [81.38,85.41,87.21]  # 对应准确率 (%)

# ---------------- 绘图 ----------------
plt.figure(figsize=(9, 5), dpi=120)  # 调整画布大小和清晰度

bars = plt.bar(
    ratios,
    accuracies,
    color='skyblue',
    edgecolor='steelblue',
    width=3,               # 条宽（因为x是10间隔，宽度设为6比较好看）
    label='识别准确度'
)

# 在每个条形图上方显示具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.4,
        f'{height:.1f}%',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

# 美化
plt.xlabel('标注设备占比 (%)', fontsize=12, fontweight='bold')
plt.ylabel('识别准确度 (%)', fontsize=12, fontweight='bold')
plt.title('不同标注设备占比下的16类星闪信号识别准确度', fontsize=14, fontweight='bold', pad=15)

plt.xticks(ratios)                    # 确保横轴只显示这些刻度
plt.ylim(65, 100)                     # 根据数据调整纵轴范围，避免太挤
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(loc='lower right')

# 可选：添加一条基准线（例如全标注时的性能）
plt.axhline(y=96.8, color='red', linestyle='--', alpha=0.6,
            label='全标注基准 (100%)')

plt.tight_layout()
plt.show()