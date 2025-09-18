import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ----------------------------
# macOS 中文字体设置
# ----------------------------
plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 用户输入文件夹路径
input_dir = input("请输入包含学生CSV文件的文件夹路径: ").strip()
output_dir = os.path.join(input_dir, "visualization")
os.makedirs(output_dir, exist_ok=True)

# 遍历 CSV 文件
for file_name in os.listdir(input_dir):
    if not file_name.endswith(".csv"):
        continue
    file_path = os.path.join(input_dir, file_name)
    df = pd.read_csv(file_path)

    # 检查必要列
    if not {"提问入口", "提问内容", "提问时间"}.issubset(df.columns):
        print(f"{file_name} 缺少必要列，跳过")
        continue

    df["提问时间"] = pd.to_datetime(df["提问时间"])

    # 提取日期和当天时间（分钟）
    df["日期"] = df["提问时间"].dt.date
    df["天内分钟"] = df["提问时间"].dt.hour * 60 + df["提问时间"].dt.minute

    plt.figure(figsize=(12, 4))

    # 自动为每个提问入口生成颜色
    unique_entries = df["提问入口"].unique()
    n_entries = len(unique_entries)
    cmap = cm.get_cmap("tab20", n_entries)
    entry_colors = {entry: cmap(i) for i, entry in enumerate(unique_entries)}

    # 绘制散点
    for idx, row in df.iterrows():
        entry = row["提问入口"]
        color = entry_colors.get(entry, "gray")
        plt.scatter(row["日期"], row["天内分钟"], color=color, s=30)

    # 添加图例
    for entry, color in entry_colors.items():
        plt.scatter([], [], color=color, label=entry)
    plt.legend(title="提问入口", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel("日期")
    plt.ylabel("当天时间（分钟）")
    plt.title(f"{file_name} 问答时间分布")
    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"{file_name} 可视化已保存到 {save_path}")