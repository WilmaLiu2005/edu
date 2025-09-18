import os

# -----------------------------
# 目标文件夹列表
# -----------------------------
folders = [
    "/Users/vince/undergraduate/KEG/edu/real_GT_split",
    "/Users/vince/undergraduate/KEG/edu/similarity03",
    "/Users/vince/undergraduate/KEG/edu/test_split2",
    "/Users/vince/undergraduate/KEG/edu/time15"
]

# -----------------------------
# 遍历统计
# -----------------------------
for folder in folders:
    if not os.path.exists(folder):
        print(f"{folder} 不存在")
        continue

    csv_count = sum(1 for f in os.listdir(folder) if f.endswith(".csv"))
    print(f"{folder} 内 CSV 文件数量: {csv_count}")
