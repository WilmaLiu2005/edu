import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="聚类结果 CSV")
args = parser.parse_args()

df = pd.read_csv(args.input)

# 统计每簇大小
cluster_sizes = df["cluster"].value_counts().sort_index()

# 平均文本长度（字数）
df["length"] = df["dialog_text"].apply(len)

print("===== 聚类统计 =====")
for c, size in cluster_sizes.items():
    avg_len = df[df["cluster"] == c]["length"].mean()
    print(f"簇 {c}: {size} 条, 平均长度 {avg_len:.1f}")

# -----------------------------
# 可视化
# -----------------------------
plt.figure(figsize=(8,5))
cluster_sizes.plot(kind="bar")
plt.title("Cluster Sizes")
plt.xlabel("Cluster ID")
plt.ylabel("Count")
plt.savefig("cluster_sizes_bar.png", dpi=300)

plt.figure(figsize=(6,6))
cluster_sizes.plot(kind="pie", autopct="%1.1f%%")
plt.title("Cluster Distribution")
plt.ylabel("")
plt.savefig("cluster_sizes_pie.png", dpi=300)
plt.show()
