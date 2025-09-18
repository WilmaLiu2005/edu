import os
import argparse
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 命令行参数
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="输入文件夹，包含每轮对话csv")
parser.add_argument("--algorithm", type=str, default="kmeans", choices=["kmeans", "dbscan", "agg", "hdbscan", "spectral"], help="聚类算法")
parser.add_argument("--n_clusters", type=int, default=5, help="簇数")
parser.add_argument("--visualize", action="store_false", help="是否生成可视化图")
parser.add_argument("--eval", action="store_false", help="是否对聚类结果做评估")
parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN 邻域半径")
parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN 最小样本数")
parser.add_argument("--min_cluster_size", type=int, default=2, help="HDBSCAN 最小簇大小")
parser.add_argument("--metric", type=str, default="euclidean", help="HDBSCAN 距离度量")
args = parser.parse_args()

# -----------------------------
# 特征提取函数
# -----------------------------
def extract_features(file_path, file_name):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.fillna("", inplace=True)

    # QA 轮次
    qa_turns = df.shape[0]

    # 问题长度（取平均）
    question_lengths = df["提问内容"].apply(lambda x: len(str(x)))
    avg_q_len = question_lengths.mean()

    # QA 总耗时（分钟）
    df["提问时间"] = pd.to_datetime(df["提问时间"])
    total_time = (df["提问时间"].max() - df["提问时间"].min()).total_seconds() / 60

    # 提问入口不是 "班级" 的比例
    non_class_ratio = (df["提问入口"] != "班级").sum() / qa_turns

    return {
        "file": file_name,
        "qa_turns": qa_turns,
        "avg_q_len": avg_q_len,
        "total_time": total_time,
        "non_class_ratio": non_class_ratio,
    }

# -----------------------------
# 读取对话文件 & 提取特征
# -----------------------------
features = []
for file in os.listdir(args.input):
    if file.endswith(".csv"):
        path = os.path.join(args.input, file)
        feats = extract_features(path, file)
        features.append(feats)

df_features = pd.DataFrame(features)
print(f"读取 {len(df_features)} 个对话文件")

# -----------------------------
# 特征矩阵
# -----------------------------
X = df_features.drop(columns=["file"]).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 聚类
# -----------------------------
if args.algorithm == "kmeans":
    cluster_model = KMeans(n_clusters=args.n_clusters, random_state=42)
elif args.algorithm == "agg":
    cluster_model = AgglomerativeClustering(n_clusters=args.n_clusters)
elif args.algorithm == "dbscan":
    cluster_model = DBSCAN(metric="euclidean", eps=args.eps, min_samples=args.min_samples)
elif args.algorithm == "hdbscan":
    cluster_model = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, metric=args.metric)
elif args.algorithm == "spectral":
    cluster_model = SpectralClustering(n_clusters=args.n_clusters, affinity="nearest_neighbors", random_state=42)

cluster_labels = cluster_model.fit_predict(X_scaled)
df_features["cluster"] = cluster_labels

# -----------------------------
# 聚类评估
# -----------------------------
if args.eval:
    if len(set(cluster_labels)) > 1 and (set(cluster_labels) - {-1}):
        try:
            score = silhouette_score(X_scaled, cluster_labels)
            print(f"Silhouette Score: {score:.4f}")
        except Exception as e:
            print("无法计算 Silhouette Score:", e)
    else:
        print("聚类簇数不足，无法计算 Silhouette Score")

# -----------------------------
# 保存结果
# -----------------------------
output = f"feature_cluster_{args.algorithm}_{args.n_clusters}.csv"
df_features.to_csv(output, index=False, encoding="utf-8-sig")
print(f"聚类结果已保存到 {output}")

# -----------------------------
# 可视化
# -----------------------------
if args.visualize:
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=cluster_labels, cmap="tab10", s=50)
    plt.title(f"Dialog Feature Clustering ({args.algorithm})")
    plt.colorbar(scatter)
    fig_path = f"feature_cluster_visualization_{args.algorithm}_{args.n_clusters}.png"
    plt.savefig(fig_path, dpi=300)
    print(f"可视化已保存到 {fig_path}")
