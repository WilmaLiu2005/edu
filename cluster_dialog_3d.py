import os
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.metrics import silhouette_score, davies_bouldin_score

# -----------------------------
# 命令行参数
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="输入文件夹，包含每轮对话csv")
parser.add_argument("--output", type=str, default="cluster_result.csv", help="聚类结果CSV")
parser.add_argument("--algorithm", type=str, default="kmeans", choices=["kmeans", "dbscan", "agg"], help="聚类算法")
parser.add_argument("--n_clusters", type=int, default=5, help="KMeans/层次/谱聚类簇数")
parser.add_argument("--visualize", action="store_true", help="是否生成可视化图")
parser.add_argument("--dimred", type=str, default="pca", choices=["pca","tsne","umap","none"], help="降维方法")
parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer嵌入模型名称")
parser.add_argument("--eval", action="store_true", help="是否对聚类结果做评估")  # ✅ 新增参数
parser.add_argument(
    "--embedding_mode",
    type=str,
    default="full",
    choices=["full","qa_turn","question_only","answer_only"],
    help="嵌入模式：full=整轮对话, qa_turn=每轮Q-A单独嵌入并平均, question_only=只嵌入问题, answer_only=只嵌入回答"
)
parser.add_argument("--normalize", action="store_true", help="是否归一化 embeddings (normalize_embeddings=True)")
parser.add_argument("--batch_size", type=int, default=128, help="嵌入时 batch_size")
parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN 邻域半径")
parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN 最小样本数")
parser.add_argument("--min_cluster_size", type=int, default=2, help="HDBSCAN 最小簇大小")
parser.add_argument("--metric", type=str, default="cosine", help="HDBSCAN 距离度量")
args = parser.parse_args()

# -----------------------------
# 读取对话文件并拼接
# -----------------------------
dialogs = []
for file in os.listdir(args.input):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(args.input, file), encoding="utf-8-sig")
        df.fillna("", inplace=True)
        if args.embedding_mode == "full":
            turns = [f"{df.loc[i,'提问内容']} [Q-A] {df.loc[i,'AI回复']}" for i in range(len(df))]
            text = " [TURN] ".join(turns)
            dialogs.append({"file": file, "dialog_text": text})
        elif args.embedding_mode == "qa_turn":
            # 每轮 Q-A 分开做 embedding
            turns = [f"{df.loc[i,'提问内容']} [Q-A] {df.loc[i,'AI回复']}" for i in range(len(df))]
            dialogs.append({"file": file, "dialog_text_list": turns})
        elif args.embedding_mode == "question_only":
            turns = [df.loc[i,'提问内容'] for i in range(len(df))]
            dialogs.append({"file": file, "dialog_text_list": turns})
        elif args.embedding_mode == "answer_only":
            turns = [df.loc[i,'AI回复'] for i in range(len(df))]
            dialogs.append({"file": file, "dialog_text_list": turns})

df_dialogs = pd.DataFrame(dialogs)
print(f"读取 {len(df_dialogs)} 个对话文件")

# -----------------------------
# 文本嵌入
# -----------------------------
print(f"使用嵌入模型: {args.embedding_model}")
model = SentenceTransformer(args.embedding_model)

encode_kwargs = {
    "normalize_embeddings": args.normalize,
    "batch_size": args.batch_size
}

if args.embedding_mode == "full":
    embeddings = model.encode(df_dialogs["dialog_text"].tolist(), **encode_kwargs)
else:
    embeddings_list = []
    for texts in df_dialogs["dialog_text_list"]:
        embs = model.encode(texts, **encode_kwargs)
        embeddings_list.append(np.mean(embs, axis=0))  # 平均池化
    embeddings = np.vstack(embeddings_list)

# -----------------------------
# 聚类
# -----------------------------
if args.algorithm == "kmeans":
    cluster_model = KMeans(n_clusters=args.n_clusters, random_state=42)
elif args.algorithm == "agg":
    cluster_model = AgglomerativeClustering(n_clusters=args.n_clusters)
elif args.algorithm == "dbscan":
    cluster_model = DBSCAN(metric="cosine", eps=args.eps, min_samples=args.min_samples)
elif args.algorithm == "hdbscan":
    cluster_model = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, metric=args.metric)
elif args.algorithm == "spectral":
    cluster_model = SpectralClustering(n_clusters=args.n_clusters, affinity="nearest_neighbors", random_state=42)

cluster_labels = cluster_model.fit_predict(embeddings)
df_dialogs["cluster"] = cluster_labels

# -----------------------------
# 聚类评估
# -----------------------------
if args.eval:
    # 过滤掉全是 -1 (DBSCAN/HDBSCAN 全部判定为噪声) 的情况
    valid_labels = set(cluster_labels) - {-1}  # 去掉噪声
    if len(valid_labels) > 1:
        try:
            # Silhouette Score
            sil_score = silhouette_score(embeddings, cluster_labels)
            print(f"Silhouette Score: {sil_score:.4f}")
        except Exception as e:
            print("无法计算 Silhouette Score:", e)

        try:
            # Davies-Bouldin Index (值越小越好)
            db_index = davies_bouldin_score(embeddings, cluster_labels)
            print(f"Davies-Bouldin Index: {db_index:.4f}")
        except Exception as e:
            print("无法计算 Davies-Bouldin Index:", e)
    else:
        print("聚类簇数不足，无法计算 Silhouette Score 或 Davies-Bouldin Index")

# -----------------------------
# 保存结果CSV
# -----------------------------
output = f"{args.algorithm}_{args.dimred}_{args.embedding_mode}_{args.n_clusters}.csv"
df_dialogs.to_csv(output, index=False, encoding="utf-8-sig")
print(f"聚类结果已保存到 {output}")

# -----------------------------
# 可视化
# -----------------------------
if args.visualize:
    if args.dimred == "pca":
        reducer = PCA(n_components=3)
        reduced = reducer.fit_transform(embeddings)
    elif args.dimred == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    elif args.dimred == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        reduced = reducer.fit_transform(embeddings)
    else:  # none
        reduced = embeddings[:, :2]  # 直接取前两维

    # 3D 可视化
    # 构建 DataFrame
    df = pd.DataFrame({
        "Dim1": reduced[:, 0],
        "Dim2": reduced[:, 1],
        "Dim3": reduced[:, 2],
        "Cluster": cluster_labels.astype(str)  # 转成字符串便于 Plotly 显示颜色
    })

    # 交互式 3D 散点图
    fig = px.scatter_3d(
        df,
        x='Dim1',
        y='Dim2',
        z='Dim3',
        color='Cluster',
        title=f"Dialog Clustering ({args.algorithm})",
        color_discrete_sequence=px.colors.qualitative.Set1,
        width=900,
        height=700
    )

    # 显示交互式图
    fig.show()

    # 保存为 HTML 文件（可直接用浏览器打开交互）
    fig_path = f"cluster_visualization_3d_{args.algorithm}_{args.dimred}_{args.embedding_mode}_{args.n_clusters}-3dim.html"
    fig.write_html(fig_path)
    print(f"可交互 3D 可视化已保存到 {fig_path}")