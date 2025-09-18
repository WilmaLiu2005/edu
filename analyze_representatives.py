import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="聚类结果 CSV (包含 dialog_text, cluster)")
parser.add_argument("--topk", type=int, default=3, help="每簇代表样本数")
args = parser.parse_args()

# -----------------------------
# 加载数据
# -----------------------------
df = pd.read_csv(args.input)

# -----------------------------
# 加载 SentenceTransformer 模型
# -----------------------------
model = SentenceTransformer("/Users/vince/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1")  # 也可以换其他预训练模型

# 生成文本 embeddings
embeddings = model.encode(df["dialog_text"].tolist(), show_progress_bar=True)

# -----------------------------
# 每簇代表样本
# -----------------------------
print("===== 每簇代表样本 =====")
for c in sorted(df["cluster"].unique()):
    idx = df[df["cluster"] == c].index
    if len(idx) == 0:
        continue
    cluster_embs = embeddings[idx]

    # 簇中心
    centroid = cluster_embs.mean(axis=0, keepdims=True)

    # 计算到中心的余弦距离
    dists = cosine_distances(cluster_embs, centroid).reshape(-1)
    top_idx = dists.argsort()[:args.topk]
    reps = df.loc[idx[top_idx], "dialog_text"].tolist()

    print(f"\n簇 {c} 最具代表性的样本:")
    for r in reps:
        print("-", r[:120].replace("\n"," ") + "...")