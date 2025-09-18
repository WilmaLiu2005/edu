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
from collections import Counter
import re
from openai import OpenAI
import random
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import json5
import csv
from collections import Counter
from multiprocessing import Pool
from functools import partial
import warnings
warnings.filterwarnings('ignore')
# ==============================    

MAX_WORKERS = 50
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 3
API_KEY = "sk-0jErqj61bIYM135CEqhfj318rKIM1TIa"  # 填写你的 API key
BASE_URL = "https://api-gateway.glm.ai/v1"
MODEL_NAME = "gemini-2.5-flash"  # 或你自己的模型

# ==============================
# GPT API 调用
# ==============================
def gpt_api_call(messages, model=MODEL_NAME):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=10000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_RETRY_DELAY)
                time.sleep(delay)
            else:
                return None

# ==============================
# Prompt 模板
# ==============================
PROMPT_TEMPLATE = """
你是一个专业的对话分析助手。请仔细分析以下同学与AI学伴的完整对话内容，并执行以下任务：

任务要求：
1. 总结对话的核心主题和主要内容
2. 提取5个最具代表性的关键词，要求：
   - 关键词必须简洁（2-5个汉字）
   - 能准确反映对话的核心主题
   - 避免重复或近义词
   - 优先选择专业术语或核心概念词

输出格式：
严格返回JSON数组格式，仅包含5个关键词字符串，无需其他解释。
示例：["机器学习", "算法优化", "数据预处理", "模型评估", "特征工程"]

对话内容：
{dialog_text}

请提取关键词：
"""

# ==============================
# JSON 解析（鲁棒）
# ==============================
def robust_json_parse(text):
    # 优先提取第一个 ```json ... ``` 代码块
    match = re.search(r"```json(.*?)```", text, flags=re.S)
    if match:
        cleaned = match.group(1).strip()
    else:
        # 没有 ```json```，再考虑整个文本就是 JSON
        cleaned = text.strip()

    # 先尝试标准 JSON
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 再尝试 JSON5
    try:
        return json5.loads(cleaned)
    except Exception:
        print("⚠️ JSON 解析失败，原始输出：", text)
        return {}


# -----------------------------
# 关键词提取函数
# -----------------------------
def extract_keywords(dialog_text):
    prompt = PROMPT_TEMPLATE.format(dialog_text=dialog_text)
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = gpt_api_call(messages)
    if response:
        keywords = robust_json_parse(response)
        if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
            return keywords
    raise ValueError("无法提取关键词")


def process_dialog_file(file_path, model, normalize, batch_size, top_k):
    """
    处理单个对话文件
    :param file_path: CSV文件路径
    :param model: 共享的SentenceTransformer模型
    :param normalize: 是否归一化
    :param batch_size: 批处理大小
    :param top_k: 关键词数量
    :return: 文件名, 关键词列表, 嵌入向量
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.fillna("", inplace=True)
        
        # 拼接整个对话文本
        turns = [f"{df.loc[i,'提问内容']} [Q-A] {df.loc[i,'AI回复']}" for i in range(len(df))]
        text = " [TURN] ".join(turns)
        
        # 提取关键词
        keywords = extract_keywords(text)
        
        # 关键词嵌入
        encode_kwargs = {
            "normalize_embeddings": normalize,
            "batch_size": batch_size
        }
        
        if keywords:
            embs = model.encode(keywords, **encode_kwargs)
            embedding = np.mean(embs, axis=0)  # 平均池化
        else:
            embedding = np.zeros(model.get_sentence_embedding_dimension())
        
        return os.path.basename(file_path), keywords, embedding
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return os.path.basename(file_path), [], np.zeros(model.get_sentence_embedding_dimension())

# -----------------------------
# 命令行参数
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="输入文件夹，包含多个子文件夹")
parser.add_argument("--output", type=str, default="cluster_result", help="输出文件夹")
parser.add_argument("--algorithm", type=str, default="kmeans", choices=["kmeans", "dbscan", "agg", "hdbscan", "spectral"], help="聚类算法")
parser.add_argument("--n_clusters", type=int, default=5, help="KMeans/层次/谱聚类簇数")
parser.add_argument("--visualize", action="store_true", help="是否生成可视化图")
parser.add_argument("--dimred", type=str, default="pca", choices=["pca","tsne","umap","none"], help="降维方法")
parser.add_argument("--embedding_model", type=str, default="/Users/vince/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1", help="SentenceTransformer嵌入模型名称")
parser.add_argument("--eval", action="store_true", help="是否对聚类结果做评估")
parser.add_argument("--top_k", type=int, default=10, help="提取的关键词数量")
parser.add_argument("--normalize", action="store_true", help="是否归一化 embeddings")
parser.add_argument("--batch_size", type=int, default=128, help="嵌入时 batch_size")
parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN 邻域半径")
parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN 最小样本数")
parser.add_argument("--min_cluster_size", type=int, default=2, help="HDBSCAN 最小簇大小")
parser.add_argument("--metric", type=str, default="cosine", help="HDBSCAN 距离度量")
parser.add_argument("--max_workers", type=int, default=50, help="并行进程数，默认使用所有CPU核心")
args = parser.parse_args()

# -----------------------------
# 创建输出文件夹
# -----------------------------
os.makedirs(args.output, exist_ok=True)

# -----------------------------
# 获取所有子文件夹
# -----------------------------
subdirs = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
print(f"找到 {len(subdirs)} 个子文件夹待处理")

# -----------------------------
# 创建输出文件夹
# -----------------------------
os.makedirs(args.output, exist_ok=True)

# -----------------------------
# 加载模型（一次）
# -----------------------------
print(f"加载嵌入模型: {args.embedding_model}")
model = SentenceTransformer(args.embedding_model)
embedding_dim = model.get_sentence_embedding_dimension()
print(f"模型维度: {embedding_dim}")

# -----------------------------
# 遍历一级子文件夹
# -----------------------------
subdirs = [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
print(f"找到 {len(subdirs)} 个子文件夹待处理")

for subdir in subdirs:
    subdir_path = os.path.join(args.input, subdir)
    print(f"\n正在处理子文件夹: {subdir}")
    
    # 获取该子文件夹内所有CSV文件
    csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
    csv_paths = [os.path.join(subdir_path, f) for f in csv_files]
    
    if not csv_files:
        print(f"子文件夹 {subdir} 中没有CSV文件，跳过")
        continue
    
    print(f"子文件夹 {subdir} 中有 {len(csv_files)} 个CSV文件")
    
    # 创建该子文件夹的输出目录
    subdir_output = os.path.join(args.output, subdir)
    os.makedirs(subdir_output, exist_ok=True)
    
    # 使用线程池并发处理该子文件夹内的CSV文件
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_dialog_file, file_path, model, args.normalize, args.batch_size, args.top_k): file_path
            for file_path in csv_paths
        }
        
        # 收集结果
        results = []
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"文件 {file_path} 处理失败: {e}")
    
    # 解包结果
    filenames = []
    keywords_list = []
    embeddings_list = []
    
    for filename, keywords, embedding in results:
        filenames.append(filename)
        keywords_list.append(keywords)
        embeddings_list.append(embedding)
    
    # 转换为numpy数组
    embeddings = np.vstack(embeddings_list)
    
    print(f"子文件夹 {subdir} 的所有文件处理完成，共 {len(filenames)} 个对话")
    
    # -----------------------------
    # 聚类
    # -----------------------------
    print(f"使用 {args.algorithm} 算法进行聚类...")
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
    
    # -----------------------------
    # 创建结果DataFrame
    # -----------------------------
    df_results = pd.DataFrame({
        "file": filenames,
        "keywords": keywords_list,
        "cluster": cluster_labels
    })
    
    # -----------------------------
    # 聚类评估
    # -----------------------------
    if args.eval:
        # 过滤掉全是 -1 (DBSCAN/HDBSCAN 全部判定为噪声) 的情况
        if len(set(cluster_labels)) > 1 and (set(cluster_labels) - {-1}):
            try:
                score = silhouette_score(embeddings, cluster_labels)
                print(f"Silhouette Score: {score:.4f}")
            except Exception as e:
                print("无法计算 Silhouette Score:", e)
        else:
            print("聚类簇数不足，无法计算 Silhouette Score")
    
    # -----------------------------
    # 保存结果CSV
    # -----------------------------
    output_csv = os.path.join(subdir_output, f"{args.algorithm}_{args.dimred}_keywords_{args.n_clusters}.csv")
    df_results.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"聚类结果已保存到 {output_csv}")
    
    # -----------------------------
    # 可视化
    # -----------------------------
    if args.visualize:
        print("生成可视化图...")
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
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap="tab10", s=50)
        plt.title(f"Dialog Clustering ({args.algorithm}) - {subdir} ({len(filenames)} Files)")
        plt.colorbar(scatter)
        
        # 添加文件名标签（只显示部分，避免过于拥挤）
        step = max(1, len(filenames) // 20)  # 最多显示20个标签
        for i in range(0, len(filenames), step):
            plt.annotate(filenames[i], (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.7)
        
        safe_model_name = args.embedding_model.replace("/", "_")
        fig_path = os.path.join(subdir_output, f"cluster_visualization_{args.algorithm}_{args.dimred}_keywords_{args.n_clusters}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"可视化已保存到 {fig_path}")

print("\n所有子文件夹处理完成！")