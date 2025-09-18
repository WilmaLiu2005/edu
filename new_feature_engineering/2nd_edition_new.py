# 用LLM提取出他提问问法的序列，用这个序列进行聚类
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time
import random
from openai import OpenAI
import argparse
import ast
import re
import json

def _normalize_to_int_list(parsed):
    """把 parsed 转成整数列表，或返回 None 表示不符合要求。"""
    if isinstance(parsed, int):
        return [parsed]
    if isinstance(parsed, list):
        # 如果全部是 int
        if all(isinstance(i, int) for i in parsed):
            return parsed
        # 如果是整值 float 或 int，转成 int
        if all(isinstance(i, (int, float)) and float(i).is_integer() for i in parsed):
            return [int(i) for i in parsed]
    return None

def parse_qa_mode_list_str(s: str):
    """
    尝试鲁棒地从字符串 s 中解析出问答模式整数列表。
    成功返回 List[int]，失败抛出 ValueError。
    """
    if s is None:
        raise ValueError("内容为空")
    s = s.strip()
    if not s:
        raise ValueError("内容为空")
    # 去 BOM
    s = s.lstrip("\ufeff")

    # 1) 去除 markdown 三重代码块 ```json ... ``` 的包裹
    m = re.match(r"^```(?:\w+)?\s*(.*?)\s*```$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()

    # 2) 去除单行反引号包裹 `...`
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1].strip()

    # 3) 尝试 json.loads（对标准 JSON 最友好）
    try:
        parsed = json.loads(s)
        norm = _normalize_to_int_list(parsed)
        if norm is not None:
            return norm
    except Exception:
        pass

    # 4) 尝试 ast.literal_eval（对 Python 风格的列表/数字也行）
    try:
        parsed = ast.literal_eval(s)
        norm = _normalize_to_int_list(parsed)
        if norm is not None:
            return norm
    except Exception:
        pass

    # 5) 从任意文本中提取第一个方括号风格的列表（例如 "Result: [2,3]"）
    m = re.search(r"\[ *-?\d+(?: *[, ] *-?\d+)* *\]", s)
    if m:
        candidate = m.group(0)
        try:
            parsed = ast.literal_eval(candidate)
            norm = _normalize_to_int_list(parsed)
            if norm is not None:
                return norm
        except Exception:
            pass

    # 6) 回退：提取第一个整数（例如文本里只有 "3" 或 "some 3 text"）
    m = re.search(r"-?\d+", s)
    if m:
        return [int(m.group(0))]

    # 解析失败
    raise ValueError("无法解析为整数列表")

# -----------------------------
# 文件夹批量处理
# -----------------------------
input_folder = "/Users/vince/undergraduate/KEG/edu/full_split"  # CSV 文件夹
output_txt_folder = "/Users/vince/undergraduate/KEG/edu/full_split/qa_mode_txts"
os.makedirs(output_txt_folder, exist_ok=True)
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

# 解析命令行参数
parser = argparse.ArgumentParser(description='问答模式序列聚类分析')
parser.add_argument("--n_clusters", type=int, default=5, help="簇数")
parser.add_argument("--method", type=str, default="enhanced", 
                   choices=["original", "enhanced", "ngram", "similarity", "hybrid"],
                   help="聚类方法选择: original(原始), enhanced(增强特征), ngram(N-gram), similarity(序列相似度), hybrid(混合方法)")
parser.add_argument("--algorithm", type=str, default="kmeans",
                   choices=["kmeans", "dbscan", "agglomerative", "gmm"],
                   help="聚类算法: kmeans, dbscan, agglomerative, gmm")
args = parser.parse_args()

n_clusters = args.n_clusters
method = args.method
algorithm = args.algorithm

# 创建输出文件夹
output_folder = f"2nd_edition_{method}_{n_clusters}"
os.makedirs(output_folder, exist_ok=True)

print(f"使用方法: {method}, 算法: {algorithm}, 聚类数: {n_clusters}")
print(f"结果将保存到: {output_folder}")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ==============================
# 不同的特征提取方法
# ==============================

def original_qa_sequence_to_vector(seq, n_classes=5):
    """原始的特征提取方法"""
    seq = np.array(seq)
    freq_vector = np.array([(seq==i).sum()/len(seq) for i in range(n_classes)])
    switches = np.sum(seq[1:] != seq[:-1])
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    trans_matrix = np.zeros((n_classes,n_classes))
    for i in range(1,len(seq)):
        trans_matrix[seq[i-1], seq[i]] +=1
    if len(seq) > 1:
        trans_matrix = trans_matrix / (len(seq)-1)
    trans_vector = trans_matrix.flatten()
    feature_vector = np.concatenate([freq_vector,[switches],[max_run], trans_vector])
    return feature_vector

def enhanced_qa_sequence_to_vector(seq, n_classes=5):
    """增强的特征提取方法"""
    seq = np.array(seq)
    features = []
    
    # 1. 基础频率特征
    freq_vector = np.array([(seq==i).sum()/len(seq) for i in range(n_classes)])
    features.extend(freq_vector)
    
    # 2. 序列长度特征（标准化）
    seq_length = len(seq)
    features.append(seq_length)
    
    # 3. 转换特征
    if len(seq) > 1:
        switches = np.sum(seq[1:] != seq[:-1])
        switch_rate = switches / (len(seq) - 1)
        features.extend([switches, switch_rate])
    else:
        features.extend([0, 0])
    
    # 4. 连续段分析
    runs = []
    run_types = []
    current_run = 1
    current_type = seq[0]
    
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
        else:
            runs.append(current_run)
            run_types.append(current_type)
            current_run = 1
            current_type = seq[i]
    runs.append(current_run)
    run_types.append(current_type)
    
    # 连续段统计
    features.extend([
        max(runs) if runs else 1,
        np.mean(runs) if runs else 1,
        np.std(runs) if len(runs) > 1 else 0,
        len(runs)
    ])
    
    # 5. 各类别的最长连续段
    for class_id in range(n_classes):
        class_runs = [r for r, t in zip(runs, run_types) if t == class_id]
        max_class_run = max(class_runs) if class_runs else 0
        features.append(max_class_run)
    
    # 6. 位置特征
    features.append(seq[0])
    features.append(seq[-1])
    
    # 7. 模式多样性
    unique_elements = len(np.unique(seq))
    diversity = unique_elements / n_classes
    features.append(diversity)
    
    # 8. 简化的转移概率
    trans_probs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                count = 0
                for k in range(1, len(seq)):
                    if seq[k-1] == i and seq[k] == j:
                        count += 1
                prob = count / (len(seq) - 1) if len(seq) > 1 else 0
                trans_probs.append(prob)
    features.extend(trans_probs)
    
    # 9. 序列模式特征
    dominant_class = np.argmax(freq_vector)
    dominance = freq_vector[dominant_class]
    features.append(dominance)
    
    if len(seq) > 1:
        stability = np.sum(seq[1:] == seq[:-1]) / (len(seq) - 1)
    else:
        stability = 1.0
    features.append(stability)
    
    return np.array(features)

def ngram_feature_extraction(sequences, max_features=50):
    """N-gram特征提取"""
    from sklearn.feature_extraction.text import CountVectorizer
    
    documents = []
    for seq in sequences:
        seq_str = [str(x) for x in seq]
        unigrams = seq_str
        bigrams = [f"{seq[i]}-{seq[i+1]}" for i in range(len(seq)-1)]
        trigrams = [f"{seq[i]}-{seq[i+1]}-{seq[i+2]}" for i in range(len(seq)-2)]
        all_grams = unigrams + bigrams + trigrams
        doc = ' '.join(all_grams)
        documents.append(doc)
    
    vectorizer = CountVectorizer(max_features=max_features, token_pattern=r'\S+')
    X = vectorizer.fit_transform(documents)
    return X.toarray(), vectorizer

def sequence_similarity_clustering(sequences, n_clusters=5):
    """基于序列相似度的聚类"""
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    
    def edit_distance(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def sequence_distance(seq1, seq2):
        edit_dist = edit_distance(seq1, seq2) / max(len(seq1), len(seq2))
        freq1 = np.array([(np.array(seq1)==i).sum()/len(seq1) for i in range(5)])
        freq2 = np.array([(np.array(seq2)==i).sum()/len(seq2) for i in range(5)])
        freq_dist = np.linalg.norm(freq1 - freq2)
        len_diff = abs(len(seq1) - len(seq2)) / max(len(seq1), len(seq2))
        return 0.4 * edit_dist + 0.4 * freq_dist + 0.2 * len_diff
    
    n = len(sequences)
    distances = []
    
    for i in range(n):
        for j in range(i+1, n):
            dist = sequence_distance(sequences[i], sequences[j])
            distances.append(dist)
    
    linkage_matrix = linkage(distances, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
    
    return cluster_labels

# ==============================
# 聚类算法选择
# ==============================

def apply_clustering_algorithm(X_scaled, algorithm, n_clusters):
    """应用指定的聚类算法"""
    if algorithm == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X_scaled)
    elif algorithm == "dbscan":
        clusterer = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = clusterer.fit_predict(X_scaled)
        # 将噪声点(-1)重新编号
        unique_labels = np.unique(cluster_labels)
        if -1 in unique_labels:
            cluster_labels[cluster_labels == -1] = len(unique_labels) - 1
    elif algorithm == "agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X_scaled)
    elif algorithm == "gmm":
        clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X_scaled)
    else:
        raise ValueError(f"不支持的聚类算法: {algorithm}")
    
    return cluster_labels

# ==============================
# 可视化函数
# ==============================

def comprehensive_clustering_visualization(X_scaled, cluster_labels, all_sequences, dialog_names, n_clusters, output_folder):
    """综合聚类可视化"""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. PCA降维可视化
    plt.subplot(2, 3, 1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                         cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.title(f'PCA降维聚类可视化\n解释方差比: {pca.explained_variance_ratio_.sum():.3f}', fontsize=12)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
    plt.colorbar(scatter, label='聚类标签')
    
    # 添加聚类中心
    for i in range(n_clusters):
        cluster_points = X_pca[cluster_labels == i]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            plt.scatter(centroid[0], centroid[1], c='red', s=200, marker='x', linewidth=3)
            plt.annotate(f'C{i}', (centroid[0], centroid[1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # 2. t-SNE降维可视化
    plt.subplot(2, 3, 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)-1))
    X_tsne = tsne.fit_transform(X_scaled)
    
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, 
                         cmap='tab10', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.title('t-SNE降维聚类可视化', fontsize=12)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar(scatter, label='聚类标签')
    
    # 3. 聚类大小分布
    plt.subplot(2, 3, 3)
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    bars = plt.bar(range(n_clusters), cluster_counts.values, 
                   color=plt.cm.tab10(np.arange(n_clusters)), alpha=0.7, edgecolor='black')
    plt.title('各聚类样本数量分布', fontsize=12)
    plt.xlabel('聚类标签')
    plt.ylabel('样本数量')
    plt.xticks(range(n_clusters))
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 序列长度分布
    plt.subplot(2, 3, 4)
    seq_lengths = [len(seq) for seq in all_sequences]
    
    length_data = []
    length_labels = []
    for i in range(n_clusters):
        cluster_lengths = [seq_lengths[j] for j in range(len(seq_lengths)) if cluster_labels[j] == i]
        if cluster_lengths:
            length_data.append(cluster_lengths)
            length_labels.append(f'C{i}')
    
    if length_data:
        box_plot = plt.boxplot(length_data, labels=length_labels, patch_artist=True)
        colors = plt.cm.tab10(np.arange(len(length_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    plt.title('各聚类序列长度分布', fontsize=12)
    plt.xlabel('聚类标签')
    plt.ylabel('序列长度')
    plt.grid(True, alpha=0.3)
    
    # 5. 各聚类的类别分布热力图
    plt.subplot(2, 3, 5)
    
    freq_matrix = np.zeros((n_clusters, 5))
    for i in range(n_clusters):
        cluster_seqs = [all_sequences[j] for j in range(len(all_sequences)) if cluster_labels[j] == i]
        if cluster_seqs:
            cluster_freqs = []
            for seq in cluster_seqs:
                freq = [(np.array(seq)==k).sum()/len(seq) for k in range(5)]
                cluster_freqs.append(freq)
            freq_matrix[i] = np.mean(cluster_freqs, axis=0)
    
    im = plt.imshow(freq_matrix, cmap='YlOrRd', aspect='auto')
    plt.title('各聚类的问答模式分布', fontsize=12)
    plt.xlabel('问答模式类别')
    plt.ylabel('聚类标签')
    plt.xticks(range(5), ['知识型\n(0)', '练习型\n(1)', '解释型\n(2)', '追问型\n(3)', '闲聊型\n(4)'])
    plt.yticks(range(n_clusters), [f'聚类{i}' for i in range(n_clusters)])
    
    for i in range(n_clusters):
        for j in range(5):
            plt.text(j, i, f'{freq_matrix[i,j]:.2f}', 
                    ha='center', va='center', fontweight='bold',
                    color='white' if freq_matrix[i,j] > 0.5 else 'black')
    
    plt.colorbar(im, label='平均频率')
    
    # 6. 聚类特征雷达图
    plt.subplot(2, 3, 6)
    
    cluster_stats = []
    for i in range(n_clusters):
        cluster_seqs = [all_sequences[j] for j in range(len(all_sequences)) if cluster_labels[j] == i]
        if cluster_seqs:
            lengths = [len(seq) for seq in cluster_seqs]
            switches = []
            diversities = []
            
            for seq in cluster_seqs:
                if len(seq) > 1:
                    switches.append(np.sum(np.array(seq)[1:] != np.array(seq)[:-1]) / (len(seq)-1))
                else:
                    switches.append(0)
                diversities.append(len(np.unique(seq)) / 5)
            
            stats = {
                'avg_length': np.mean(lengths) / max([len(s) for s in all_sequences]),
                'avg_switches': np.mean(switches),
                'avg_diversity': np.mean(diversities),
                'dominant_freq': np.max(freq_matrix[i]),
                'stability': 1 - np.mean(switches)
            }
            cluster_stats.append(stats)
    
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(2, 3, 6, projection='polar')
    
    features = ['长度', '转换率', '多样性', '主导频率', '稳定性']
    
    for i, stats in enumerate(cluster_stats[:min(3, len(cluster_stats))]):
        values = list(stats.values())
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'聚类{i}', alpha=0.7)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(0, 1)
    plt.title('聚类特征对比（前3个聚类）', fontsize=12, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'comprehensive_clustering_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_sequence_patterns(all_sequences, cluster_labels, n_clusters, output_folder, max_examples=3):
    """可视化每个聚类的序列模式"""
    fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 3*n_clusters))
    if n_clusters == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    labels = ['知识型', '练习型', '解释型', '追问型', '闲聊型']
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        
        cluster_seqs = [all_sequences[i] for i in range(len(all_sequences)) 
                       if cluster_labels[i] == cluster_id]
        
        if not cluster_seqs:
            ax.text(0.5, 0.5, f'聚类 {cluster_id}: 无数据', 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        selected_seqs = cluster_seqs[:max_examples]
        
        for i, seq in enumerate(selected_seqs):
            y_pos = i
            x_pos = 0
            
            for j, class_id in enumerate(seq):
                ax.barh(y_pos, 1, left=x_pos, color=colors[class_id], 
                       alpha=0.8, edgecolor='white', linewidth=1)
                
                if len(seq) <= 20:
                    ax.text(x_pos + 0.5, y_pos, str(class_id), 
                           ha='center', va='center', fontweight='bold',
                           color='white' if class_id in [3, 4] else 'black')
                x_pos += 1
        
        ax.set_xlim(0, max(len(seq) for seq in selected_seqs) if selected_seqs else 1)
        ax.set_ylim(-0.5, len(selected_seqs) - 0.5)
        ax.set_yticks(range(len(selected_seqs)))
        ax.set_yticklabels([f'样本{i+1}' for i in range(len(selected_seqs))])
        ax.set_xlabel('序列位置')
        ax.set_title(f'聚类 {cluster_id} 序列模式 (共{len(cluster_seqs)}个序列)')
        ax.grid(True, alpha=0.3)
    
    legend_elements = [plt.Rectangle((0,0),1,1, color=colors[i], label=f'{i}-{labels[i]}') 
                      for i in range(5)]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sequence_patterns_by_cluster.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_cluster_transitions(all_sequences, cluster_labels, n_clusters, output_folder):
    """绘制各聚类的转移模式"""
    fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 4))
    if n_clusters == 1:
        axes = [axes]
    
    for cluster_id in range(n_clusters):
        ax = axes[cluster_id]
        
        cluster_seqs = [all_sequences[i] for i in range(len(all_sequences)) 
                       if cluster_labels[i] == cluster_id]
        
        trans_matrix = np.zeros((5, 5))
        total_transitions = 0
        
        for seq in cluster_seqs:
            for i in range(1, len(seq)):
                trans_matrix[seq[i-1], seq[i]] += 1
                total_transitions += 1
        
        if total_transitions > 0:
            trans_matrix = trans_matrix / total_transitions
        
        im = ax.imshow(trans_matrix, cmap='Blues', vmin=0, vmax=trans_matrix.max())
        
        for i in range(5):
            for j in range(5):
                if trans_matrix[i, j] > 0:
                    ax.text(j, i, f'{trans_matrix[i,j]:.2f}', 
                           ha='center', va='center', fontweight='bold',
                           color='white' if trans_matrix[i,j] > trans_matrix.max()/2 else 'black')
        
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(['知识', '练习', '解释', '追问', '闲聊'])
        ax.set_yticklabels(['知识', '练习', '解释', '追问', '闲聊'])
        ax.set_xlabel('转移到')
        ax.set_ylabel('转移从')
        ax.set_title(f'聚类 {cluster_id} 转移模式\n({len(cluster_seqs)}个序列)')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cluster_transition_patterns.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_detailed_statistics(all_sequences, cluster_labels, n_clusters, output_folder):
    """详细统计分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 长度分布直方图
    ax1 = axes[0, 0]
    seq_lengths = [len(seq) for seq in all_sequences]
    
    for i in range(n_clusters):
        cluster_lengths = [seq_lengths[j] for j in range(len(seq_lengths)) if cluster_labels[j] == i]
        if cluster_lengths:
            ax1.hist(cluster_lengths, alpha=0.6, label=f'聚类{i}', bins=20)
    
    ax1.set_xlabel('序列长度')
    ax1.set_ylabel('频次')
    ax1.set_title('各聚类序列长度分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 多样性分析
    ax2 = axes[0, 1]
    diversities = []
    cluster_ids = []
    
    for i in range(n_clusters):
        cluster_seqs = [all_sequences[j] for j in range(len(all_sequences)) if cluster_labels[j] == i]
        for seq in cluster_seqs:
            diversity = len(np.unique(seq)) / 5
            diversities.append(diversity)
            cluster_ids.append(i)
    
    df_diversity = pd.DataFrame({'diversity': diversities, 'cluster': cluster_ids})
    sns.boxplot(data=df_diversity, x='cluster', y='diversity', ax=ax2)
    ax2.set_title('各聚类序列多样性')
    ax2.set_xlabel('聚类标签')
    ax2.set_ylabel('多样性指数')
    
    # 3. 稳定性分析
    ax3 = axes[1, 0]
    stabilities = []
    cluster_ids = []
    
    for i in range(n_clusters):
        cluster_seqs = [all_sequences[j] for j in range(len(all_sequences)) if cluster_labels[j] == i]
        for seq in cluster_seqs:
            if len(seq) > 1:
                stability = np.sum(np.array(seq)[1:] == np.array(seq)[:-1]) / (len(seq) - 1)
            else:
                stability = 1.0
            stabilities.append(stability)
            cluster_ids.append(i)
    
    df_stability = pd.DataFrame({'stability': stabilities, 'cluster': cluster_ids})
    sns.violinplot(data=df_stability, x='cluster', y='stability', ax=ax3)
    ax3.set_title('各聚类序列稳定性')
    ax3.set_xlabel('聚类标签')
    ax3.set_ylabel('稳定性指数')
    
    # 4. 类别偏好
    ax4 = axes[1, 1]
    preference_matrix = np.zeros((n_clusters, 5))
    
    for i in range(n_clusters):
        cluster_seqs = [all_sequences[j] for j in range(len(all_sequences)) if cluster_labels[j] == i]
        if cluster_seqs:
            all_freqs = []
            for seq in cluster_seqs:
                freq = [(np.array(seq)==k).sum()/len(seq) for k in range(5)]
                all_freqs.append(freq)
            preference_matrix[i] = np.mean(all_freqs, axis=0)
    
    x = np.arange(5)
    width = 0.8 / n_clusters
    
    for i in range(n_clusters):
        ax4.bar(x + i*width, preference_matrix[i], width, label=f'聚类{i}', alpha=0.8)
    
    ax4.set_xlabel('问答模式类别')
    ax4.set_ylabel('平均频率')
    ax4.set_title('各聚类的问答模式偏好')
    ax4.set_xticks(x + width * (n_clusters-1) / 2)
    ax4.set_xticklabels(['知识型', '练习型', '解释型', '追问型', '闲聊型'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'detailed_cluster_statistics.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ==============================
# 主程序 - 你原有的代码保持不变，只在最后添加聚类和可视化部分
# ==============================

# [你原有的GPT API调用和文件处理代码保持不变]
# ...

# 修改特征提取部分
all_sequences = []
all_vectors = []
dialog_names = []

print(f"开始使用 {method} 方法提取特征...")

# 根据选择的方法进行特征提取
if method == "similarity":
    # 对于相似度方法，先收集所有序列
    for file in os.listdir(output_txt_folder):
        if not file.endswith("_qa_modes.txt"):
            continue
        txt_path = os.path.join(output_txt_folder, file)
        with open(txt_path, "r", encoding="utf-8") as f:
            qa_mode_list_str = f.read()
        
        try:
            qa_seq = parse_qa_mode_list_str(qa_mode_list_str)
            if qa_seq and len(qa_seq) > 0:
                all_sequences.append(qa_seq)
                dialog_names.append(file.replace("_qa_modes.txt", ".csv"))
        except Exception as e:
            print(f"{file} 处理失败: {e}")
            continue
    
    # 使用序列相似度聚类
    cluster_labels = sequence_similarity_clustering(all_sequences, n_clusters)
    X_scaled = None  # 相似度方法不需要特征矩阵
    
elif method == "ngram":
    # 先收集所有序列
    for file in os.listdir(output_txt_folder):
        if not file.endswith("_qa_modes.txt"):
            continue
        txt_path = os.path.join(output_txt_folder, file)
        with open(txt_path, "r", encoding="utf-8") as f:
            qa_mode_list_str = f.read()
        
        try:
            qa_seq = parse_qa_mode_list_str(qa_mode_list_str)
            if qa_seq and len(qa_seq) > 0:
                all_sequences.append(qa_seq)
                dialog_names.append(file.replace("_qa_modes.txt", ".csv"))
        except Exception as e:
            print(f"{file} 处理失败: {e}")
            continue
    
    # 使用N-gram特征
    X, vectorizer = ngram_feature_extraction(all_sequences, max_features=50)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cluster_labels = apply_clustering_algorithm(X_scaled, algorithm, n_clusters)
    
else:
    # 对于原始、增强和混合方法
    for file in os.listdir(output_txt_folder):
        if not file.endswith("_qa_modes.txt"):
            continue
        txt_path = os.path.join(output_txt_folder, file)
        with open(txt_path, "r", encoding="utf-8") as f:
            qa_mode_list_str = f.read()
        
        try:
            qa_seq = parse_qa_mode_list_str(qa_mode_list_str)
            if not qa_seq or len(qa_seq) == 0:
                continue
                
            # 根据方法选择特征提取函数
            if method == "original":
                vec = original_qa_sequence_to_vector(qa_seq, n_classes=5)
            elif method == "enhanced":
                vec = enhanced_qa_sequence_to_vector(qa_seq, n_classes=5)
            elif method == "hybrid":
                # 混合方法：结合原始和增强特征
                vec1 = original_qa_sequence_to_vector(qa_seq, n_classes=5)
                vec2 = enhanced_qa_sequence_to_vector(qa_seq, n_classes=5)
                vec = np.concatenate([vec1, vec2])
            
            all_vectors.append(vec)
            all_sequences.append(qa_seq)
            dialog_names.append(file.replace("_qa_modes.txt", ".csv"))
            
        except Exception as e:
            print(f"{file} 处理失败: {e}")
            continue
    
    # 标准化和聚类
    X = np.vstack(all_vectors)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cluster_labels = apply_clustering_algorithm(X_scaled, algorithm, n_clusters)

# 计算聚类质量
if X_scaled is not None:
    try:
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        print(f"轮廓系数: {silhouette_avg:.4f}")
    except Exception as e:
        print(f"无法计算轮廓系数: {e}")

# 保存聚类结果
df_result = pd.DataFrame({
    "dialog_file": dialog_names, 
    "cluster_label": cluster_labels,
    "sequence_length": [len(seq) for seq in all_sequences]
})
result_filename = f"dialog_cluster_result_{method}_{algorithm}_{n_clusters}.csv"
df_result.to_csv(os.path.join(output_folder, result_filename), index=False, encoding="utf-8-sig")

print(f"聚类结果已保存到: {os.path.join(output_folder, result_filename)}")

# 分析每个聚类的特点
print("\n聚类分析:")
for i in range(max(cluster_labels) + 1):
    cluster_seqs = [all_sequences[j] for j in range(len(all_sequences)) if cluster_labels[j] == i]
    if cluster_seqs:
        avg_length = np.mean([len(seq) for seq in cluster_seqs])
        all_freqs = []
        for seq in cluster_seqs:
            freq = [(np.array(seq)==k).sum()/len(seq) for k in range(5)]
            all_freqs.append(freq)
        avg_freq = np.mean(all_freqs, axis=0)
        
        print(f"聚类 {i}: {len(cluster_seqs)} 个序列")
        print(f"  - 平均长度: {avg_length:.1f}")
        print(f"  - 平均类别分布: {[f'{f:.2f}' for f in avg_freq]}")
        
        if len(cluster_seqs) <= 3:
            for seq in cluster_seqs:
                print(f"    {seq}")
        else:
            for seq in cluster_seqs[:2]:
                print(f"    {seq}")
            print(f"    ... (还有{len(cluster_seqs)-2}个)")
        print()

# 生成可视化
print("开始生成可视化...")

n_actual_clusters = len(np.unique(cluster_labels))

# 如果是相似度方法，创建一个虚拟的特征矩阵用于可视化
if X_scaled is None:
    # 使用简单的统计特征进行可视化
    temp_vectors = []
    for seq in all_sequences:
        temp_vec = original_qa_sequence_to_vector(seq, n_classes=5)
        temp_vectors.append(temp_vec)
    X_scaled = StandardScaler().fit_transform(np.vstack(temp_vectors))

comprehensive_clustering_visualization(X_scaled, cluster_labels, all_sequences, dialog_names, n_actual_clusters, output_folder)
plot_sequence_patterns(all_sequences, cluster_labels, n_actual_clusters, output_folder, max_examples=5)
plot_cluster_transitions(all_sequences, cluster_labels, n_actual_clusters, output_folder)
plot_detailed_statistics(all_sequences, cluster_labels, n_actual_clusters, output_folder)

print(f"\n所有可视化已完成！结果保存在文件夹: {output_folder}")
print("生成的文件：")
print(f"- {result_filename}: 聚类结果")
print("- comprehensive_clustering_analysis.png: 综合分析")
print("- sequence_patterns_by_cluster.png: 序列模式")
print("- cluster_transition_patterns.png: 转移模式")
print("- detailed_cluster_statistics.png: 详细统计")

# 保存方法配置信息
config_info = {
    "method": method,
    "algorithm": algorithm,
    "n_clusters": n_clusters,
    "n_sequences": len(all_sequences),
    "feature_dim": X_scaled.shape[1] if X_scaled is not None else "N/A",
    "silhouette_score": silhouette_avg if 'silhouette_avg' in locals() else "N/A"
}

config_filename = os.path.join(output_folder, "config_info.txt")
with open(config_filename, "w", encoding="utf-8") as f:
    for key, value in config_info.items():
        f.write(f"{key}: {value}\n")

print(f"配置信息已保存到: {config_filename}")