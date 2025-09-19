import os
import argparse
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 如果是macOS，优先使用系统字体
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
elif platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
else:  # Linux
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei']
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
# 加载班级信息和题目明细
# -----------------------------
class_info_path = "/Users/vince/undergraduate/KEG/edu/report4/Data/学堂在线数据研究20250812/班级情况.csv"
homework_path = "/Users/vince/undergraduate/KEG/edu/report4/Data/学堂在线数据研究20250812/题目明细.csv"

# 读取班级信息
try:
    df_class = pd.read_csv(class_info_path, encoding="utf-8-sig")
    print(f"读取班级信息：{len(df_class)} 条记录")
except Exception as e:
    print(f"读取班级信息失败: {e}")
    df_class = pd.DataFrame()

# 读取题目明细
try:
    df_homework = pd.read_csv(homework_path, encoding="utf-8-sig")
    print(f"读取题目明细：{len(df_homework)} 条记录")
    # 转换发布时间为datetime格式
    df_homework['发布时间'] = pd.to_datetime(df_homework['发布时间'], errors='coerce')
    print(f"有效发布时间记录：{df_homework['发布时间'].notna().sum()} 条")
except Exception as e:
    print(f"读取题目明细失败: {e}")
    df_homework = pd.DataFrame()

# -----------------------------
# 特征提取函数
# -----------------------------
def extract_features(file_path, file_name):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.fillna("", inplace=True)
    
    # 获取教学班ID
    if df.empty:
        return None
    class_id = df["教学班ID"].iloc[0]
    
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
    
    # 新特征1：这门课留了几次作业和考试
    total_assignments = 0
    if not df_class.empty:
        class_info = df_class[df_class["教学班ID"] == class_id]
        if not class_info.empty:
            homework_count = class_info["布置作业个数"].iloc[0] if "布置作业个数" in class_info.columns else 0
            exam_count = class_info["布置考试个数"].iloc[0] if "布置考试个数" in class_info.columns else 0
            total_assignments = homework_count + exam_count
    
    # 新特征2：距离下一次作业截止的时间（天数）
    days_to_next_deadline = float('inf')  # 默认值：无穷大（表示没有即将到来的作业）
    
    if not df_homework.empty:
        # 筛选该班级的作业/考试
        class_homework = df_homework[df_homework["教学班ID"] == class_id]
        
        if not class_homework.empty and '提交截止时间' in class_homework.columns:
            # 获取有效的截止时间
            deadline_times = pd.to_datetime(class_homework['提交截止时间'], errors='coerce').dropna()
            
            if not deadline_times.empty:
                # 计算对话期间距离下一次作业截止的最短时间
                min_days_to_deadline = []
                
                for qa_time in df["提问时间"]:
                    # 找到在对话时间之后的所有截止时间
                    future_deadlines = deadline_times[deadline_times > qa_time]
                    
                    if not future_deadlines.empty:
                        # 找到最近的一个截止时间
                        next_deadline = future_deadlines.min()
                        days_diff = (next_deadline - qa_time).total_seconds() / (24 * 3600)
                        min_days_to_deadline.append(days_diff)
                    else:
                        # 如果没有未来的截止时间，查看是否有刚过期的（可能还在延期内）
                        recent_deadlines = deadline_times[deadline_times <= qa_time]
                        if not recent_deadlines.empty:
                            latest_deadline = recent_deadlines.max()
                            days_diff = (qa_time - latest_deadline).total_seconds() / (24 * 3600)
                            # 如果刚过期不久（比如3天内），用负数表示
                            if days_diff <= 3:
                                min_days_to_deadline.append(-days_diff)
                
                if min_days_to_deadline:
                    days_to_next_deadline = np.mean(min_days_to_deadline)
                    print(f"文件 {file_name}, 班级 {class_id}: 平均距离下次作业截止 {days_to_next_deadline:.2f} 天")
    
    # 如果距离太远（比如超过30天），可能数据有问题，设为一个合理的上限
    if days_to_next_deadline == float('inf') or days_to_next_deadline > 30:
        days_to_next_deadline = 30  # 或者设为 NaN
    
    return {
        "file": file_name,
        "class_id": class_id,
        "qa_turns": qa_turns,
        "avg_q_len": avg_q_len,
        "total_time": total_time,
        "non_class_ratio": non_class_ratio,
        # "total_assignments": total_assignments,
        # "days_to_next_deadline": days_to_next_deadline,  # 新特征名
    }

# -----------------------------
# 特征重要性分析函数
# -----------------------------
def analyze_feature_importance(X_scaled, cluster_labels, feature_names):
    """分析各特征对聚类的重要性"""
    print("\n" + "="*60)
    print("特征重要性分析")
    print("="*60)
    
    # 过滤掉噪声点
    valid_mask = cluster_labels != -1
    X_valid = X_scaled[valid_mask]
    labels_valid = cluster_labels[valid_mask]
    
    if len(set(labels_valid)) < 2:
        print("聚类结果不足以进行特征重要性分析")
        return
    
    # 方法1: 计算每个特征在不同簇间的方差
    print("\n1. 特征簇间差异分析（方差比）:")
    feature_importance_variance = []
    
    for i, feature_name in enumerate(feature_names):
        feature_values = X_valid[:, i]
        
        # 计算簇间方差和簇内方差
        cluster_means = []
        cluster_vars = []
        
        for cluster_id in set(labels_valid):
            cluster_mask = labels_valid == cluster_id
            cluster_data = feature_values[cluster_mask]
            if len(cluster_data) > 0:
                cluster_means.append(np.mean(cluster_data))
                cluster_vars.append(np.var(cluster_data))
        
        # 簇间方差
        between_var = np.var(cluster_means) if len(cluster_means) > 1 else 0
        # 平均簇内方差
        within_var = np.mean(cluster_vars) if cluster_vars else 1e-8
        # 方差比（越大越重要）
        variance_ratio = between_var / (within_var + 1e-8)
        
        feature_importance_variance.append(variance_ratio)
        print(f"  {feature_name:20}: {variance_ratio:.4f}")
    
    # 方法2: 单独移除每个特征后的聚类效果变化
    print("\n2. Leave-One-Out 特征重要性分析:")
    baseline_score = silhouette_score(X_valid, labels_valid) if len(set(labels_valid)) > 1 else 0
    print(f"  基线轮廓系数: {baseline_score:.4f}")
    
    feature_importance_loo = []
    for i, feature_name in enumerate(feature_names):
        # 移除第i个特征
        X_without_feature = np.delete(X_valid, i, axis=1)
        
        # 重新聚类
        if args.algorithm == "kmeans":
            temp_model = KMeans(n_clusters=args.n_clusters, random_state=42)
        elif args.algorithm == "agg":
            temp_model = AgglomerativeClustering(n_clusters=args.n_clusters)
        elif args.algorithm == "spectral":
            temp_model = SpectralClustering(n_clusters=args.n_clusters, affinity="nearest_neighbors", random_state=42)
        else:
            # 对于DBSCAN和HDBSCAN，使用kmeans作为替代
            temp_model = KMeans(n_clusters=args.n_clusters, random_state=42)
        
        try:
            temp_labels = temp_model.fit_predict(X_without_feature)
            if len(set(temp_labels)) > 1:
                temp_score = silhouette_score(X_without_feature, temp_labels)
                importance = baseline_score - temp_score  # 移除后分数下降越多，特征越重要
            else:
                importance = baseline_score  # 如果移除后无法聚类，说明特征很重要
        except:
            importance = baseline_score
        
        feature_importance_loo.append(importance)
        print(f"  移除 {feature_name:20}: 轮廓系数 = {temp_score:.4f}, 重要性 = {importance:.4f}")
    
    # 方法3: 计算特征与聚类标签的互信息
    print("\n3. 互信息分析:")
    feature_importance_mi = []
    for i, feature_name in enumerate(feature_names):
        mi = mutual_info_regression(X_valid[:, i].reshape(-1, 1), labels_valid, random_state=42)[0]
        feature_importance_mi.append(mi)
        print(f"  {feature_name:20}: {mi:.4f}")
    
    # 方法4: PCA贡献度分析
    print("\n4. PCA 主成分贡献度分析:")
    pca = PCA(n_components=min(len(feature_names), len(X_valid)))
    pca.fit(X_valid)
    
    # 计算每个特征对前两个主成分的贡献
    components = pca.components_[:2]  # 前两个主成分
    explained_variance = pca.explained_variance_ratio_[:2]
    
    feature_importance_pca = []
    for i, feature_name in enumerate(feature_names):
        # 加权贡献度（按解释方差加权）
        contribution = sum(abs(components[j][i]) * explained_variance[j] for j in range(len(components)))
        feature_importance_pca.append(contribution)
        print(f"  {feature_name:20}: {contribution:.4f}")
    
    # 综合排名
    print("\n5. 特征重要性综合排名:")
    print("-" * 80)
    
    # 标准化各种重要性分数
    def normalize_scores(scores):
        scores = np.array(scores)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    norm_variance = normalize_scores(feature_importance_variance)
    norm_loo = normalize_scores(feature_importance_loo)
    norm_mi = normalize_scores(feature_importance_mi)
    norm_pca = normalize_scores(feature_importance_pca)
    
    # 综合分数（等权重平均）
    comprehensive_scores = (norm_variance + norm_loo + norm_mi + norm_pca) / 4
    
    # 排序
    importance_ranking = sorted(zip(feature_names, comprehensive_scores, 
                                  feature_importance_variance, feature_importance_loo,
                                  feature_importance_mi, feature_importance_pca), 
                               key=lambda x: x[1], reverse=True)
    
    print(f"{'排名':<4} {'特征名称':<20} {'综合分数':<10} {'方差比':<10} {'LOO':<10} {'互信息':<10} {'PCA':<10}")
    print("-" * 80)
    for rank, (name, comp_score, var_score, loo_score, mi_score, pca_score) in enumerate(importance_ranking, 1):
        print(f"{rank:<4} {name:<20} {comp_score:.4f}   {var_score:.4f}   {loo_score:.4f}   {mi_score:.4f}   {pca_score:.4f}")
    
    print(f"\n✨ 最重要的特征: {importance_ranking[0][0]}")
    print(f"✨ 最不重要的特征: {importance_ranking[-1][0]}")
    
    return importance_ranking

# -----------------------------
# 可视化特征重要性
# -----------------------------
def plot_feature_importance(importance_ranking, algorithm, n_clusters):
    """绘制特征重要性图"""
    feature_names = [item[0] for item in importance_ranking]
    comprehensive_scores = [item[1] for item in importance_ranking]
    
    plt.figure(figsize=(15, 10))
    
    # 主图：综合重要性分数
    plt.subplot(2, 3, 1)  # 改为2x3布局
    bars = plt.bar(range(len(feature_names)), comprehensive_scores, color='skyblue', alpha=0.7)
    plt.title('特征重要性综合排名')
    plt.xlabel('特征')
    plt.ylabel('综合重要性分数')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 子图：各种方法的重要性分数
    methods = ['方差比', 'Leave-One-Out', '互信息', 'PCA贡献']
    score_indices = [2, 3, 4, 5]  # 对应importance_ranking中的索引
    
    for idx, (method, score_idx) in enumerate(zip(methods, score_indices)):
        plt.subplot(2, 3, idx + 2)  # 现在idx + 2的范围是2-5，都在1-6范围内
        scores = [item[score_idx] for item in importance_ranking]
        plt.bar(range(len(feature_names)), scores, alpha=0.7)
        plt.title(f'{method}重要性')
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    
    plt.tight_layout()
    
    importance_fig_path = f"feature_importance_{algorithm}_{n_clusters}.png"
    plt.savefig(importance_fig_path, dpi=300, bbox_inches='tight')
    print(f"\n特征重要性图已保存到 {importance_fig_path}")

# -----------------------------
# 读取对话文件 & 提取特征
# -----------------------------
features = []
processed_files = 0
error_files = 0

for file in os.listdir(args.input):
    if file.endswith(".csv"):
        path = os.path.join(args.input, file)
        try:
            feats = extract_features(path, file)
            if feats is not None:
                features.append(feats)
                processed_files += 1
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            error_files += 1

print(f"\n文件处理完成:")
print(f"成功处理: {processed_files} 个文件")
print(f"处理失败: {error_files} 个文件")

df_features = pd.DataFrame(features)

if df_features.empty:
    print("没有成功处理的文件，程序退出")
    exit()

# 显示特征统计信息
feature_columns = ["qa_turns", "avg_q_len", "total_time", "non_class_ratio"]
feature_names_cn = ["QA轮次", "平均问题长度", "总耗时(分钟)", "非班级入口比例"]

print(f"\n特征统计信息 (共 {len(df_features)} 个对话):")
print(df_features[feature_columns].describe())

# -----------------------------
# 特征矩阵 (修复版本)
# -----------------------------
X = df_features[feature_columns].values

# 更彻底地处理异常值
X = np.nan_to_num(X, nan=0.0, posinf=30.0, neginf=-30.0)

# 检查是否还有异常值
print(f"数据中是否还有NaN: {np.isnan(X).any()}")
print(f"数据中是否还有无穷值: {np.isinf(X).any()}")

# 如果还有问题，进一步清理
finite_mask = np.isfinite(X).all(axis=1)
if not finite_mask.all():
    print(f"发现 {(~finite_mask).sum()} 行数据有异常值，已移除")
    X = X[finite_mask]
    df_features = df_features[finite_mask].reset_index(drop=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 再次检查标准化后的数据
print(f"标准化后数据形状: {X_scaled.shape}")
print(f"标准化后是否有异常值: {np.isnan(X_scaled).any() or np.isinf(X_scaled).any()}")

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

# 显示聚类结果
unique_labels = set(cluster_labels)
noise_points = (cluster_labels == -1).sum() if -1 in unique_labels else 0

print(f"\n聚类完成:")
print(f"总簇数: {len(unique_labels)}")
print(f"噪声点: {noise_points}")

# -----------------------------
# 特征重要性分析
# -----------------------------
importance_ranking = analyze_feature_importance(X_scaled, cluster_labels, feature_names_cn)

# -----------------------------
# 聚类评估
# -----------------------------
if args.eval:
    valid_mask = cluster_labels != -1
    if valid_mask.sum() > 1 and len(set(cluster_labels[valid_mask])) > 1:
        try:
            score = silhouette_score(X_scaled[valid_mask], cluster_labels[valid_mask])
            print(f"\nSilhouette Score: {score:.4f}")
        except Exception as e:
            print(f"无法计算 Silhouette Score: {e}")
    else:
        print("\n聚类簇数不足，无法计算 Silhouette Score")

# -----------------------------
# 保存结果
# -----------------------------
output = f"feature_cluster_{args.algorithm}_{args.n_clusters}.csv"
df_features.to_csv(output, index=False, encoding="utf-8-sig")
print(f"\n聚类结果已保存到 {output}")

# 保存特征重要性结果
importance_df = pd.DataFrame(importance_ranking, 
                           columns=['特征名称', '综合重要性', '方差比重要性', 'LOO重要性', '互信息重要性', 'PCA重要性'])
importance_output = f"feature_importance_{args.algorithm}_{args.n_clusters}.csv"
importance_df.to_csv(importance_output, index=False, encoding="utf-8-sig")
print(f"特征重要性分析结果已保存到 {importance_output}")

# -----------------------------
# 可视化
# -----------------------------
if args.visualize:
    # 聚类可视化
    reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=cluster_labels, cmap="tab10", s=50, alpha=0.7)
    plt.title(f"Dialog Feature Clustering ({args.algorithm}) - 6 Features\nSamples: {len(df_features)}, Clusters: {len(unique_labels)}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter)
    
    fig_path = f"/Users/vince/undergraduate/KEG/edu/1st_edition_cluster_Figure_1_{args.n_clusters}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"聚类可视化已保存到 {fig_path}")
    
    # 特征重要性可视化
    plot_feature_importance(importance_ranking, args.algorithm, args.n_clusters)
    
    plt.show()