import os
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def load_clusters_from_dir(folder):
    """
    从一个目录加载对话划分结果
    返回一个 dict: {学生ID: pd.Series(index=行序号, values=对话编号)}
    文件按后缀数字排序，保证 conv_id 按时间顺序。
    """
    clusters = {}
    
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    files.sort(key=lambda x: int(x.rsplit("_", 1)[1].replace(".csv","")))
    
    for fname in files:
        parts = fname.rsplit("_", 1)
        student_id = parts[0]
        conv_id = int(parts[1].replace(".csv", ""))
        
        path = os.path.join(folder, fname)
        df = pd.read_csv(path)
        if "行序号" not in df.columns:
            raise ValueError(f"{fname} 缺少 '行序号' 列")
        df["行序号"] = df["行序号"].astype(int)
        
        labels = pd.Series(conv_id, index=df["行序号"])
        if student_id not in clusters:
            clusters[student_id] = labels
        else:
            clusters[student_id] = pd.concat(
                [clusters[student_id], labels[~labels.index.isin(clusters[student_id].index)]]
            )
    return clusters

def row_based_accuracy(mapping_true, mapping_pred):
    """
    基于行序号对齐计算 Accuracy
    """
    common_indices = set(mapping_true.index) & set(mapping_pred.index)
    if not common_indices:
        return 0.0
    common_indices = sorted(common_indices)
    
    correct = sum(mapping_true.loc[common_indices] == mapping_pred.loc[common_indices])
    return correct / len(common_indices)

def compare_clusters(clusters1, clusters2):
    """
    输入两个划分结果，输出 ARI / NMI / Jaccard / Accuracy（基于行序号）
    """
    results = []
    for sid in set(clusters1.keys()) & set(clusters2.keys()):
        s1 = clusters1[sid]
        s2 = clusters2[sid]
        common_index = s1.index.intersection(s2.index)
        if len(common_index) == 0:
            continue
        
        y1 = s1.loc[common_index].values
        y2 = s2.loc[common_index].values
        
        ari = adjusted_rand_score(y1, y2)
        nmi = normalized_mutual_info_score(y1, y2)
        
        sets1 = [set(s1[s1 == c].index) for c in pd.unique(y1)]
        sets2 = [set(s2[s2 == c].index) for c in pd.unique(y2)]
        jaccs = []
        for a in sets1:
            best = max((len(a & b) / len(a | b) for b in sets2), default=0)
            jaccs.append(best)
        jaccard_mean = sum(jaccs) / len(jaccs) if jaccs else 0

        acc = row_based_accuracy(s1, s2)
        results.append((sid, ari, nmi, jaccard_mean, acc))
    
    return pd.DataFrame(results, columns=["学生ID", "ARI", "NMI", "Jaccard", "Accuracy"])


# =====================
# 用法示例
# =====================
folder1 = "/Users/vince/undergraduate/KEG/edu/test_split_reverse"
folder2 = "/Users/vince/undergraduate/KEG/edu/real_GT_split"

clusters1 = load_clusters_from_dir(folder1)
clusters2 = load_clusters_from_dir(folder2)

df_result = compare_clusters(clusters1, clusters2)
print(df_result.head())

print("整体平均：")
print(df_result[["ARI", "NMI", "Jaccard", "Accuracy"]].mean())
