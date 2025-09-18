import os
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score, recall_score, f1_score

def load_clusters_from_dir(folder):
    """
    从一个目录加载对话簇划分结果
    返回一个 dict: {学生ID: pd.Series(index=行序号, values=对话编号)}
    文件按后缀数字排序，保证 conv_id 按时间顺序。
    """
    clusters = {}
    
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    files.sort(key=lambda x: int(x.rsplit("_", 1)[1].replace(".csv","")))
    
    for fname in files:
        parts = fname.rsplit("_", 1)
        student_id = parts[0]
        conv_id = int(parts[1].replace(".csv", ""))  # 对话编号
        
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

def hungarian_accuracy(y_true, y_pred):
    """
    使用匈牙利算法对齐聚类标签，然后计算 Accuracy / Precision / Recall / F1-score
    """
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    
    # 构建代价矩阵（负数，因为 linear_sum_assignment 是最小化）
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
    for i, lt in enumerate(labels_true):
        for j, lp in enumerate(labels_pred):
            cost_matrix[i, j] = -np.sum((y_true == lt) & (y_pred == lp))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 重新映射预测标签
    mapping = {labels_pred[j]: labels_true[i] for i, j in zip(row_ind, col_ind)}
    y_pred_mapped = np.array([mapping.get(l, l) for l in y_pred])
    
    # 计算指标
    acc = np.mean(y_pred_mapped == y_true)
    prec = precision_score(y_true, y_pred_mapped, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred_mapped, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred_mapped, average="macro", zero_division=0)
    
    return acc, prec, rec, f1

def compare_clusters_hungarian(clusters1, clusters2):
    """
    输入两个划分结果，输出每个学生的 Accuracy / Precision / Recall / F1-score
    以及学生平均和全局指标
    """
    results = []
    total_y_true = []
    total_y_pred = []

    common_sids = set(clusters1.keys()) & set(clusters2.keys())

    for sid in common_sids:
        s1 = clusters1[sid]
        s2 = clusters2[sid]

        # 对齐相同的行
        common_index = s1.index.intersection(s2.index)
        if len(common_index) == 0:
            continue

        y_true = s1.loc[common_index].values
        y_pred = s2.loc[common_index].values

        acc, prec, rec, f1 = hungarian_accuracy(y_true, y_pred)
        results.append((sid, acc, prec, rec, f1))

        total_y_true.extend(y_true)
        total_y_pred.extend(y_pred)

    df = pd.DataFrame(results, columns=["学生ID", "Accuracy", "Precision", "Recall", "F1-score"])

    # 学生平均指标
    student_mean = df[["Accuracy", "Precision", "Recall", "F1-score"]].mean()

    # 全局指标
    global_acc, global_prec, global_rec, global_f1 = hungarian_accuracy(np.array(total_y_true), np.array(total_y_pred))

    return df, student_mean, ((float)(global_acc), global_prec, global_rec, global_f1)


# 示例用法
folder1 = "/Users/vince/undergraduate/KEG/edu/time15"
folder2 = "/Users/vince/undergraduate/KEG/edu/real_GT_split"

clusters1 = load_clusters_from_dir(folder1)
clusters2 = load_clusters_from_dir(folder2)

df_result, student_mean, global_metrics = compare_clusters_hungarian(clusters1, clusters2)

# print("每个学生指标：")
# print(df_result.head())

print("\n学生平均指标：")
print(student_mean)

print("\n全局指标 (Accuracy, Precision, Recall, F1):")
print(global_metrics)
