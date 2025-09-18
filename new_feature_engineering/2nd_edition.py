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

MAX_WORKERS = 50
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 3
API_KEY = "sk-0jErqj61bIYM135CEqhfj318rKIM1TIa"  # 填写你的 API key
BASE_URL = "https://api-gateway.glm.ai/v1"
MODEL_NAME = "gemini-2.5-flash"  # 或你自己的模型

parser = argparse.ArgumentParser()
parser.add_argument("--n_clusters", type=int, default=5, help="簇数")
args = parser.parse_args()
n_clusters = args.n_clusters

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

# -----------------------------
# 将问答模式序列转固定向量（32维）
# -----------------------------
def qa_sequence_to_vector(seq, n_classes=5):
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

# -----------------------------
# 文件夹批量处理
# -----------------------------
input_folder = "/Users/vince/undergraduate/KEG/edu/full_split"  # CSV 文件夹
output_txt_folder = "/Users/vince/undergraduate/KEG/edu/full_split/qa_mode_txts"
os.makedirs(output_txt_folder, exist_ok=True)

from concurrent.futures import ThreadPoolExecutor, as_completed
'''
# 并发处理一个 CSV 文件
def process_csv(file, input_folder, output_txt_folder, model_name):
    if not file.endswith(".csv"):
        return file, None  # 非 CSV，返回 None

    csv_path = os.path.join(input_folder, file)
    txt_filename = os.path.splitext(file)[0] + "_qa_modes.txt"
    txt_path = os.path.join(output_txt_folder, txt_filename)

    # 如果已经存在分类结果，跳过
    if os.path.exists(txt_path):
        print(f"{file} 已存在分类结果，跳过处理")
        return file, "skipped"

    try:
        # 读取 CSV
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df.fillna("", inplace=True)
        questions = df["提问内容"].tolist()

        # prompt 构造
        system_prompt = {
            "role": "system",
            "content": (
                "你是一个教育场景问答分类助手，将每个问题按照问答模式分类。"
                "模式类别编号如下：0=知识型, 1=练习型, 2=解释型, 3=追问型, 4=闲聊型。"
                "请只输出一个列表，列表中每个元素是对应问题的类别编号，不要解释或额外文本。"
            )
        }
        user_prompt_text = "下面是一个对话的问题列表，请按顺序输出对应的问答模式编号列表：\n"
        for idx, q in enumerate(questions, 1):
            user_prompt_text += f"{idx}. {q}\n"
        user_prompt = {"role": "user", "content": user_prompt_text}

        # 调 GPT 分类
        qa_mode_list_str = gpt_api_call([system_prompt, user_prompt], model=model_name)
        print(f"{file} 分类结果: {qa_mode_list_str}")

        if qa_mode_list_str is None:
            print(f"{file} 处理失败")
            return file, "failed"

        # 保存结果
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(qa_mode_list_str)
        print(f"{file} 问答模式已保存到 {txt_path}")

        return file, "success"

    except Exception as e:
        print(f"{file} 处理异常: {e}")
        return file, "error"

def main(input_folder, output_txt_folder, model_name, max_workers=5):
    files = os.listdir(input_folder)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_csv, file, input_folder, output_txt_folder, model_name): file
            for file in files
            if file.endswith(".csv")
        }
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                file, status = future.result()
                results.append((file, status))
            except Exception as e:
                print(f"{file} 线程执行异常: {e}")
                results.append((file, "error"))

    print("\n=== 总结 ===")
    for file, status in results:
        print(f"{file}: {status}")

main(input_folder, output_txt_folder, MODEL_NAME, max_workers=MAX_WORKERS)
'''
# -----------------------------
# 从所有 txt 文件读取并向量化
# -----------------------------
import ast
import re
import json
all_vectors = []
dialog_names = []

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

# ----------------- 主处理循环（替换你原来的解析逻辑） -----------------
for file in os.listdir(output_txt_folder):
    if not file.endswith("_qa_modes.txt"):
        continue
    txt_path = os.path.join(output_txt_folder, file)

    with open(txt_path, "r", encoding="utf-8") as f:
        qa_mode_list_str = f.read()

    try:
        qa_seq = parse_qa_mode_list_str(qa_mode_list_str)
        if not isinstance(qa_seq, list) or not all(isinstance(i, int) for i in qa_seq):
            raise ValueError("解析后不是整数列表")
        if len(qa_seq) == 0:
            # 解析为空，按你的意愿也可以删除，这里选择删除并继续
            print(f"{file} 解析为空，删除文件")
            try:
                os.remove(txt_path)
            except Exception as e:
                print(f"删除 {file} 失败: {e}")
            continue
    except Exception as e:
        # 解析失败 -> 删除出错文件（按你的要求）
        print(f"{file} 问答模式解析失败（{e}），删除文件: {txt_path}")
        try:
            os.remove(txt_path)
        except Exception as e2:
            print(f"删除 {file} 失败: {e2}")
        continue

    # 下面保持原有处理逻辑
    vec = qa_sequence_to_vector(qa_seq)
    all_vectors.append(vec)
    dialog_names.append(file.replace("_qa_modes.txt", ".csv"))

# -----------------------------
# KMeans 聚类
# -----------------------------
X = np.vstack(all_vectors)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

df_result = pd.DataFrame({"dialog_file": dialog_names, "cluster_label": cluster_labels})
df_result.to_csv("dialog_cluster_result_LLM.csv", index=False, encoding="utf-8-sig")
print("聚类结果已保存：dialog_cluster_result_LLM.csv")

# -----------------------------
# 聚类可视化
# -----------------------------
dimred_method = "pca"  # 可改 "tsne"
if dimred_method == "pca":
    reducer = PCA(n_components=2)
    X_2d = reducer.fit_transform(X_scaled)
elif dimred_method == "tsne":
    reducer = TSNE(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="tab10", s=50)
plt.title("Dialog Clustering (KMeans)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar(scatter, label="Cluster Label")
plt.tight_layout()
plt.savefig("llm_feature_cluster_visualization.png", dpi=300)
plt.show()
print("可视化已保存到 llm_feature_cluster_visualization.png")

# -----------------------------
# 轮廓系数
# -----------------------------
try:
    score = silhouette_score(X_scaled, cluster_labels)
    print(f"Silhouette Score: {score:.4f}")
except Exception as e:
    print("无法计算 Silhouette Score:", e)