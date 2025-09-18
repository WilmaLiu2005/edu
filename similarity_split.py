import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# 配置
# -------------------------------
folder_path = "/Users/vince/undergraduate/KEG/edu/Data/pre_validation4"  # CSV文件夹路径
output_folder = "/Users/vince/undergraduate/KEG/edu/similarity08"  # 输出文件夹
os.makedirs(output_folder, exist_ok=True)
model_name = "/Users/vince/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1"
similarity_threshold = 0.8  # 相似度大于0.3认为是同一轮对话

# -------------------------------
# 初始化模型
# -------------------------------
model = SentenceTransformer(model_name)

# -------------------------------
# 遍历CSV文件
# -------------------------------
for file in os.listdir(folder_path):
    if not file.endswith(".csv"):
        continue

    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df = df.sort_values("提问时间").reset_index(drop=True)
    questions = df["提问内容"].astype(str).tolist()

    if len(questions) == 0:
        continue

    # 生成向量
    embeddings = model.encode(questions, convert_to_tensor=True)

    # -------------------------------
    # 按语义相似度划分对话
    # -------------------------------
    sessions = []
    current_session = [0]  # 第一个问题属于第一轮

    for i in range(1, len(questions)):
        sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
        if sim >= similarity_threshold:
            current_session.append(i)
        else:
            sessions.append(current_session)
            current_session = [i]

    if current_session:
        sessions.append(current_session)

    # -------------------------------
    # 导出每轮对话为CSV
    # -------------------------------
    base_name, ext = os.path.splitext(file)
    # base_name_clean = base_name.rsplit("_", 1)[0]
    for idx, session_indices in enumerate(sessions, start=1):
        session_df = df.iloc[session_indices].copy()
        save_path = os.path.join(output_folder, f"{base_name}_{idx}{ext}")
        session_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"{file} -> {save_path} 已保存，行数：{len(session_df)}")
