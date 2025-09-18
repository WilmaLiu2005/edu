import pandas as pd
import os
import csv
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# 文件路径
# -----------------------------
input_csv = "/Users/vince/undergraduate/KEG/edu/Data/filtered_dialog.csv"   # 已处理好的 CSV
output_folder = "/Users/vince/undergraduate/KEG/edu/Data/轮次分类CSV_similarity"       # 输出文件夹

# -----------------------------
# 创建输出文件夹
# -----------------------------
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# 读取 CSV
# -----------------------------
df = pd.read_csv(
    input_csv,
    sep=",",
    quotechar='"',
    doublequote=True,
    encoding="utf-8-sig",
    engine="python",
    keep_default_na=False
)

# 确保列名没有空格或不可见字符
df.columns = df.columns.str.strip()

# -----------------------------
# 加载 embedding 模型
# -----------------------------
model = SentenceTransformer("/Users/vince/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1")

# -----------------------------
# 分割轮次逻辑（相似度）
# -----------------------------
similarity_threshold = 0.5

for student_id, student_group in df.groupby('学生ID'):
    student_group_sorted = student_group.sort_values(by="提问时间").copy()
    student_group_sorted.reset_index(drop=True, inplace=True)
    
    # 生成 embeddings
    questions = student_group_sorted["提问内容"].astype(str).tolist()
    embeddings = model.encode(questions, convert_to_tensor=True)
    
    # 初始化
    session_index = 1
    session_rows = [student_group_sorted.iloc[0]]  # 第一条问题开始新轮次
    
    for i in range(1, len(student_group_sorted)):
        sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
        row = student_group_sorted.iloc[i]
        
        if sim > similarity_threshold:
            # 与上一个问题相似 → 同一轮次
            session_rows.append(row)
        else:
            # 相似度低 → 输出当前轮次，开启新轮次
            session_df = pd.DataFrame(session_rows)
            session_df['行序号'] = range(1, len(session_df)+1)
            output_csv = os.path.join(output_folder, f"{student_id}_{session_index}.csv")
            session_df.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
            print(f"✅ 已生成 {output_csv}，共 {len(session_df)} 行")
            
            session_index += 1
            session_rows = [row]
    
    # 输出最后一轮
    if session_rows:
        session_df = pd.DataFrame(session_rows)
        session_df['行序号'] = range(1, len(session_df)+1)
        output_csv = os.path.join(output_folder, f"{student_id}_{session_index}.csv")
        session_df.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        print(f"✅ 已生成 {output_csv}，共 {len(session_df)} 行")

print("🎉 所有学生轮次分类完成！")
