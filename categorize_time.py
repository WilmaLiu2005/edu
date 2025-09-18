import pandas as pd
import os
import csv
from datetime import timedelta

# -----------------------------
# 文件夹路径
# -----------------------------
input_folder = "/Users/vince/undergraduate/KEG/edu/Data/pre_validation5"   # 输入文件夹
output_folder = "/Users/vince/undergraduate/KEG/edu/time15"  # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# 分割轮次参数
# -----------------------------
time_col = '提问时间'
time_threshold = timedelta(minutes=5)  # 超过30分钟就分新轮次

# -----------------------------
# 遍历文件夹里的 CSV
# -----------------------------
for file_name in os.listdir(input_folder):
    if not file_name.endswith(".csv"):
        continue
    
    input_csv = os.path.join(input_folder, file_name)
    df = pd.read_csv(input_csv, sep=",", quotechar='"', encoding="utf-8-sig", engine="python", keep_default_na=False)
    
    # 清理列名
    df.columns = df.columns.str.strip()
    
    # 时间列转 datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # 按学生分组
    for student_id, student_group in df.groupby('学生ID'):
        student_group_sorted = student_group.sort_values(by=time_col).copy()
        student_group_sorted.reset_index(drop=True, inplace=True)
        
        # 初始化
        session_index = 1
        session_rows = []
        previous_time = None
        
        for idx, row in student_group_sorted.iterrows():
            current_time = row[time_col]

            if previous_time is None:
                # 第一行
                session_rows = [row]
            else:
                if current_time - previous_time > time_threshold:
                    # 输出当前轮次
                    session_df = pd.DataFrame(session_rows)
                    # session_df['行序号'] = range(1, len(session_df)+1)
                    base_name, ext = os.path.splitext(file_name)
                    output_csv = os.path.join(output_folder, f"{student_id}_{session_index}{ext}")
                    session_df.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
                    print(f"已生成 {output_csv}，共 {len(session_df)} 行")
                    
                    # 开启新轮次
                    session_index += 1
                    session_rows = [row]
                else:
                    session_rows.append(row)

            previous_time = current_time

        # 输出最后一轮
        if session_rows:
            session_df = pd.DataFrame(session_rows)
            # session_df['行序号'] = range(1, len(session_df)+1)
            base_name, ext = os.path.splitext(file_name)
            output_csv = os.path.join(output_folder, f"{student_id}_{session_index}{ext}")
            session_df.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
            print(f"已生成 {output_csv}，共 {len(session_df)} 行")

print("所有 CSV 文件轮次分类完成！")
