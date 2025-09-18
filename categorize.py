import pandas as pd
import os
import csv
from datetime import timedelta

# -----------------------------
# 文件路径
# -----------------------------
input_csv = "/Users/vince/undergraduate/KEG/edu/Data/filtered_dialog_masked.csv"   # 已处理好的 CSV
output_folder = "/Users/vince/undergraduate/KEG/edu/Data/轮次分类CSV_original"       # 输出文件夹

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
# 时间列转 datetime
# -----------------------------
time_col = '提问时间'
df[time_col] = pd.to_datetime(df[time_col])

# -----------------------------
# 分割轮次逻辑
# -----------------------------
# time_threshold = timedelta(minutes=30)  # 超过30分钟就分新轮次

for student_id, student_group in df.groupby('学生ID'):
    student_group_sorted = student_group.sort_values(by=time_col).copy()
    student_group_sorted.reset_index(drop=True, inplace=True)
    
    # 初始化
    session_index = 1
    session_rows = []
    previous_time = None
    
    for idx, row in student_group_sorted.iterrows():
        current_time = row[time_col]
        session_rows.append(row)
        '''
        if previous_time is None:
            # 第一行
            session_rows.append(row)
        else:
            if current_time - previous_time > time_threshold:
                # 时间间隔超过阈值 -> 输出当前轮次 CSV
                session_df = pd.DataFrame(session_rows)
                session_df['行序号'] = range(1, len(session_df)+1)
                output_csv = os.path.join(output_folder, f"{student_id}_{session_index}.csv")
                session_df.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"已生成 {output_csv}，共 {len(session_df)} 行")
                
                # 开启新轮次
                session_index += 1
                session_rows = [row]
            else:
                session_rows.append(row)
        previous_time = current_time
        '''
    # 输出最后一轮
    if session_rows:
        session_df = pd.DataFrame(session_rows)
        session_df['行序号'] = range(1, len(session_df)+1)
        output_csv = os.path.join(output_folder, f"{student_id}_{session_index}.csv")
        session_df.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        print(f"已生成 {output_csv}，共 {len(session_df)} 行")

print("所有学生轮次分类完成！")