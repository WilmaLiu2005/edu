import pandas as pd
import os
import csv
from datetime import timedelta
import re

# -----------------------------
# 文件路径
# -----------------------------
input_dir = "/Users/vince/undergraduate/KEG/edu/Data/split_field_output"   # 输入目录
output_base_dir = "/Users/vince/undergraduate/KEG/edu/Data/field_dialog_split"  # 输出基础目录

# -----------------------------
# 创建输出基础目录
# -----------------------------
os.makedirs(output_base_dir, exist_ok=True)

# -----------------------------
# 遍历输入目录中的所有CSV文件
# -----------------------------
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        # 构建输入文件路径
        input_csv = os.path.join(input_dir, filename)
        
        # 创建与CSV文件同名的输出文件夹
        # 移除.csv扩展名
        folder_name = os.path.splitext(filename)[0]
        output_folder = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"正在处理文件: {filename}")
        print(f"输出目录: {output_folder}")
        
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
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
        else:
            print(f"警告: 文件 {filename} 中没有找到 '{time_col}' 列")
            continue
        
        # -----------------------------
        # 分割轮次逻辑
        # -----------------------------
        time_threshold = timedelta(minutes=15)  # 超过30分钟就分新轮次
        for student_id, student_group in df.groupby('学生ID'):
            student_group_sorted = student_group.sort_values(by=time_col).copy()
            student_group_sorted.reset_index(drop=True, inplace=True)
            
            # 初始化
            session_index = 1
            session_rows = []
            previous_time = None
            
            for idx, row in student_group_sorted.iterrows():
                current_time = row[time_col]
                # session_rows.append(row)

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
            # 输出最后一轮
            if session_rows:
                session_df = pd.DataFrame(session_rows)
                session_df['行序号'] = range(1, len(session_df)+1)
                output_csv = os.path.join(output_folder, f"{student_id}_{session_index}.csv")
                session_df.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"已生成 {output_csv}，共 {len(session_df)} 行")
        
        print(f"文件 {filename} 处理完成！")

print("所有文件处理完成！")