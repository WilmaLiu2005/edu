import os
import pandas as pd

# 用户输入文件夹路径
input_dir = input("请输入包含学生CSV文件的文件夹路径: ").strip()

# 输出目录 test
output_dir = os.path.join(os.getcwd(), "real_GT_split")
os.makedirs(output_dir, exist_ok=True)


# 遍历 CSV 文件
for file_name in os.listdir(input_dir):
    if not file_name.endswith(".csv"):
        continue

    file_path = os.path.join(input_dir, file_name)
    df = pd.read_csv(file_path)

    # 检查必要列
    if not {"学生ID", "提问入口", "提问时间"}.issubset(df.columns):
        print(f"{file_name} 缺少必要列，跳过")
        continue

    # 转换时间格式
    df["提问时间"] = pd.to_datetime(df["提问时间"])
    df = df.sort_values("提问时间").reset_index(drop=True)

    # 划分会话
    sessions = []
    current_session = []
    last_time = None
    last_entry = None

    for idx, row in df.iterrows():
        entry = row["提问入口"]
        time = row["提问时间"]

        if (
            last_time is None
            or entry != last_entry                      # 提问入口变化
            or (time - last_time).total_seconds() > 15 * 60  # 超过15分钟
        ):
            # 开启新会话
            if current_session:
                sessions.append(pd.DataFrame(current_session))
                current_session = []

        current_session.append(row)
        last_time = time
        last_entry = entry

    # 保存最后一段会话
    if current_session:
        sessions.append(pd.DataFrame(current_session))

    # 导出每个会话为一个新的 CSV
    base_name, ext = os.path.splitext(file_name)
    for i, session_df in enumerate(sessions, start=1):
        save_path = os.path.join(output_dir, f"{base_name}_{i}{ext}")
        session_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"{file_name} -> {save_path} 已保存")


# 找到 CSV 文件
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
file_count = len(csv_files)
print(f"找到 {file_count} 个 CSV 文件，开始处理...")

total_sessions = 0  # 用于统计总轮次

for file in csv_files:
    path = os.path.join(input_dir, file)
    df = pd.read_csv(path)
    
    if "提问轮次(一个会话一天内)" in df.columns:
        num_sessions = df["提问轮次(一个会话一天内)"].nunique()
        total_sessions += num_sessions
        print(f"{file} 中共有 {num_sessions} 轮对话")
    else:
        print(f"{file} 缺少 '提问轮次(一个会话一天内)' 列，跳过统计")

print(f"处理完所有文件，总轮次对话数: {total_sessions}")