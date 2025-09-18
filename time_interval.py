import os
import pandas as pd
import matplotlib.pyplot as plt
# 设置全局字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ------------------ 配置 ------------------
folder_path = "/Users/vince/undergraduate/KEG/edu/Data/轮次分类CSV"  # 替换成你的文件夹路径
time_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 360, float('inf')]  # 单位：分钟
time_labels = ["0-5 min","5-10 min","10-15 min","15-20 min","20-25 min","25-30 min","30-35 min","35-40 min","40-45 min","45-50 min","50-55 min", "55-60 min", "1-6 h", "6+ h"]

# ------------------ 遍历 CSV ------------------
all_intervals = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # 将提问时间转换为 datetime
        df['提问时间'] = pd.to_datetime(df['提问时间'], format='%Y-%m-%d %H:%M:%S')
        df = df.sort_values(by='提问时间')
        
        # 按学生ID分组计算连续提问间隔
        for student_id, group in df.groupby('学生ID'):
            times = group['提问时间'].sort_values()
            intervals = times.diff().dropna().dt.total_seconds() / 60  # 转为分钟
            all_intervals.extend(intervals.tolist())

# ------------------ 分组统计 ------------------
interval_series = pd.Series(all_intervals)
interval_counts = pd.cut(interval_series, bins=time_bins, labels=time_labels).value_counts().sort_index()

# ------------------ 绘制柱状图 ------------------
plt.figure(figsize=(8,5))
interval_counts.plot(kind='bar', color='skyblue')
plt.title("学生连续提问间隔分布")
plt.ylabel("次数")
plt.xlabel("连续提问间隔")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()