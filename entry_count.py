import pandas as pd
import os

# 输入 CSV 文件路径
input_csv = "Data/filtered_dialog_masked.csv"
# 输出文件夹
output_path = "Data/full_student2"
os.makedirs(output_path, exist_ok=True)

# 读取 CSV
df = pd.read_csv(input_csv)
'''
# 过滤掉提问入口为 "班级" 的问答
df = df[df['提问入口'] != '班级']
'''
# 获取所有学生 ID
qualified_students = df['学生ID'].unique().tolist()



# 每个学生：按提问时间排序，生成单独 CSV
for student_id in qualified_students:
    student_df = df[df['学生ID'] == student_id].copy()
    student_df['提问时间'] = pd.to_datetime(student_df['提问时间'])
    student_df.sort_values('提问时间', inplace=True)
    
    output_file = os.path.join(output_path, f"{student_id}.csv")
    student_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"{student_id} 的问答已保存到 {output_file}")

print(f"共 {len(qualified_students)} 个学生生成单独 CSV")
