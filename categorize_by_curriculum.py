import os
import csv
from collections import defaultdict

def split_csv_by_course(input_csv_path, output_folder_path):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder_path, exist_ok=True)

    # 用字典存储每个课程的数据行
    course_rows = defaultdict(list)

    # 读取 CSV
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames  # 保存表头
        for row in reader:
            course_name = row['课程名称']
            course_rows[course_name].append(row)
    
    course_list = []
    # 为每个课程写入一个 CSV
    for course_name, rows in course_rows.items():
        # 生成合法的文件名
        course_list.append(course_name)
        safe_course_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in course_name)
        output_csv_path = os.path.join(output_folder_path, f"{safe_course_name}.csv")
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
    with open(os.path.join(output_folder_path, 'course_list.txt'), 'w', encoding='utf-8') as f_list:
        for course in course_list:
            f_list.write(course + '\n')
            
    print(f"已按课程名称拆分完成，输出路径：{output_folder_path}")

# 示例用法
input_csv = "/Users/vince/undergraduate/KEG/edu/Data/filtered_dialog_masked.csv"        # 输入 CSV 路径
output_folder = "/Users/vince/undergraduate/KEG/edu/Data/split_courses_output"   # 输出文件夹
split_csv_by_course(input_csv, output_folder)
