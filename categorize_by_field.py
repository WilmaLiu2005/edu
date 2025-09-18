import pandas as pd
import os
import csv
import re

# -----------------------------
# 课程领域分类字典
# -----------------------------
field_dict = {
  "电气与信息工程": [
    "电路原理",
    "电路原理(英)",
    "电路分析基础A",
    "电路分析基础A（混合式）",
    "电路分析基础B",
    "电路分析基础",
    "电路原理I",
    "数字电子技术",
    "模拟电子线路A",
    "电机学（上）-H",
    "传感器与检测技术",
    "信号与系统A",
    "数字信号处理B",
    "信号检测与估计"
  ],
  "计算机科学与技术": [
    "计算机程序设计基础",
    "Fortran程序设计",
    "C程序设计",
    "Java程序设计",
    "数据结构",
    "操作系统",
    "数据库系统原理与应用",
    "数据库系统和信息管理",
    "数据库技术与应用",
    "计算机系统基础Ⅰ（混合式）",
    "网络技术与应用（混合式）",
    "机器学习A",
    "计算机在化学中的应用"
  ],
  "化学与化工": [
    "有机化学B",
    "无机及分析化学B",
    "化工原理",
    "化工原理B",
    "化工设计",
    "化工仪表及自动化",
    "有机化学",
    "分析化学",
    "中药化学",
    "医用化学",
    "基础化学2",
    "物理化学 I2",
    "物理化学 II2"
  ],
  "物理学": [
    "核辐射物理及探测学",
    "物理",
    "大学物理AI",
    "波谱学",
    "天气学原理"
  ],
  "数学": [
    "离散数学",
    "高等数学Ⅰ2",
    "高等数学AⅡ"
  ],
  "医学与生命科学": [
    "野生动物及异宠医学",
    "疾病与健康",
    "家庭急救",
    "医学图像处理",
    "外科护理学A1",
    "病理学",
    "组织胚胎学",
    "医学免疫学（Ⅰ）（双语）",
    "解剖学（Ⅰ）",
    "生理学（双语）",
    "医学影像设备学",
    "生物化学与分子生物学",
    "生理学",
    "医学免疫学",
    "病毒学检验",
    "细胞生物学"
  ],
  "语言与人文": [
    "博士生英语",
    "大学英语A2",
    "英语视听说2",
    "外国文化概论",
    "普通语言学",
    "英语Ⅱ",
    "中国语言之美：成语篇（25春）",
    "教育文化学",
    "摄影技术与艺术",
    "科学进步与技术革命",
    "中国近现代史纲要",
    "毛泽东思想和中国特色社会主义理论体系概论"
  ],
  "机械工程与能源": [
    "机械工程基础",
    "工程热力学",
    "水力机组辅助设备",
    "飞机几何造型技术"
  ],
  "经济与管理": [
    "民法总论",
    "知识产权法与创新保护",
    "工程合同管理",
    "战略管理（全英文）(混合式)",
    "管理学",
    "审计实务",
    "税务筹划",
    "高级财务会计",
    "审计学原理",
    "金融风险管理"
  ],
  "环境与土木工程": [
    "实验室安全与环境保护",
    "隧道与地铁工程",
    "防排烟工程",
    "消防给水工程"
  ]
}

# -----------------------------
# 文件路径
# -----------------------------
input_csv = "/Users/vince/undergraduate/KEG/edu/Data/filtered_dialog_masked.csv"   # 已处理好的 CSV
output_folder = "/Users/vince/undergraduate/KEG/edu/Data/split_field_output"       # 输出文件夹

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
# 创建课程到领域的映射
# -----------------------------
course_to_field = {}
for field, courses in field_dict.items():
    for course in courses:
        course_to_field[course] = field

# -----------------------------
# 按领域分组并生成CSV
# -----------------------------
# 使用map函数为每行数据分配领域（不添加到DataFrame）
field_groups = df.groupby(df['课程名称'].map(course_to_field).fillna('其他')) # 先groupby再map

for field_name, field_group in field_groups:
    # 清理领域名称用于文件名
    safe_field = re.sub(r'[\\/*?:"<>|]', '_', field_name)
    
    # 按提问时间排序
    field_group_sorted = field_group.sort_values(by='提问时间').copy()
    field_group_sorted.reset_index(drop=True, inplace=True)
    
    # 生成CSV文件
    output_csv = os.path.join(output_folder, f"{safe_field}.csv")
    field_group_sorted.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    print(f"已生成 {output_csv}，共 {len(field_group_sorted)} 行")

print("所有领域分类完成！")