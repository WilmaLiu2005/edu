import pandas as pd
import csv

# -----------------------------
# 文件路径
# -----------------------------
dialog_csv = "fixed_with_index.csv"    # 对话 CSV
class_csv = "activated_list.csv"             # 班级信息 CSV
output_csv = "Data/filtered_dialog.csv"    # 输出 CSV

# -----------------------------
# 读取对话 CSV
# -----------------------------
df_dialog = pd.read_csv(
    dialog_csv,
    sep=",",
    quotechar='"',
    doublequote=True,
    encoding="utf-8-sig",
    keep_default_na=False,
    engine="python"   # Python engine 更稳定处理多行字段
)

df_dialog.columns = df_dialog.columns.str.strip()
df_dialog['教学班ID'] = df_dialog['教学班ID'].astype(str).str.strip()

# -----------------------------
# 读取班级 CSV
# -----------------------------
df_class = pd.read_csv(class_csv, sep=",", encoding="utf-8-sig")
df_class.columns = df_class.columns.str.strip()
df_class['class_id'] = df_class['class_id'].astype(str).str.strip()

# -----------------------------
# 过滤对话 CSV：只保留教学班ID在AIclass_id中的行
# -----------------------------
filtered_df = df_dialog[df_dialog['教学班ID'].isin(df_class['class_id'])].copy()

# -----------------------------
# 输出 CSV
# -----------------------------
filtered_df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_ALL
)

print(f"已生成过滤后的 CSV 文件: {output_csv}, 共 {len(filtered_df)} 行")
