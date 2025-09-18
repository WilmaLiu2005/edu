import pandas as pd
import csv

# 文件路径
input_csv = "active_class_dataset.csv"
output_csv = "activated_list.csv"

# 读取 CSV
df = pd.read_csv(
    input_csv,
    sep=",",
    quotechar='"',
    doublequote=True,
    encoding="utf-8",
    keep_default_na=False,
    engine="python"  # Python engine 对多行字段更稳
)

# 添加行序号列
df.insert(0, '行序号', range(1, len(df) + 1))

# 输出 CSV，保证 Excel 正确显示
df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_ALL
)

print(f"已生成修复并加行序号的 CSV 文件: {output_csv}, 共 {len(df)} 行")