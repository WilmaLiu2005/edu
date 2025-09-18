import json
import ast
import pandas as pd
import re

def load_jsonl(jsonl_file):
    """读取 jsonl 文件，返回 {行序号: {"emails": [], "phones": [], "names": []}}"""
    mapping = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    data = ast.literal_eval(line)
                row_id = data.get("行序号")
                if not row_id:
                    continue
                
                emails = data.get("contacts", {}).get("EMAIL", [])
                phones = data.get("contacts", {}).get("PHONE", [])
                names  = data.get("names", [])
                
                mapping[row_id] = {
                    "emails": emails,
                    "phones": phones,
                    "names": names
                }
            except Exception as e:
                print("跳过无法解析的行：", line.strip(), "错误：", e)
                continue
    return mapping

def replace_placeholders(text, replacements):
    """替换 EMAIL / PHONE / NAME 占位符"""
    for email in replacements["emails"]:
        text = text.replace(email, "[EMAIL]")
    for phone in replacements["phones"]:
        text = text.replace(phone, "[PHONE]")
    for name in replacements["names"]:
        # 用正则确保全字匹配，避免误伤
        text = re.sub(rf"\b{name}\b", "[NAME]", text)
    return text

def process_files(jsonl_file, csv_file, output_csv):
    # 1. 读取 jsonl -> 行序号映射
    mapping = load_jsonl(jsonl_file)

    # 2. 读取 csv
    df = pd.read_csv(csv_file)

    # 3. 遍历替换
    for idx, row in df.iterrows():
        row_id = row["行序号"]
        if row_id in mapping:
            replacements = mapping[row_id]
            for col in ["提问内容", "AI回复"]:  # 只替换这两列
                if pd.notna(row[col]):
                    df.at[idx, col] = replace_placeholders(str(row[col]), replacements)

    # 4. 输出
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"已写入 {output_csv}")

# 用法示例
process_files("/Users/vince/undergraduate/KEG/edu/Data/sensitive_rows_jieba_clean7.jsonl", "/Users/vince/undergraduate/KEG/edu/Data/filtered_dialog.csv", "/Users/vince/undergraduate/KEG/edu/Data/filtered_dialog_masked.csv")
