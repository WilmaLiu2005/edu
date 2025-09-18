import csv
import json

# 输入 CSV 文件路径
input_csv = '/Users/vince/undergraduate/KEG/edu/Data/filtered_dialog.csv'
# 输出 JSONL 文件路径
output_jsonl = '/Users/vince/undergraduate/KEG/edu/Data/log.jsonl'

with open(input_csv, newline='', encoding='utf-8') as csvfile, open(output_jsonl, 'w', encoding='utf-8') as jsonlfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        prompt_text = f"提问内容：{row['提问内容']}\nAI回复：{row['AI回复']}"
        json_line = json.dumps({"prompt": prompt_text}, ensure_ascii=False)
        jsonlfile.write(json_line + '\n')

print(f"已生成 JSONL 文件：{output_jsonl}")
