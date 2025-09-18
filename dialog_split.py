from openai import OpenAI
import random
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import json5
import pandas as pd
import csv
# ==============================    

MAX_WORKERS = 50
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 3
API_KEY = "sk-0jErqj61bIYM135CEqhfj318rKIM1TIa"  # 填写你的 API key
BASE_URL = "https://api-gateway.glm.ai/v1"
MODEL_NAME = "gemini-2.5-flash"  # 或你自己的模型
INPUT_FOLDER = "/Users/vince/undergraduate/KEG/edu/Data/pre_validation5"   # 输入 CSV 文件夹
OUTPUT_FOLDER = "/Users/vince/undergraduate/KEG/edu/test_split2" # 输出 CSV 文件夹

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==============================
# GPT API 调用
# ==============================
def gpt_api_call(messages, model=MODEL_NAME):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=10000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_RETRY_DELAY)
                time.sleep(delay)
            else:
                return None

# ==============================
# Prompt 模板
# ==============================
PROMPT_TEMPLATE = """
你是一个对话分析助手。
以下是一个学生和 AI 学伴的完整交互记录，按时间顺序排列。
每条记录包含:
- 行序号
- 提问时间
- 提问内容
- AI回复

请按照对话轮次（session）将这些交互分组：
- 每个轮次必须保持原始顺序，不能跳跃
- **连续相关的问题和回答属于同一轮次**
- **如果话题变化明显，则开启新轮次**
- 输出 JSON 格式：{{行号: 轮次编号}}，轮次编号从 1 开始
- 只输出 JSON，不要输出其他说明

对话记录如下：
{dialogues_json}
"""

# ==============================
# JSON 解析（鲁棒）
# ==============================
def robust_json_parse(text):
    # 去掉 ```json ... ``` 包裹
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            return json5.loads(cleaned)
        except Exception:
            print("⚠️ JSON 解析失败，原始输出：", text[:200])
            return {}

# ==============================
# 单个 CSV 处理函数
# ==============================
def process_csv(file_path, output_folder):
    df = pd.read_csv(file_path, encoding="utf-8-sig", engine="python", quotechar='"', doublequote=True)
    df.columns = df.columns.str.strip()
    df = df.sort_values(by="提问时间").reset_index(drop=True)

    # 提取必要字段
    dialogues = []
    for idx, row in df.iterrows():
        dialogues.append({
            "行号": int(idx) + 1,
            "提问时间": str(row["提问时间"]),
            "提问内容": str(row["提问内容"]),
            "AI回复": str(row["AI回复"])
        })

    dialogues_json = json.dumps(dialogues, ensure_ascii=False)
    prompt = PROMPT_TEMPLATE.format(dialogues_json=dialogues_json)

    messages = [
        {"role": "system", "content": "你是对话轮次分割助手"},
        {"role": "user", "content": prompt}
    ]

    response = gpt_api_call(messages)
    if not response:
        print(f"⚠️ {os.path.basename(file_path)} 分割失败")
        return

    mapping = robust_json_parse(response)

    # 不在 df 里加轮次列，而是直接用 mapping 拆分
    student_id = df["学生ID"].iloc[0] if "学生ID" in df.columns else os.path.splitext(os.path.basename(file_path))[0]

    # mapping: {行号: 轮次}
    for session_id in sorted(set(mapping.values()), key=lambda x: int(x)):
        # 找到属于这个轮次的行
        indices = [int(k) - 1 for k, v in mapping.items() if str(v) == str(session_id)]
        group = df.iloc[indices].copy()
        group.reset_index(drop=True, inplace=True)
    
        output_csv = os.path.join(output_folder, f"{student_id}_{session_id}.csv")
        group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        print(f"✅ 已生成 {output_csv}，共 {len(group)} 行")

# ==============================
# 主程序（并发）
# ==============================
def main():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_csv, os.path.join(INPUT_FOLDER, f), OUTPUT_FOLDER): f for f in files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ 处理 {file} 出错: {str(e)}")

    print("🎉 所有文件处理完成！")

if __name__ == "__main__":
    main()