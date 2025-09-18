# 先LLM，再时间间隔15分钟
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
MODEL_NAME = "gpt-5-2025-08-07"  # 或你自己的模型
INPUT_FOLDER = "/Users/vince/undergraduate/KEG/edu/Data/pre_validation5"   # 输入 CSV 文件夹
OUTPUT_FOLDER = "/Users/vince/undergraduate/KEG/edu/test_split" # 输出 CSV 文件夹

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
以下是某个学生和 AI 学伴的完整交互记录，按时间顺序排列。
每条记录包含:
- 行序号
- 提问时间
- 提问内容
- AI回复

请将这些交互划分成多段独立的对话（session）。  
划分规则：
1. **必须保持原始顺序，不能打乱或跳跃**  
2. **如果两条交互之间的时间间隔过长，就一定要分成新的对话**  
3. **如果话题发生明显变化，也要开启新对话**  
4. 否则视为同一段对话  

输出格式：
- JSON 格式：{{行号: 对话编号}}，对话编号从 1 开始  
- 不要输出其他说明或文字  

交互记录如下：
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
# 二阶段划分逻辑
# ==============================
def two_stage_split(df, mapping, time_threshold=15):
    """
    Stage 1: 根据 LLM 输出的 mapping 进行初步分组
    Stage 2: 在每组内按时间间隔 <= time_threshold (分钟) 再次划分
    """
    results = []
    df["提问时间"] = pd.to_datetime(df["提问时间"])

    for session_id in sorted(set(mapping.values()), key=lambda x: int(x)):
        indices = [int(k) - 1 for k, v in mapping.items() if str(v) == str(session_id)]
        group = df.iloc[indices].copy().sort_values("提问时间").reset_index(drop=True)

        # Stage 2: 时间划分
        sub_session = 1
        current_chunk = [group.iloc[0]]

        for i in range(1, len(group)):
            delta = (group.iloc[i]["提问时间"] - group.iloc[i - 1]["提问时间"]).total_seconds() / 60
            if delta > time_threshold:
                results.append((session_id, sub_session, pd.DataFrame(current_chunk)))
                sub_session += 1
                current_chunk = [group.iloc[i]]
            else:
                current_chunk.append(group.iloc[i])

        if current_chunk:
            results.append((session_id, sub_session, pd.DataFrame(current_chunk)))

    return results

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
        {"role": "system", "content": "你是交互轮次分割助手"},
        {"role": "user", "content": prompt}
    ]

    response = gpt_api_call(messages)
    if not response:
        print(f"⚠️ {os.path.basename(file_path)} 分割失败")
        return

    mapping = robust_json_parse(response)

    student_id = df["学生ID"].iloc[0] if "学生ID" in df.columns else os.path.splitext(os.path.basename(file_path))[0]

    # Two-stage 分割
    results = two_stage_split(df, mapping)

    seq = 1
    for session_id, sub_session, group in results:
        # 使用 4 位零填充：0001, 0002, ...
        filename = f"{student_id}_{seq}.csv"
        output_csv = os.path.join(output_folder, filename)
        group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        print(f"✅ 已生成 {output_csv}（原：{session_id}_{sub_session} -> 序号 {seq}），共 {len(group)} 行")
        seq += 1

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