# 先时间间隔15分钟，然后再LLM划分
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
OUTPUT_FOLDER = "/Users/vince/undergraduate/KEG/edu/test_split_reverse" # 输出 CSV 文件夹

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def robust_read_csv(file_path, text_columns=None):
    """
    安全读取 CSV 文件，适用于含有换行符、Markdown 图片链接、双引号等情况。
    参数：
        file_path: CSV 文件路径
        text_columns: 需要处理换行符的文本列列表，例如 ["提问内容", "AI回复"]
    返回：
        pd.DataFrame 或 None（空文件或格式异常）
    """
    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            engine="python",       # 更灵活，支持多种复杂 CSV
            quotechar='"',         # 引号包裹的字段
            doublequote=True,      # 双引号转义
            keep_default_na=False, # 保留空字符串而不是 NaN
        )
    except Exception as e:
        print(f"⚠️ 读取 CSV 失败: {file_path}, 错误: {e}")
        return None

    if df.empty:
        print(f"⚠️ 文件 {file_path} 是空的")
        return None

    # 去除列名前后空格
    df.columns = df.columns.str.strip()

    # 检查必要列
    required_cols = ["行序号", "提问时间"]
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️ 文件 {file_path} 缺少必要列: {col}")
            return None

    # 处理文本列换行符
    if text_columns:
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("\n", " ", regex=False)

    # 转换行序号为整数
    df["行序号"] = pd.to_numeric(df["行序号"], errors="coerce")
    df = df.dropna(subset=["行序号"])
    df["行序号"] = df["行序号"].astype(int)

    # 转换时间列
    df["提问时间"] = pd.to_datetime(df["提问时间"], errors="coerce")
    df = df.dropna(subset=["提问时间"])

    if df.empty:
        print(f"⚠️ 文件 {file_path} 全部行无效（行序号或提问时间解析失败）")
        return None

    df = df.sort_values("提问时间").reset_index(drop=True)
    return df

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
# JSON 解析（鲁棒）
# ==============================
def robust_json_parse(text):
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
# Stage 1: 时间划分
# ==============================
def time_based_split(df, time_threshold=15):
    results = []
    if df.empty:
        return results

    current_chunk = [df.iloc[0]]
    for i in range(1, len(df)):
        delta = (df.iloc[i]["提问时间"] - df.iloc[i - 1]["提问时间"]).total_seconds() / 60
        if delta > time_threshold:
            results.append(pd.DataFrame(current_chunk))
            current_chunk = [df.iloc[i]]
        else:
            current_chunk.append(df.iloc[i])
    if current_chunk:
        results.append(pd.DataFrame(current_chunk))
    return results

# ==============================
# Stage 2: LLM 划分
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
1. 必须保持原始顺序，不能打乱或跳跃
2. 如果两条交互之间的时间间隔过长，就一定要分成新的对话
3. 如果话题发生明显变化，也要开启新对话
4. 否则视为同一段对话  

输出格式：
- JSON 格式：{{行号: 对话编号}}，对话编号从 1 开始  
- 不要输出其他说明或文字  

交互记录如下：
{dialogues_json}
"""

def llm_split(group_df):
    dialogues = []
    for idx, row in group_df.iterrows():
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
        return {}
    return robust_json_parse(response)

# ==============================
# 处理单个 CSV
# ==============================
def process_csv(file_path, output_folder):
    try:
        # 读取 CSV
        df = robust_read_csv(file_path, text_columns=["提问内容", "AI回复"])
        if df is None or df.empty:
            print(f"⚠️ {file_path} 无有效数据，跳过")
            return

        student_id = df["学生ID"].iloc[0] if "学生ID" in df.columns else os.path.splitext(os.path.basename(file_path))[0]

        # Stage 1: 时间间隔划分
        time_splits = time_based_split(df)  # 每块顺序保持不变

        file_index = 1  # 全局编号

        # Stage 2: LLM 划分
        for time_idx, group in enumerate(time_splits, start=1):
            mapping = llm_split(group)
            # print(mapping)
            if not mapping:
                print(f"⚠️ {student_id} 时间片 {time_idx} LLM 划分失败，跳过")
                continue

            # 将 LLM 输出的 session_id 顺序映射到预对话顺序
            session_ids = []
            prev_session = 1
            for idx in range(len(group)):
                key = str(idx + 1)  # LLM 的行号通常从 1 开始
                sid = mapping.get(key, prev_session)
                session_ids.append(sid)
                prev_session = sid

            # 根据 session_id 切分子会话
            current_session = []
            current_id = session_ids[0]
            for idx, sid in enumerate(session_ids):
                if sid != current_id:
                    # 保存上一段子会话
                    sub_group = group.iloc[current_session].copy().reset_index(drop=True)
                    output_csv = os.path.join(output_folder, f"{student_id}_{file_index}.csv")
                    sub_group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                    print(f"✅ 已生成 {output_csv}, 共 {len(sub_group)} 行")
                    file_index += 1

                    # 开启新子会话
                    current_session = [idx]
                    current_id = sid
                else:
                    current_session.append(idx)

            # 保存最后一段子会话
            if current_session:
                sub_group = group.iloc[current_session].copy().reset_index(drop=True)
                output_csv = os.path.join(output_folder, f"{student_id}_{file_index}.csv")
                sub_group.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
                print(f"✅ 已生成 {output_csv}, 共 {len(sub_group)} 行")
                file_index += 1

    except Exception as e:
        print(f"❌ 处理文件 {file_path} 出错: {e}")


# ==============================
# 主程序
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
                print(f"❌ 处理 {file} 出错: {e}")

    print("🎉 所有文件处理完成！")

if __name__ == "__main__":
    main()