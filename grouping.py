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
            
PROMPT_TEMPLATE = """
你是一个课程分类助手。
以下是所有课程的名称列表，请将它们分类到合适的类别中。
要求输出的 JSON 格式如下：
{{
  "类别1": ["课程A", "课程B"],
  "类别2": ["课程C"]
}}
类别1、类别2 等等请根据课程名称自动生成。
请确保输出的 JSON 格式正确，且包含所有课程名称。
课程名称列表：{course_list}
请开始分类。
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
        
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        course_list = [line.strip() for line in f if line.strip()]
    course_list_str = ", ".join(course_list)
    prompt = PROMPT_TEMPLATE.format(course_list=course_list_str)
    messages = [
        {"role": "user", "content": prompt}
    ]
    response_text = gpt_api_call(messages)
    if response_text is None:
        print(f"⚠️ API 调用失败，跳过文件 {file_path}")
        return
    result = robust_json_parse(response_text)
    if not result:
        print(f"⚠️ 解析结果为空，跳过文件 {file_path}")
        return
    print(result)
    output_path = "/Users/vince/undergraduate/KEG/edu/Data/split_courses_output/course_list_categorized.json"
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(result, f_out, ensure_ascii=False, indent=2)

process_file("/Users/vince/undergraduate/KEG/edu/Data/split_courses_output/course_list.txt")