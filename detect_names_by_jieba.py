import re
import jieba.posseg as pseg
import pandas as pd

def detect_contacts(text: str):
    if not isinstance(text, str):
        return {}

    findings = {}

    # 邮箱
    emails = re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', text)
    if emails:
        findings["EMAIL"] = emails

    # 中国大陆手机号
    phones = re.findall(r'\b1[3-9]\d{9}\b', text)
    if phones:
        findings["PHONE"] = phones

    return findings


def detect_names(text: str, context_window: int = 5):
    """
    使用 jieba 词性标注检测疑似人名（nr），
    只有当人名附近出现“老师”或“同学”时才判定为敏感信息。
    
    参数:
        text (str): 输入文本
        context_window (int): 上下文窗口大小，默认 2 个词
    返回:
        List[str]: 检测到的敏感人名
    """
    if not isinstance(text, str):
        return []
    
    findings = []
    words = list(pseg.cut(text))  # 转换为列表方便索引
    
    for i, (word, flag) in enumerate(words):
        if flag == "nr":  # nr 表示人名
            # 取前后 context_window 个词
            left_context = [w for w, _ in words[max(0, i-context_window):i]]
            right_context = [w for w, _ in words[i+1:i+1+context_window]]
            context = left_context + right_context

            if any(kw in context for kw in ["老师", "同学", "教授", "讲师", "助教", "学生", "你", "您", "我", "咱", "他", "她"]):
                if word not in findings:
                    findings.append(word)
    
    return findings

def detect_sensitive_in_csv(file_path: str):
    df = pd.read_csv(file_path)

    results = []
    for i, row in df.iterrows():
        text = str(row.get("提问内容", "")) + " " + str(row.get("AI回复", ""))
        contacts = detect_contacts(text)
        names = detect_names(text)
        if contacts or names:
            results.append({
                "行序号": row["行序号"],
                "contacts": contacts,
                "names": names
            })
    return results

if __name__ == "__main__":
    input_csv = "/Users/vince/undergraduate/KEG/edu/Data/filtered_dialog.csv"  # 替换你的 CSV 路径
    output_json = "/Users/vince/undergraduate/KEG/edu/Data/sensitive_rows_jieba.jsonl"
    sensitive_rows = detect_sensitive_in_csv(input_csv)
    with open(output_json, "w", encoding="utf-8") as f:
        for item in sensitive_rows:
            f.write(f"{item}\n")
    print(f"检测完成，共 {len(sensitive_rows)} 行含敏感信息，已保存到 {output_json}")