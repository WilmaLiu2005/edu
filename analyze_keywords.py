import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import re

# 停用词列表
stopwords = set([
    "the", "and", "of", "in", "to", "https", "com",
    "图片", "综上所述", "答案", "总结", "所以", "因为",
    "用户", "例如", "这", "那", "来说", "总之", "一个", "一些", "可以", "就是"
])

# -----------------------------
# 命令行参数
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="聚类结果 CSV (包含 dialog_text 和 cluster 列)")
parser.add_argument("--topk", type=int, default=100, help="每簇提取的关键词数（过滤掉纯数字）")
args = parser.parse_args()

df = pd.read_csv(args.input)

# -----------------------------
# TF-IDF 提取关键词（全局词表）
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["dialog_text"])
words = vectorizer.get_feature_names_out()

# 保存每个簇的关键词集合和排名
cluster_keywords = {}
cluster_rankings = {}

for c in sorted(df["cluster"].unique()):
    cluster_texts = df[df["cluster"] == c]["dialog_text"]
    if cluster_texts.empty:
        continue

    sub_vec = vectorizer.transform(cluster_texts)
    tfidf_scores = sub_vec.mean(axis=0).A1
    top_idx = tfidf_scores.argsort()[::-1]

    top_words = []
    rankings = {}
    rank_counter = 1
    for i in top_idx:
        w = words[i]
        if re.fullmatch(r"\d+", w):  # 去掉纯数字
            continue
        if w.lower() in stopwords or any(stopword in w for stopword in stopwords):
            continue
        top_words.append(w)
        rankings[w] = rank_counter
        rank_counter += 1
        if len(top_words) >= args.topk:
            break

    cluster_keywords[c] = top_words
    cluster_rankings[c] = rankings

# -----------------------------
# 计算公共高频词（出现于所有簇的词）
# -----------------------------
all_keywords_sets = [set(kw) for kw in cluster_keywords.values()]
common_keywords = set.intersection(*all_keywords_sets)

print("===== 公共高频词 =====")
print(common_keywords)
print("===== 公共高频词在各簇中的排名对比 =====")
df_compare = pd.DataFrame(
    {c: {w: cluster_rankings[c].get(w, None) for w in common_keywords}
     for c in cluster_rankings}
).T  # 行=簇，列=词

# 排序：公共词按在所有簇的平均排名升序
df_compare = df_compare[sorted(df_compare.columns, key=lambda w: df_compare[w].mean(skipna=True))]

print(df_compare.fillna("-").to_string())

print("\n===== 去掉公共高频词后的前100个关键词 =====")
for c, kw in cluster_keywords.items():
    filtered_kw = [w for w in kw if w not in common_keywords][:args.topk]
    print(f"簇 {c}: {', '.join(filtered_kw)}")