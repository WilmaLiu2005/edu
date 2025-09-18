'''
import json
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
from collections import Counter, defaultdict

file_path = "/Users/vince/undergraduate/KEG/edu/Data/tag_2nd.jsonl"   # 修改成你的文件名
level_counters = defaultdict(Counter)

# 读取数据，统计前三个有效标签
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        tags = record.get("tags", [])
        tags = [t for t in tags[:3] if t]  # 只取前三个
        for level, tag in enumerate(tags, start=1):
            level_counters[level][tag] += 1

# 获取每一级的标签全集（保持顺序统一）
all_tags = set()
for counter in level_counters.values():
    all_tags.update(counter.keys())
all_tags = sorted(all_tags)  # 排序保证一致性

# 绘制前三个层级的饼状图（带图例）
for level in range(1, 4):
    counter = level_counters.get(level, {})
    if not counter:
        continue
    
    values = [counter.get(tag, 0) for tag in all_tags]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(values, autopct="%1.1f%%", startangle=140)
    
    # 添加图例在右侧
    ax.legend(wedges, all_tags,
              title="标签",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))  # 图例放右边
    
    plt.title(f"第 {level} 级 Tags 占比统计（位置统一）")
    plt.show()

'''
import json
from collections import defaultdict, Counter
import plotly.graph_objects as go

file_path = "/Users/vince/undergraduate/KEG/edu/Data/tag_2nd.jsonl"   # 修改成你的文件名

# 保存层级统计（转成 dict 方便序列化）
level_counters = {1: {"root": Counter()},
                  2: defaultdict(Counter),
                  3: defaultdict(Counter)}

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        tags = record.get("tags", [])
        tags = [t for t in tags[:3] if t]  # 只取前三个
        if len(tags) >= 1:
            level_counters[1]["root"][tags[0]] += 1
        if len(tags) >= 2:
            level_counters[2][tags[0]][tags[1]] += 1
        if len(tags) >= 3:
            # tuple 改成用字符串拼接，避免 json.dumps 报错
            key = tags[0] + "|||" + tags[1]
            level_counters[3][key][tags[2]] += 1

# 把 Counter 转成普通 dict，保证能 json.dumps
def to_dict(d):
    if isinstance(d, Counter):
        return dict(d)
    if isinstance(d, defaultdict) or isinstance(d, dict):
        return {k: to_dict(v) for k, v in d.items()}
    return d

level_counters = to_dict(level_counters)

# 初始图（一级标签）
labels = list(level_counters[1]["root"].keys())
values = list(level_counters[1]["root"].values())

fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.3))
fig.update_layout(title="一级标签分布")

# 导出 HTML
html_file = "tags_drilldown.html"
fig.write_html(html_file, include_plotlyjs="cdn", full_html=True)

# 注入 JS 脚本
extra_js = f"""
<script>
var levelData = {json.dumps(level_counters, ensure_ascii=False)};
var historyStack = [];

var myPlot = document.getElementsByClassName('plotly-graph-div')[0];

myPlot.on('plotly_click', function(data) {{
    var level = historyStack.length + 1;
    var point = data.points[0].label;

    var counts;
    if (level == 1) {{
        historyStack.push(point);
        counts = levelData[2][point];
    }} else if (level == 2) {{
        var parent = historyStack[0];
        historyStack.push(point);
        var key = parent + "|||" + point;
        counts = levelData[3][key];
    }} else {{
        return;  // 第三级不再下钻
    }}

    if (!counts) return;

    var labels = Object.keys(counts);
    var values = Object.values(counts);

    Plotly.react(myPlot, [{{
        type: 'pie',
        labels: labels,
        values: values,
        hole: 0.3
    }}], {{
        title: (level+1) + '级标签分布 - ' + historyStack.join(" > ")
    }});
}});

// 右键返回上一级
myPlot.oncontextmenu = function(e) {{
    e.preventDefault();
    if (historyStack.length === 0) return;

    historyStack.pop();
    var level = historyStack.length + 1;

    var counts;
    if (level == 1) {{
        counts = levelData[1]["root"];
    }} else if (level == 2) {{
        var parent = historyStack[0];
        counts = levelData[2][parent];
    }} else if (level == 3) {{
        var parent = historyStack[0];
        var child = historyStack[1];
        var key = parent + "|||" + child;
        counts = levelData[3][key];
    }}

    var labels = Object.keys(counts);
    var values = Object.values(counts);

    Plotly.react(myPlot, [{{
        type: 'pie',
        labels: labels,
        values: values,
        hole: 0.3
    }}], {{
        title: level + '级标签分布 - ' + (historyStack.join(" > ") || "")
    }});
}};
</script>
"""

with open(html_file, "a", encoding="utf-8") as f:
    f.write(extra_js)

print(f"已生成交互式 drill-down 文件: {html_file}")