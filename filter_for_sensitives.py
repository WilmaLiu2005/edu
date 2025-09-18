import json
# 单姓 + 常见复姓（可自行扩展）
single_surnames = set("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁")  
double_surnames = {"欧阳","司马","诸葛","上官","东方","夏侯","皇甫","尉迟","公孙","慕容","长孙","宇文","司徒","轩辕","令狐","钟离","闾丘","子车","端木","百里","呼延","南宫","独孤"}  

def is_chinese_name(s: str) -> bool:
    """判断一个字符串是否可能是中文名字"""
    s = s.strip()
    if len(s) < 2 or len(s) > 3:  # 常见人名 2~3 字
        return False
    # 复姓
    if len(s) >= 2 and s[:2] in double_surnames:
        return True
    # 单姓
    if s[0] in single_surnames:
        return True
    return False

def clean_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue
            try:
                # 用 ast.literal_eval 解析单引号风格的 dict
                data = json.loads(line.strip())
                
                # 清洗 names 字段：去掉单字
                if "names" in data and isinstance(data["names"], list):
                    ban_words = {"马上会", "明星","罗宾逊","金点子","安抚","范文", "云图", "陈述", "安康", "马屁精", "唐诗", "孟母", "范畴", "严重性", "祝福", "张力", "马拉松", "陈列", "安静", "严谨性", "康复", "安全性", "乐高", "史诗", "成正比", "方大同"}
                    ban_characters = {"氏", "老师", "同学", "某"}
                    data["names"] = [n for n in data["names"] if len(n) > 1 and len(n) < 4 and n not in ban_words and not any(char in n for char in ban_characters) and is_chinese_name(n)]
                
                if data["names"] or data["contacts"]:
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            except Exception as e:
                print("跳过无法解析的行：", line.strip(), "错误：", e)
                continue

# 使用方法
clean_jsonl("/Users/vince/undergraduate/KEG/edu/Data/sensitive_rows_jieba_clean6.jsonl", "/Users/vince/undergraduate/KEG/edu/Data/sensitive_rows_jieba_clean7.jsonl")

# ban_words = {"索引", "修正", "智慧", "耶稣", "孔庙", "纪律性", "马修" ,"谢谢", "明白", "英勇", "雷达", "傅里叶", "恩格斯", "马克思", "列宁", "斯大林", "毛泽东", "邓小平", "江泽民", "胡锦涛", "习近平", "贝叶斯", "高斯", "拉普拉斯", "泊松", "卡方", "赫兹", "焦耳", "牛顿", "瓦特", "安培", "伏特", "欧姆", "法拉第", "库仑", "特斯拉", "爱因斯坦", "玻尔", "薛定谔", "海森堡", "狄拉克", "费曼", "普朗克", "居里夫人", "达尔文", "孟德尔", "华生", "克里克", "沃森", "道德", "荣耀", "克隆", "伦理", "笛卡尔", "康德", "黑格尔", "尼采", "柏拉图", "亚里士多德", "苏格拉底", "孔子", "孟子", "庄子", "老子", "墨子", "荀子", "韩非子", "孙子", "孙武", "曹操", "诸葛亮", "刘备", "关羽", "张飞", "司马懿", "周瑜", "吕布", "貂蝉", "大乔", "小乔", "文明", "林桂", "拓宽", "权威", "真善美", "亚文化"}