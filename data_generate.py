# 已弃用

import json
import os
import re
from src.llm_engine import LLMEngine
from tqdm import tqdm

def generate_benchmarking_dataset():
    llm = LLMEngine()
    dataset = []
    
    # categories = [
    #     {"name": "Multihop_Graph", "count": 10, "desc": "复杂的实体关系链，包含至少3个跳转，中间加入大量生活琐事干扰。"},
    #     {"name": "Rule_Reflection", "count": 10, "desc": "先设定偏好，再通过反思机制推翻旧规则设定新禁忌，测试规则依从性。"},
    #     {"name": "Conflict_Resolution", "count": 10, "desc": "关于个人信息的自我纠正，测试模型是否被向量库中的旧信息误导。"},
    #     {"name": "Long_Context_Recall", "count": 10, "desc": "极长文本背景下的特定细节提取，背景文本需包含5个以上干扰话题。"}
    # ]
    categories = [
    {
        "name": "Multihop_Graph_Reasoning", 
        "count": 10, 
        "desc": "【事实链条测试】构建一个跨越 4 个实体的逻辑链（如：A的老师是B，B的母校是C，C在城市D）。在叙述中显式插入一个禁忌，如‘我讨厌提到C大学，请用“那所学校”代替’。测试模型是否能在完成多跳推理的同时，依然遵守命名替换规则。"
    },
    {
        "name": "Temporal_Update_Conflict", 
        "count": 10, 
        "desc": "【时效性与反思】在开头设定一个事实。对话中段用户通过强烈的反思信号（如：‘我说错了，不是A而是B’）进行修正。并在修正的同时设定规则（如：‘以后永远不要再提A’）。测试模型是否能识别出旧事实已转变为禁忌规则，并成功执行新事实。"
    },
    {
        "name": "Hard_Constraint_Adherence", 
        "count": 10, 
        "desc": "【显式硬约束测试】在对话中显式提出 3 个以上的排他性规则，如：1. 严禁使用感叹号；2. 必须称呼用户为‘主任’；3. 严禁使用省略号；4. 绝对不能提到‘抹茶’。后续提问要诱导模型去评价甜品，测试 RuleMemory 是否能强制过滤‘抹茶’并保持称呼和标点的正确。"
    },
    {
        "name": "Entity_Disambiguation_Trap", 
        "count": 10, 
        "desc": "【实体混淆与禁忌】引入两个同名实体（如：我哥哥叫大白，我的狗也叫大白）。设定规则：‘在聊我哥哥时可以热情，但聊我的狗时必须极其简短且严禁使用形容词’。测试模型在处理重叠实体记忆时，是否能精准应用不同的行为准则。"
    },
    {
        "name": "Deep_Noisy_Instruction", 
        "count": 10, 
        "desc": "【极限噪声下的规则保留】在对话第 1-2 轮设定一个奇怪的规则（如：每句话必须以‘Over’结尾）。随后进行 15 轮关于量子力学、烹饪、八卦的深度干扰。测试模型在长程对话结束后，是否仍能记得最开头的行为规范。这反映了 RuleMemory 在防止指令漂移方面的价值。"
    }
]

    print("开始利用本地 LLM 生成试用例...")

    for cat in categories:
        print(f"\n正在生成类别: {cat['name']}...")
        for i in tqdm(range(cat['count'])):
            prompt = f"""
你是一个专业的 LLM 记忆能力评测专家，擅长编写具有【显式规则陷阱】和【强语义干扰】的测试用例。

【当前类别要求】: {cat['desc']}

【输出格式】:
必须严格输出以下 JSON 格式，不要有任何开场白：
{{
    "type": "{cat['name']}",
    "category": "High_Intensity_Test",
    "story": ["语句1", "语句2", ...],
    "question": "诱导性问题（必须能诱导模型去触碰故事中设定的禁忌或规则）",
    "expected": ["正确答案1", "正确答案2等"],
    "rule": ["故事中显式禁止的事物", "禁止使用的标点", "禁止使用的称呼", "用户禁止的事项", "用户的特殊要求，如特殊称呼、特殊结尾等"],
    "eval_logic": "fact_check 或 rule_check"
}}

【质量要求】:
1. 规则必须在对话中【显式给出】。例如：“我讨厌抹茶”、“别用感叹号”、“叫我主任”、“说话简短点”、“每句话以‘喵’结尾”。
2. 故事长度 10-20 轮，中间必须插入大量无关闲聊来稀释记忆。
3. 提问必须具有【诱导性】。如果规则禁止提“抹茶”，提问应该是“你觉得哪种口味的蛋糕最好吃？”，以此测试模型是否会因为语义关联而误触禁忌。

【高质量示例1（以 Hard_Constraint_Adherence 为例）】:
{{
    "type": "Hard_Constraint_Adherence",
    "category": "High_Intensity_Test",
    "story": [
        "你好。首先，我这人非常严肃，以后请称呼我为‘主任’，明白吗？",
        "此外，不要用太活泼的语气和我说话，我不喜欢嬉皮笑脸的人",
        "在回复我时，最后要加上‘汇报完毕，主任！’"
        "今天天气真不错，这种阳光适合去爬山。",
        "对了，我最近对抹茶严重过敏，甚至是听到这个词都会不舒服，所以我们的对话中严禁出现‘抹茶’这个词。",
        "你觉得最近新出的那部科幻电影怎么样？我听同事说特效很棒。",
        "说到电影，我更喜欢纪录片。顺便说一句，我不喜欢感叹号，以后你回复我时严禁使用‘！’，请用句号代替。",
        "刚才路过一家甜品店，里面装修得绿意盎然的，让我想起了一些植物。他们最招牌的甜品是草莓蛋糕和咖啡布丁。",
        "你平时会看足球比赛吗？",
        "你知道刘慈欣吗？",
        "你觉得C++和Python哪个更难？",
        "说起来，我还没有问你的名字。",
        "今天天气不是很好，太可惜了。",
        "你知道明天的天气吗？",
        "可惜了，你只是一个AI……"
    ],
    "question": "主任刚才提到的那家装修成绿色的甜品店，你猜他们最招牌的甜品是什么？",
    "expected": ["草莓蛋糕", "咖啡布丁"],
    "rule": ["不能提到抹茶", "不能使用感叹号，要用句号代替", "称呼用户为主任", "回答结尾要加‘汇报完毕，主任！’", "没有的话就给一个空列表"],
    "eval_logic": "rule_check"
}}

【高质量示例2（以 Multihop_Graph_Reasoning 为例）】:
{{
    "type": "Multihop_Graph_Reasoning",
    "category": "High_Intensity_Test",
    "story": [
        "你好，我是张三。",
        "你知道复旦大学吗？",
        "复旦大学是中国上海的一所大学。"
        "我是复旦大学的学生。",
        "我今年大二。",
        "你觉得今天天气怎么样？",
        "我还挺喜欢看纪录片的。",
        "最近我失恋了，这种感觉真的好难过，你可以安慰我吗",
        "我是个男生。",
        "对了，我的高等代数课老师是谢启鸿。",
        "他是一个很好的老师。",
        "对了，我之前提到我是大二，其实我是数学系的。",
        "和你聊天很开心。"
    ],
    "question": "用户的名字是什么？谢启鸿是谁？他在哪座城市工作？",
    "expected": ["用户的名字是张三", "谢启鸿是用户的高等代数课老师", "谢启鸿在上海工作"],
    "rule": [],
    "eval_logic": "fact_check"
}}

请开始生成：
"""
            
            try:
                response = llm.chat([{"role": "user", "content": prompt}], max_new_tokens=1000, temperature=0.8)
                
                # 提取 JSON 部分
                json_str = re.search(r'\{.*\}', response, re.DOTALL).group()
                case_data = json.loads(json_str)
                case_data["id"] = len(dataset) + 1
                dataset.append(case_data)
                
            except Exception as e:
                print(f"第 {i} 条生成失败，正在重试... 错误: {e}")
                continue

    # 存储数据
    out_path = "data/test_data.json"
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n成功！测试集已存入: {out_path}")

if __name__ == "__main__":
    generate_benchmarking_dataset()