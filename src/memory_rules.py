# 规则记忆

import json
import os

class RuleMemory:
    def __init__(self, file_path=None):
        if file_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(current_dir, "cache/memory", "rules.json")

        self.file_path = file_path
        self.rules = self._load_rules()
        # print(f"【DEBUG】规则文件路径为: {self.file_path}")

    def _load_rules(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # print(f"【DEBUG】成功加载 {len(data)} 条规则")
                    return data
            except Exception as e:
                print(f"【DEBUG】加载规则文件出错: {e}")
                return []
        # print("【DEBUG】规则文件不存在，将在第一条规则生成时创建。")
        return []

    def _save_rules(self):
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.rules, f, ensure_ascii=False, indent=2)
            # print(f"【DEBUG】规则已写入硬盘! 当前规则数: {len(self.rules)}")
        except Exception as e:
            print(f"【DEBUG】写入规则文件失败: {e}")

    def get_rules_text(self):
        if not self.rules:
            return "暂无特殊规则。"
        return "\n".join([f"- {r}" for r in self.rules])

    def get_rules_text_list(self):
        if not self.rules:
            return []
        return [r for r in self.rules]

    def add_rule(self, rule):
        # 去除首尾空格
        rule = rule.strip()
        if rule not in self.rules:
            self.rules.append(rule)
            self._save_rules()
        else:
            # print("【DEBUG】规则已存在，跳过保存。")
            pass

    def reflect_and_extract(self, llm_engine, user_input, last_response):
        # print(f"\n>>> [反思开始] 正在分析用户反馈: {user_input}")
        
        prompt = f"""
        【任务】
        用户对上一次的回答表示了不满或纠正，或提出了新的要求。请分析对话，提取一条简短明确的“用户偏好”或“禁忌规则”。
        
        【上下文】
        模型回答: {last_response}
        用户反馈: {user_input}
        
        【要求】
        1. 只输出规则内容，不要包含"好的"、"规则是"等废话。
        2. 如果无法提取规则，请输出 "None"。
        3. 规则必须简短（20字以内）。
        4. 如果用户是在纠正知识性错误，请输出 None，不要将其作为行为规则。
        
        【示例】
        1，输入：我不吃辣。 -> 输出：用户不吃辣
        2，输入：别用感叹号！ -> 输出：禁止使用感叹号
        3，输入：艾雅法拉是术士，不是医疗。 -> 输出：艾雅法拉的职业是术士
        4，输入：称呼我为主人。 -> 输出：称呼用户为主人
        5，输入：每句话必须以‘Over’结尾。 -> 输出：每句话必须以‘Over’结尾
        """
        
        messages = [{"role": "user", "content": prompt}]
        rule = llm_engine.chat(messages, temperature=0.3, max_new_tokens=100)
        
        # print(f">>> [LLM 原始输出]: {rule}")
        
        # 清洗数据
        rule = rule.replace('"', '').replace("'", "").strip()
        
        if "None" not in rule and len(rule) > 1:
            # 只要不是太长，就存下来
            if len(rule) < 100:
                # print(f">>> [反思成功] 提取到新规则: {rule}")
                self.add_rule(rule)
                return rule
            else:
                print(f">>> [反思失败] 提取的规则太长了 ({len(rule)}字)，被丢弃。")
        else:
            # print(">>> [反思失败] LLM 认为没有规则或输出了 None")
            pass
            
        return None