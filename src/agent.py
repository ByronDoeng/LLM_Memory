from src.llm_engine import LLMEngine
from src.memory_vector import VectorMemory
from src.memory_rules import RuleMemory
from src.memory_graph import GraphMemory
from src.memory_summary import SummaryMemory
import wandb

class DualMemoryAgent:
    def __init__(self, llm_engine, use_vector=True, use_rules=True, use_graph=True):
        self.llm = llm_engine # 外部模型，防止出现多个实例
        self.vector_mem = VectorMemory() if use_vector else None
        self.rule_mem = RuleMemory() if use_rules else None
        self.graph_mem = GraphMemory() if use_graph else None
        
        self.use_vector = use_vector
        self.use_rules = use_rules
        self.use_graph = use_graph
        
        # 短期记忆 (滑动窗口)
        self.history = [] 
        self.last_response = ""

        # summary初始化
        self.summary_mem = None
        if self.use_vector:
            self.summary_mem = SummaryMemory(self.vector_mem, self.llm, trigger_threshold=5)

        # self.wandb_table = wandb.Table(columns=["User Input", "Response", "Vector Context", "Graph Context", "Rules"])

    def check_if_reflection_needed(self, user_input):
        """使用 LLM 判断是否需要反思，而不是靠关键词"""
        prompt = f"""
            用户输入: "{user_input}"
            请判断：用户这句话是否包含需要记住的规则？规则可以是对某种行为的禁止，比如禁止使用感叹号、省略号；规则也可以是特定的要求，比如称呼用户为主人，或每次对话以over结尾；规则也可以是对某种事实的修正，比如用户认为不是A而是B。
            只回答 "Yes" 或 "No"。
        """
        res = self.llm.chat([{"role": "user", "content": prompt}], max_new_tokens=5)
        return "Yes" in res

    def chat(self, user_input):
        trigger_keywords = ["不对", "错了", "不是", "don't", "wrong", "不喜欢", "别", "禁止", "no", "stop"]

        should_reflect = False
        if self.use_rules and self.last_response:
            if user_input.startswith("/learn"): # 后门指令
                should_reflect = True
            elif any(w in user_input for w in trigger_keywords):
                should_reflect = True
            elif self.check_if_reflection_needed(user_input):
                should_reflect = True
        
        if should_reflect:
            # print(f"【DEBUG】触发反思机制")
            self.rule_mem.reflect_and_extract(self.llm, user_input, self.last_response)
        else:
            if self.last_response:
                # print("【DEBUG】未触发反思 (未检测到关键词)")
                pass

        rules_text = self.rule_mem.get_rules_text() if self.use_rules else "无"

        vector_context = "无"
        if self.use_vector:
            docs = self.vector_mem.retrieve(user_input, n_results=5)
            vector_context = "\n".join(docs) if docs else "无"

        graph_context = "无"
        if self.use_graph:
            facts = self.graph_mem.retrieve(user_input, self.llm)
            if facts:
                graph_context = "\n".join(facts)
                # print(f"【DEBUG】图谱命中事实 -> {graph_context}")
            else:
                graph_context = "无"

        # System Prompt
        system_prompt = system_prompt = f"""
        你是一个拥有【知识图谱】和【向量记忆】的智能助手。

        【关键事实】(来自知识图谱 - 绝对准确)
        {graph_context}

        【历史片段】(来自向量检索 - 仅供参考)
        {vector_context}

        【用户规则】(必须严格遵守)
        {rules_text}

        【指令】
        1. 当【关键事实】与【历史片段】冲突时，信赖【关键事实】。
        2. 区分 "User" (你正在对话的人) 和 "Character" (故事里的人)。
        3. 严格执行【用户规则】中的风格限制。

        请回答：
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-5:]) 
        messages.append({"role": "user", "content": user_input})

        response = self.llm.chat(messages)

        # 更新对话表格
        # self.wandb_table.add_data(user_input, response, vector_context[:100], graph_context, rules_text)
        
        # WandB
        wandb.log({
            # "chat_history": self.wandb_table,
            "user_input": user_input,
            "response": response,
            "vector_context_len": len(vector_context),
            "graph_context_len": len(graph_context),
            "active_rules_count": len(self.rule_mem.rules) if self.rule_mem else 0
        })

        # 更新状态
        self.last_response = response
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        # 更新记忆
        if self.use_vector:
            self.vector_mem.add_memory(user_input, response)
            if self.summary_mem:
                self.summary_mem.add_turn(user_input, response)
        if self.use_graph:
            self.graph_mem.update_graph(self.llm, user_input, response)

        return response
