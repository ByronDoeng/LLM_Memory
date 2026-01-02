# 图记忆

import networkx as nx
import json
import os

class GraphMemory:
    def __init__(self, file_path=None):
        if file_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            file_path = os.path.join(current_dir, "cache/memory", "knowledge_graph.json")
        
        self.file_path = file_path
        self.graph = nx.Graph()
        self._load_graph()

    def _load_graph(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                # print(f"【DEBUG】已加载知识图谱，包含 {self.graph.number_of_nodes()} 个实体")
            except Exception as e:
                print(f"【DEBUG】图谱加载失败: {e}")

    def _save_graph(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update_graph(self, llm_engine, user_input, assistant_response):
        """让 LLM 提取三元组并更新图谱"""
        text = f"User: {user_input}\nAI: {assistant_response}"
        
        prompt = f"""
        【任务】
        从下方的对话中提取核心实体关系三元组。
        
        【对话】
        {text}
        
        【要求】
        1. 格式为 JSON 列表：[["实体1", "关系", "实体2"], ...]
        2. 实体必须精简（如 "Alex", "李雷", "喜欢红色"）。
        3. 关系必须明确（如 "是", "职业是", "位于", "偏好"），注意事实求是，给出的三元组必须精确，不能凭空想象。
        4. 如果没有明确事实，输出 []。
        5. 区分清楚 "User" (用户) 和 "Character" (虚构人物)。
        6. 注意区分User的回答和AI的回答，以用户说的为准，有些AI说的并不准确或者与用户说的无关，不应该被处理防止出现幻觉。
        7. **强制归一化**：如果用户提到“我”，实体必须存为 "User"。

        【示例】
        输入：我叫Alex，我是个作家。
        输出：[["Alex", "身份", "用户"], ["Alex", "职业", "作家"]]
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = llm_engine.chat(messages, temperature=0.1, max_new_tokens=200)
        
        try:
            # 清洗数据，提取 JSON 部分
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                triplets = json.loads(json_str)
                
                count = 0
                for head, relation, tail in triplets:
                    # 添加节点和边
                    self.graph.add_node(head)
                    self.graph.add_node(tail)
                    self.graph.add_edge(head, tail, relation=relation)
                    count += 1
                
                if count > 0:
                    # print(f"【DEBUG】图谱新增 {count} 条关系: {triplets}")
                    self._save_graph()
        except Exception as e:
            # print(f"【DEBUG】三元组提取失败: {e}")
            pass

    # def retrieve_multihop(self, entity_start, entity_end):
    #     """
    #     寻找两个实体之间的推理路径
    #     Query: "李雷和Alex是什么关系？"
    #     """
    #     try:
    #         # 使用 NetworkX 寻找最短路径
    #         path = nx.shortest_path(self.graph, source=entity_start, target=entity_end)
            
    #         # 将路径转化为文本
    #         # Path: [李雷, Alex, 红色]
    #         explanation = []
    #         for i in range(len(path)-1):
    #             u, v = path[i], path[i+1]
    #             rel = self.graph[u][v].get('relation', '关联')
    #             explanation.append(f"{u} {rel} {v}")
            
    #         return " -> ".join(explanation)
    #     except nx.NetworkXNoPath:
    #         return None

    def retrieve(self, query, llm_engine):
        """
        1. 让 LLM 提取 query 中的实体 (Entity Linking)
        2. 在图中找这些实体的邻居
        """
        # print(f"【DEBUG】正在对 Query 进行实体提取 -> {query}")
        
        # LLM 提取关键实体
        prompt = f"""
        【任务】
        从用户的提问中提取关键实体，用于在知识图谱中检索。
        
        【用户提问】
        {query}
        
        【要求】
        1. 输出一个 JSON 字符串列表。
        2. 将代词（如"我"、"我的"）转换为标准实体 "User"。
        3. 提取专有名词（如人名、书名、术语）。
        4. 不要输出多余的解释。
        
        【示例】
        输入：我的名字是什么？
        输出：["User"]
        
        输入：李雷的职业是什么？
        输出：["李雷"]
        
        输入：北京折叠是谁写的？
        输出：["北京折叠"]
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = llm_engine.chat(messages, temperature=0.1, max_new_tokens=50)
        
        target_entities = []
        try:
            # 清洗并解析 JSON
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                target_entities = json.loads(json_str)
                # print(f"【DEBUG】LLM 提取到的实体: {target_entities}")
        except Exception as e:
            print(f"【DEBUG】实体提取失败: {e}")
            # 如果 LLM 挂了，回退到简单的关键词匹配
            if "我" in query: target_entities.append("User")
        
        # 图谱检索
        found_facts = []
        for entity in target_entities:
            # 模糊匹配
            matched_nodes = [n for n in self.graph.nodes() if str(n) == entity]
            
            # 如果没完全匹配，尝试包含匹配
            if not matched_nodes:
                 matched_nodes = [n for n in self.graph.nodes() if entity in str(n)]
            
            for node in matched_nodes:
                # 1-hop
                edges = self.graph[node]
                for neighbor, attr in edges.items():
                    relation = attr.get('relation', '相关')
                    fact = f"{node} {relation} {neighbor}"
                    found_facts.append(fact)
                    
                    # 2-hop 邻居
                    # 只有当节点是 "User" 或邻居数较少时才做，防止爆炸
                    if node == "User" or len(self.graph[neighbor]) < 5:
                        if self.graph.has_node(neighbor):
                             sub_edges = self.graph[neighbor]
                             for sub_n, sub_attr in sub_edges.items():
                                 # 防止回到自己
                                 if sub_n != node:
                                     sub_rel = sub_attr.get('relation', '相关')
                                     found_facts.append(f"(推断) {node} 的 {neighbor} {sub_rel} {sub_n}")

        # 去重
        unique_facts = list(set(found_facts))
        if unique_facts:
            # print(f"【DEBUG】检索到的图谱路径: {unique_facts}")
            pass
        return unique_facts