# 定期总结vector库

import uuid

class SummaryMemory:
    def __init__(self, vector_memory, llm_engine, trigger_threshold=5):
        self.vector_mem = vector_memory
        self.llm = llm_engine
        self.threshold = trigger_threshold
        self.buffer = [] # 暂存对话历史

    def add_turn(self, user_input, assistant_response):
        """每轮对话后调用"""
        self.buffer.append(f"User: {user_input}\nAI: {assistant_response}")
        
        # 达到阈值，触发总结
        if len(self.buffer) >= self.threshold:
            # print(f"【DEBUG】触发对话总结 (Buffer={len(self.buffer)})...")
            self._summarize_and_store()

    def _summarize_and_store(self):
        context = "\n".join(self.buffer)
        
        prompt = f"""
        【任务】
        请简要总结以下对话的核心信息。
        
        【对话内容】
        {context}
        
        【要求】
        1. 包含关键事实（如人名、偏好、事件）。
        2. 忽略寒暄和废话。
        3. 总结为一段话，不超过100字。
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            summary = self.llm.chat(messages, temperature=0.1, max_new_tokens=200)
            
            # print(f"【DEBUG】生成摘要 -> {summary}")
            
            # 将摘要存入向量库，标记为 'summary' 类型
            self.vector_mem.add_memory(
                user_input="[系统摘要]", 
                assistant_response=summary, 
                metadata={"type": "summary"}
            )
            
            # 清空缓冲区
            self.buffer = []
            
        except Exception as e:
            print(f"【DEBUG】摘要生成失败: {e}")