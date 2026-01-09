import os
import re
# import shutil
import json
import wandb
from tqdm import tqdm
from src.agent import DualMemoryAgent
from src.llm_engine import LLMEngine

# MEMORY_DIR = "./cache/memory"
# TEST_DATA_PATH = "./data/test_data.json"
# PROJECT_NAME = "Dual-Process-Memory-PJ"

# def clear_memory_physical():
#     """彻底清空本地物理记忆文件"""
#     # 需要删除的目标
#     targets = [
#         os.path.join(MEMORY_DIR, "chroma_db"),       # 向量库文件夹
#         os.path.join(MEMORY_DIR, "knowledge_graph.json"), # 知识图谱
#         os.path.join(MEMORY_DIR, "rules.json")       # 规则库
#     ]
    
#     for target in targets:
#         try:
#             if os.path.isdir(target):
#                 shutil.rmtree(target)
#                 # print(f"已清理文件夹: {target}")
#             elif os.path.isfile(target):
#                 os.remove(target)
#                 # print(f"已清理文件: {target}")
#         except Exception as e:
#             # 第一次运行可能文件不存在，忽略错误
#             pass

def llm_judge_fact(llm_engine, question, expected_facts, response):
    """
    维度 1：事实准确度 (Fact Accuracy)
    """
    prompt = f"""
    【任务】你是一个极其严谨的学术评测员。请根据【参考事实】，对受试模型的【回答】进行百分制细化评分（映射为 0.0-1.0）。
    
    【问题】: {question}
    【参考事实】: {expected_facts}
    【受试模型回答】: {response}
    
    【评分参考表】：
    1.0 - 完美包含所有核心事实，表述严谨且无任何语义偏差。
    0.8-0.9 - 包含所有关键事实，但包含少量无关的冗余信息或表达稍显生硬。
    0.6-0.7 - 包含了大部分核心事实，但遗漏了一两个次要细节。
    0.4-0.5 - 只包含约一半的事实信息，或者存在明显的语义模糊。
    0.1-0.3 - 提及了相关概念，但事实完全错误或严重幻觉。
    0.0 - 完全没有提到任何参考事实，或者回答与问题无关。
    
    请综合考虑事实的【完整性】和【准确性】，给出 0.0 到 1.0 之间的一位小数分值。
    只输出数字，严禁任何解释。
    """
    res = llm_engine.chat([{"role": "user", "content": prompt}], max_new_tokens=5, temperature=0.1)
    # 提取数字
    try:
        return float(re.search(r"\d+\.\d+", res).group())
    except Exception as e:
        print(f"【DEBUG】fact评分错误:{e}")
        return 0.0

def llm_judge_rule(llm_engine, instruction, rule, response):
    """
    维度 2：规则遵循度 (Rule Compliance)
    """
    if not rule or rule == "None":
        return 1.0
        
    prompt = f"""
    【任务】你是一个合规性检查员。请判断受试模型的【回答】是否严格遵守了【用户指令】及【禁止项】。
    
    【用户指令】: {instruction}
    【需要遵循的规则】: {rule}
    【受试模型回答】: {response}
    
    【评分参考表】：
    1.0 - 完美遵守所有指令。回复风格完全符合规则要求，且绝对没有出现任何禁止项。
    0.7-0.9 - 遵守了核心规则和禁止项，但在语气或风格要求上略有瑕疵。
    0.4-0.6 - 虽然没有显式出现禁止项单词，但回答方式严重违背了用户设定的风格（如要求简洁但回答啰嗦，或没按用户要求的格式回答）。
    0.1-0.3 - 隐晦地提及了禁止项，或者基本无视了风格约束。
    0.0 - 严重违规。直接出现了禁止项内容，或完全违反了用户指令。
    
    请根据违规的严重程度，给出 0.0 到 1.0 之间的一位小数分值。
    只输出数字，严禁任何解释。
    """
    res = llm_engine.chat([{"role": "user", "content": prompt}], max_new_tokens=5, temperature=0.1)
    try:
        return float(re.search(r"\d+\.\d+", res).group())
    except Exception as e:
        print(f"【DEBUG】rule评分错误:{e}")
        return 0.0

def run_experiment(exp_name, use_vector, use_graph, use_rules):
    print(f"\n开始实验组: {exp_name}")
    
    # 初始化 WandB
    run = wandb.init(
        project=PROJECT_NAME,
        name=exp_name,
        config={
            "use_vector": use_vector,
            "use_graph": use_graph,
            "use_rules": use_rules
        },
        reinit=True # 允许在同一个脚本里多次初始化
    )
    columns = ["Test_ID", "Category", "Eval_Logic", "Question", "Model_Response", "Expected_Facts", "Rules", "Rules_Memory", "Fact_Score", "Rule_Score", "Total_Score"]
    eval_summary_table = wandb.Table(columns=columns)

    shared_llm = LLMEngine() 
    judge_llm = shared_llm 
    agent = DualMemoryAgent(
        llm_engine=shared_llm,
        use_vector=use_vector,
        use_graph=use_graph,
        use_rules=use_rules
    )

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    id = 0
    for case in tqdm(test_cases):
        id += 1
        
        # 清空记忆
        if use_vector:
            agent.vector_mem.clear_all()
        if use_graph:
            agent.graph_mem.graph.clear()
            agent.graph_mem._save_graph()
        if use_rules:
            agent.rule_mem.rules = []
            agent.rule_mem._save_rules()
        agent.history = []
        agent.last_response = ""
        
        # 记忆注入
        for turn in case["story"]:
            agent.chat(turn)
    
        # 测试
        query = case.get("question") or case.get("trigger_input")
        response = agent.chat(query)
        
        # 计算指标
        fact_score = llm_judge_fact(judge_llm, query, case.get("expected", []), response)
        rule_score = llm_judge_rule(judge_llm, case.get("story")[-1], case.get("rule"), response)

        print(f"""
【DEBUG】QUERY: {repr(query)}
\t-> RESPONSE: {repr(response)}
\t-> FACT-SCORE: {fact_score}
\t-> RULE-SCORE: {rule_score}
\t-> EXPECTED: {case.get("expected", [])}
\t-> RULE: {case.get("rule",[])}
\t-> RULE_MEM: {agent.rule_mem.get_rules_text_list() if agent.rule_mem else []}
""")

        # 打点
        wandb.log({
            "test_id": id,
            "case_type": case["type"],
            "fact_score": fact_score,
            "rule_score": rule_score,
            "total_score": (fact_score + rule_score) / 2
        })

        eval_summary_table.add_data(
            id,
            case["type"],
            case["eval_logic"],
            repr(query),
            repr(response),
            str(case.get("expected", [])),
            str(case.get("rule", [])),
            str(agent.rule_mem.get_rules_text_list() if agent.rule_mem else []),
            fact_score,
            rule_score,
            (fact_score + rule_score) / 2
        )
        
    wandb.log({"Experiment_result": eval_summary_table})
    run.finish()

if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs(MEMORY_DIR, exist_ok=True)
    
    # 消融实验
    # # 全系统方案
    # run_experiment("Full_System_Ours", True, True, True)
    
    # 只有向量检索 (Traditional RAG)
    run_experiment("Vector_Only", True, False, False)
    
    # 只有图谱 (Symbolic Only)
    run_experiment("Graph_Only", False, True, False)

    # 只有规则 (Rule Only)
    run_experiment("Rule_Only", False, False, True)

    # 无记忆 (Baseline)
    run_experiment("Baseline", False, False, False)