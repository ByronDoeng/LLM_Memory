import argparse
import wandb
from src.agent import DualMemoryAgent
from src.llm_engine import LLMEngine
from termcolor import colored

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_vector", action="store_true", help="禁用向量记忆")
    parser.add_argument("--no_rules", action="store_true", help="禁用规则记忆")
    parser.add_argument("--no_graph", action="store_true", help="禁用图记忆")
    parser.add_argument("--project_name", type=str, default="pj_hzdong_LLMmemory")
    args = parser.parse_args()

    # 初始化 WandB
    wandb.init(project=args.project_name, config=args)

    print(colored("正在初始化双重记忆 Agent...", "cyan"))
    shared_llm = LLMEngine() 
    agent = DualMemoryAgent(
        llm_engine=shared_llm,
        use_vector=not args.no_vector,
        use_rules=not args.no_rules,
        use_graph=not args.no_graph
    )

    print(colored("\n系统就绪！开始对话 (输入 'exit' 退出)", "green"))
    print(colored("提示：尝试告诉模型你的喜好，然后故意说它错了，看它能否记住。", "yellow"))

    while True:
        try:
            user_input = input(colored("\nUser: ", "blue"))
            if user_input.lower() in ["exit", "quit"]:
                break
            
            response = agent.chat(user_input)
            print(colored(f"Agent: {response}", "green"))
            
        except KeyboardInterrupt:
            break

    wandb.finish()

if __name__ == "__main__":
    main()