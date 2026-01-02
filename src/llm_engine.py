# llm加载与推理封装

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download

class LLMEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(current_dir, "cache", "models")

        print(f"正在尝试通过 ModelScope 下载/加载模型: {model_name} ...")
        try:
            model_dir = snapshot_download(model_name, cache_dir=cache_dir)
            print(f"模型已下载至: {model_dir}")
        except Exception as e:
            print(f"ModelScope 下载失败: {e}")
            raise e
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("模型加载完成。")

    def chat(self, messages, max_new_tokens=512, temperature=0.7):
        """
        messages 格式: [{"role": "user", "content": "..."}, ...]
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response