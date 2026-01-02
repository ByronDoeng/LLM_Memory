# 情景记忆

import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import uuid
from sentence_transformers import CrossEncoder
from modelscope import snapshot_download
import os


current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hf_cache_dir = os.path.join(current_dir, "cache", "huggingface")
os.environ["HF_HOME"] = hf_cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = hf_cache_dir
print(f"HuggingFace 缓存目录已设置为: {hf_cache_dir}")

class VectorMemory:
    def __init__(self, persist_dir=None):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(current_dir, "cache", "models")

        if persist_dir is None:
            persist_dir = os.path.join(current_dir, "cache/memory", "chroma_db")

        try:
            print("正在通过 ModelScope 下载 Embedding 模型...")
            embed_model_dir = snapshot_download(
                'AI-ModelScope/all-MiniLM-L6-v2', 
                cache_dir=cache_dir
            )
            print(f"Embedding 模型已就绪: {embed_model_dir}")
        except Exception as e:
            print(f"Embedding 模型下载失败，请检查网络: {e}")
            raise e

        # 初始化ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                allow_reset=True, # 允许重置
                anonymized_telemetry=False
            )
        )
        
        # embedding模型
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model_dir 
        )
        
        self.collection = self.client.get_or_create_collection(
            name="episodic_memory",
            embedding_function=self.emb_fn
        )

        # 打分模型
        try:
            print("正在通过 ModelScope 下载 Re-ranker 模型...")
            rerank_model_dir = snapshot_download(
                'BAAI/bge-reranker-base', 
                cache_dir=cache_dir
            )
            print(f"Re-ranker 模型已就绪: {rerank_model_dir}")
            self.reranker = CrossEncoder(rerank_model_dir)
        except Exception as e:
            print(f"Re-ranker 下载失败，将降级运行 (不使用重排序): {e}")
            self.reranker = None
        


    def add_memory(self, user_input, assistant_response, metadata=None):
        """将一轮对话存入向量库"""
        if metadata is None:
            metadata = {"type": "dialogue"}

        text = f"User: {user_input}\nAssistant: {assistant_response}"

        if metadata.get("type") == "summary":
            text = f"【历史摘要】{assistant_response}"

        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )

    def retrieve(self, query, n_results=10):
        """检索相关的历史对话"""
        results = self.collection.query(query_texts=[query], n_results=n_results*2)
        if not results['documents'] or not results['documents'][0]:
            return []
        docs = results['documents'][0]
        
        if not docs: return []

        if self.reranker:
            pairs = [[query, doc] for doc in docs]
            scores = self.reranker.predict(pairs)
            # 排序后取 Top-5
            sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
            return sorted_docs[:5]
        else:
            # 降级方案
            return docs[:5]

    def clear_all(self):
        """显式清空向量库"""
        try:
            self.client.reset() # 物理清空数据库
            # 重启获取 collection
            self.collection = self.client.get_or_create_collection(
                name="episodic_memory",
                embedding_function=self.emb_fn
            )
        except Exception as e:
            print(f"【DEBUG】向量库重置失败: {e}")