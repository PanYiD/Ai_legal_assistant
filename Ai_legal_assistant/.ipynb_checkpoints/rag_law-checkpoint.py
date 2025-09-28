# -*- coding: utf-8 -*-
import json
import time
from pathlib import Path
from typing import List, Dict

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.vector_stores import ChromaVectorStore  # ✅ 新版导入
from llama_index.vector_stores.chroma import ChromaVectorStore  # ✅ 改成这样


# from llama_index.llms.openai_like import OpenAILikeLLM
# from llama_index.core import Settings
from llama_index.llms.vllm import Vllm



from llama_index.core import PromptTemplate

QA_TEMPLATE = (
    "<|im_start|>system\n"
    "你是一个专业的法律助手，请严格根据以下法律条文回答问题：\n"
    "相关法律条文：\n{context_str}\n<|im_end|>\n"
    "<|im_start|>user\n{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response_template = PromptTemplate(QA_TEMPLATE)

# ================== 配置区 ==================
class Config:
    EMBED_MODEL_PATH = r"/mnt/workspace/LLM/BAAI/bge-small-zh-v1___5"
    LLM_MODEL_PATH = r"/mnt/workspace/LLM/Qwen/Qwen3-4B-Instruct-2507"
    
    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./chroma_db"
    PERSIST_DIR = "./storage"
    
    COLLECTION_NAME = "chinese_labor_laws"
    TOP_K = 3
    FORCE_REBUILD = False  # True 时强制重建索引


# ================== 初始化模型 ==================


# 配置vLLM服务端参数
class VLLMConfig:
    API_BASE = "http://localhost:8000/v1"  # vLLM的默认端点
    MODEL_NAME = "DeepSeek-R1-Distill-Qwen-1___5B"
    API_KEY = "no-key-required"  # vLLM默认不需要密钥
    TIMEOUT = 60  # 请求超时时间

# # 初始化LLM（替换原来的HuggingFaceLLM）
# def init_vllm_llm():
#     return OpenAILikeLLM(
#         model=VLLMConfig.MODEL_NAME,
#         api_base=VLLMConfig.API_BASE,
#         api_key=VLLMConfig.API_KEY,
#         temperature=0.3,
#         max_tokens=1024,
#         timeout=VLLMConfig.TIMEOUT,
#         is_chat_model=True,  # 适用于对话模型
#         additional_kwargs={"stop": ["<|im_end|>"]}  # DeepSeek的特殊停止符
#     )

# # 在全局设置中配置
# Settings.llm = init_vllm_llm()


def init_models():
    """初始化模型并验证"""
    # Embedding模型
    embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL_PATH,
        device="cuda"  # ✅ 显式指定
    )
    
    # LLM
    llm = HuggingFaceLLM(
        model_name=Config.LLM_MODEL_PATH,
        tokenizer_name=Config.LLM_MODEL_PATH,
        model_kwargs={
            "trust_remote_code": True,
            "device_map": "auto"
        },
        tokenizer_kwargs={"trust_remote_code": True},
        generate_kwargs={"temperature": 0.3}
    )
    
    # # LLM
    # llm = Vllm(
    #         model=Config.LLM_MODEL_PATH,
    #         # tokenizer_name=Config.LLM_MODEL_PATH,
    #         tensor_parallel_size=1,       # 根据 GPU 数量调整
    #         trust_remote_code=True,
    #         # gpu_memory_utilization=0.9,   # 显存利用率
    #         # max_model_len=4096,           # 取决于模型
    #         # dtype="bfloat16"              # 可以改为 "float16"
    #     )

    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # 验证模型
    test_embedding = embed_model.get_text_embedding("测试文本")
    print(f"Embedding维度验证：{len(test_embedding)}")
    
    return embed_model, llm


# ================== 数据处理 ==================
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件"""
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"未找到JSON文件于 {data_dir}"
    
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                all_data.extend({
                    "content": item,
                    "metadata": {"source": json_file.name}
                } for item in data)
            except Exception as e:
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")
    
    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data


def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """构建 TextNode 节点"""
    nodes = []
    for entry in raw_data:
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]
        
        for full_title, content in law_dict.items():
            node_id = f"{source_file}::{full_title}"
            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"
            
            node = TextNode(
                text=content,
                id_=node_id,
                metadata={
                    "law_name": law_name,
                    "article": article,
                    "full_title": full_title,
                    "source_file": source_file,
                    "content_type": "legal_article"
                }
            )
            nodes.append(node)
    
    print(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
    return nodes


# ================== 向量存储 ==================
def init_vector_store(nodes: List[TextNode]) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    if Config.FORCE_REBUILD or (chroma_collection.count() == 0 and nodes is not None):
        print(f"创建新索引（{len(nodes)}个节点）...")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        print("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    print("\n存储验证结果：")
    print(f"DocStore记录数：{len(storage_context.docstore.docs)}")
    return index


# ================== 主程序 ==================
def main():
    embed_model, llm = init_models()
    
    if Config.FORCE_REBUILD or not Path(Config.VECTOR_DB_DIR).exists():
        print("\n初始化数据...")
        raw_data = load_and_validate_json_files(Config.DATA_DIR)
        nodes = create_nodes(raw_data)
    else:
        nodes = None
    
    print("\n初始化向量存储...")
    start_time = time.time()
    index = init_vector_store(nodes)
    print(f"索引加载耗时：{time.time()-start_time:.2f}s")
    
    query_engine = index.as_query_engine(
        similarity_top_k=Config.TOP_K,
        response_mode="compact",  # ✅ 兼容新版
        verbose=True
    )
    
    while True:
        question = input("\n请输入法律相关问题（输入q退出）: ")
        if question.lower() == 'q':
            break
        
        response = query_engine.query(question)
        
        print(f"\n智能助手回答：\n{response.response}")
        print("\n支持依据：")
        for idx, node in enumerate(response.source_nodes, 1):
            meta = node.metadata
            print(f"\n[{idx}] {meta['full_title']}")
            print(f"  来源文件：{meta['source_file']}")
            print(f"  法律名称：{meta['law_name']}")
            print(f"  条款内容：{node.text[:100]}...")
            print(f"  相关度得分：{node.score:.4f}")


if __name__ == "__main__":
    main()
