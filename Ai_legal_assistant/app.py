# app.py
# -*- coding: utf-8 -*-
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext, Settings, PromptTemplate
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ================== 页面与模板 ==================
st.set_page_config(page_title="AI法律助手", page_icon="🦜🔗", layout="wide")
st.title("🦜🔗  AI法律助手")

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
    EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", r"/mnt/workspace/LLM/BAAI/bge-small-zh-v1___5")
    LLM_MODEL_PATH   = os.getenv("LLM_MODEL_PATH",   r"/mnt/workspace/LLM/Qwen/Qwen3-4B-Instruct-2507")

    DATA_DIR      = os.getenv("DATA_DIR", "./data")
    VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_db")
    PERSIST_DIR   = os.getenv("PERSIST_DIR", "./storage")

    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chinese_labor_laws")
    TOP_K = int(os.getenv("TOP_K", "3"))

# ================== 数据处理 ==================
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件"""
    json_files = list(Path(data_dir).glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"未找到JSON文件于 {data_dir}")

    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 验证数据结构
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
    return all_data

def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """添加ID稳定性保障"""
    nodes: List[TextNode] = []
    for entry in raw_data:
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]
        for full_title, content in law_dict.items():
            node_id = f"{source_file}::{full_title}"  # 稳定ID
            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article  = parts[1] if len(parts) > 1 else "未知条款"
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
    return nodes

# ================== 向量存储 / 索引 ==================
def init_vector_store(nodes: Optional[List[TextNode]], force_rebuild: bool = False) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)

    # 可选：强制重建时清空旧集合
    if force_rebuild:
        try:
            chroma_client.delete_collection(Config.COLLECTION_NAME)
            st.toast(f"已删除旧集合：{Config.COLLECTION_NAME}", icon="🗑️")
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    need_build = (chroma_collection.count() == 0) and (nodes is not None)

    if need_build:
        st.info(f"创建新索引（{len(nodes)} 个节点）...")
        # （可选）以下一行不是必须；VectorStoreIndex 会写 docstore，这里只是演示显式添加
        storage_context.docstore.add_documents(nodes)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        # 双重持久化保障
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        st.info("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )

    # 统计信息
    try:
        doc_count = len(storage_context.docstore.docs)
        st.caption(f"DocStore记录数：{doc_count} | Chroma向量条数：{chroma_collection.count()}")
    except Exception:
        pass

    return index

# ================== 模型初始化（缓存） ==================
@st.cache_resource(show_spinner=True)
def init_models(embed_path: str, llm_path: str):
    """初始化模型并在 Settings 中注册。返回 True 表示成功。"""
    embed_model = HuggingFaceEmbedding(
        model_name=embed_path,
        # 如需归一化或设备指定，可在不同版本传参
        # encode_kwargs={"normalize_embeddings": True}
    )
    Settings.embed_model = embed_model

    llm = HuggingFaceLLM(
        model_name=llm_path,
        tokenizer_name=llm_path,
        model_kwargs={
            "trust_remote_code": True,
            # "device_map": "auto"
        },
        tokenizer_kwargs={"trust_remote_code": True},
        generate_kwargs={"temperature": 0.3}
    )
    Settings.llm = llm

    # 验证
    try:
        test_embedding = embed_model.get_text_embedding("测试文本")
        st.caption(f"✅ 嵌入模型加载完成，维度：{len(test_embedding)}")
    except Exception as e:
        st.error(f"嵌入模型验证失败：{e}")

    return True

@st.cache_resource(show_spinner=True)
def build_query_engine(
    data_dir: str,
    force_rebuild: bool,
    top_k: int
):
    """构建或载入索引，并返回 query_engine。"""
    # 若向量库目录不存在或选择强制重建，则加载数据
    nodes: Optional[List[TextNode]] = None
    need_data = force_rebuild or (not Path(Config.VECTOR_DB_DIR).exists())

    if need_data:
        raw = load_and_validate_json_files(data_dir)
        nodes = create_nodes(raw)

    index = init_vector_store(nodes, force_rebuild=force_rebuild)

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        text_qa_template=response_template,
        verbose=False
    )
    return query_engine

# ================== 侧边栏参数 ==================
with st.sidebar:
    st.header("⚙️ 配置")
    embed_path = st.text_input("Embedding 模型路径", Config.EMBED_MODEL_PATH)
    llm_path   = st.text_input("LLM 模型路径",       Config.LLM_MODEL_PATH)
    data_dir   = st.text_input("数据目录（*.json）",  Config.DATA_DIR)

    top_k = st.number_input("Top-K（召回条数）", 1, 20, Config.TOP_K, 1)
    force_rebuild = st.checkbox("强制重建索引（清空集合后重建）", value=False)

    init_btn = st.button("初始化 / 重新加载")

# ================== 初始化（模型 & 索引） ==================
if "query_engine" not in st.session_state or init_btn:
    with st.spinner("正在加载模型与索引..."):
        try:
            init_models(embed_path, llm_path)
            st.session_state["query_engine"] = build_query_engine(
                data_dir=data_dir,
                force_rebuild=force_rebuild,
                top_k=int(top_k),
            )
            st.success("索引就绪 ✅")
        except Exception as e:
            st.error(f"初始化失败：{e}")

# ================== 对话区 ==================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是 AI 小法。请提出你的法律问题（我将严格依据法律条文作答）。"}
    ]

# 展示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "聊天记录已清空。请提出你的法律问题。"}
    ]
st.sidebar.button("🧹 清空聊天记录", on_click=clear_chat_history)

# 用户输入
prompt = st.chat_input("请输入法律相关问题…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# 生成回答
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        if "query_engine" not in st.session_state or st.session_state["query_engine"] is None:
            st.error("检索引擎尚未就绪，请先在侧边栏完成初始化。")
        else:
            with st.spinner("思考中..."):
                t0 = time.time()
                try:
                    resp = st.session_state["query_engine"].query(prompt)
                    answer_text = getattr(resp, "response", str(resp))
                    st.markdown(answer_text)

                    # 溯源展示
                    if getattr(resp, "source_nodes", None):
                        st.markdown("**支持依据：**")
                        for i, node in enumerate(resp.source_nodes, 1):
                            meta = node.metadata or {}
                            sc = getattr(node, "score", None)
                            title = meta.get("full_title", "未知标题")
                            with st.expander(f"[{i}] {title}  | 相似度：{sc}"):
                                st.write(f"来源文件：{meta.get('source_file', '-')}")
                                st.write(f"法律名称：{meta.get('law_name', '-')}")
                                st.write(f"条款：{meta.get('article', '-')}")
                                st.write("---")
                                st.write(node.text)

                    st.caption(f"耗时：{time.time()-t0:.2f}s")
                except Exception as e:
                    st.error(f"检索或生成失败：{e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"检索或生成失败：{e}"})
                    st.stop()

    # 把最终回答加入对话历史
    st.session_state.messages.append({"role": "assistant", "content": answer_text})


# ================== 使用说明 ==================
with st.sidebar.expander("📦 使用说明", expanded=False):
    st.markdown(        """
**数据格式**：`./data/*.json`，每个文件是一个列表，元素是字典，示例：
```json
[
  {"劳动法 第三十六条": "……条文内容……"},
  {"劳动法 第四十一条": "……条文内容……"}
]
 """
)


