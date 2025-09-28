# app.py
# -*- coding: utf-8 -*-
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor

# ------------------------ 页面配置 ------------------------
st.set_page_config(page_title="RAG_demo", page_icon="🦜🔗", layout="wide")
st.title("🦜🔗  AI法律助手")

# ------------------------ 配置区 ------------------------
class Config:
    # 模型（按你的实际路径修改）
    EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "/mnt/workspace/LLM/BAAI/bge-small-zh-v1___5")
    LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "/mnt/workspace/LLM/Qwen/Qwen3-4B-Instruct-2507")

    # 数据 & 持久化
    DATA_DIR = os.getenv("DATA_DIR", "./data")                # 放 *.json 法条的目录
    VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_db") # Chroma 向量库
    PERSIST_DIR = os.getenv("PERSIST_DIR", "./storage")       # LlamaIndex 存储

    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chinese_labor_laws")
    DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
    DEFAULT_CUTOFF = float(os.getenv("SIM_CUTOFF", "0.75"))

# ------------------------ 模板 ------------------------
LEGAL_QA_TMPL = """
你是严格的法律助理。仅依据“已检索到的法律条文”回答中文问题：
- 不要输出与条文无关的内容，不要臆测。
- 如条文不足以回答，请答：“未在已提供的法律条文中找到明确依据。”
- 回复时尽量引用相关法名称与条号（自然语言表述，无需链接）。

【已检索到的法律条文】：
{context_str}

【用户问题】：
{query_str}

【你的回答】：
""".strip()
RESPONSE_TEMPLATE = PromptTemplate(LEGAL_QA_TMPL)

# ------------------------ 数据加载&校验 ------------------------
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    json_files = list(Path(data_dir).glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"未在 {data_dir} 找到任意 JSON 文件")

    all_data = []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ValueError(f"文件 {jf.name} 根元素应为列表")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {jf.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str) or not v.strip():
                            raise ValueError(f"{jf.name} 中键 '{k}' 的值不是非空字符串")
                all_data.extend({"content": item, "metadata": {"source": jf.name}} for item in data)
            except Exception as e:
                raise RuntimeError(f"加载文件 {jf} 失败: {str(e)}")
    return all_data

def build_nodes(raw_data: List[Dict]) -> List[TextNode]:
    nodes: List[TextNode] = []
    for entry in raw_data:
        law_dict = entry["content"]
        src = entry["metadata"]["source"]
        for full_title, content in law_dict.items():
            node_id = f"{src}::{full_title}"
            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"
            nodes.append(
                TextNode(
                    text=content.strip(),
                    id_=node_id,
                    metadata={
                        "law_name": law_name,
                        "article": article,
                        "full_title": full_title,
                        "source_file": src,
                        "content_type": "legal_article",
                    },
                )
            )
    return nodes

def chunk_nodes(nodes: List[TextNode], chunk_size: int = 512, chunk_overlap: int = 64) -> List[TextNode]:
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked: List[TextNode] = []
    for n in nodes:
        pieces = splitter.split_text(n.text)
        for i, t in enumerate(pieces):
            chunked.append(
                TextNode(
                    text=t,
                    id_=f"{n.id_}#chunk{i}",
                    metadata=dict(n.metadata),
                )
            )
    return chunked

# ------------------------ 向量库与索引 ------------------------
def create_or_load_index(
    nodes: Optional[List[TextNode]],
    force_rebuild: bool,
    top_k: int,
    sim_cutoff: float,
):
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)

    # 强制重建：删除旧集合避免重复
    if force_rebuild:
        try:
            chroma_client.delete_collection(Config.COLLECTION_NAME)
            st.toast(f"已删除旧集合：{Config.COLLECTION_NAME}", icon="🗑️")
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    need_build = (chroma_collection.count() == 0) and (nodes is not None)

    if force_rebuild or need_build:
        st.info(f"开始创建新索引（{0 if nodes is None else len(nodes)} 个节点）...")
        index = VectorStoreIndex(nodes or [], storage_context=storage_context, show_progress=True)
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        st.info("加载已有索引...")
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection),
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model,
        )

    st.caption(f"DocStore记录数：{len(storage_context.docstore.docs)} | Chroma向量条数：{chroma_collection.count()}")
    # 组装 query_engine（加入模板与相似度阈值）
    query_engine = index.as_query_engine(
        similarity_top_k=max(3, top_k),
        response_mode="compact",
        text_qa_template=RESPONSE_TEMPLATE,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=sim_cutoff)],
        verbose=False,
    )
    return query_engine

# ------------------------ 模型初始化（缓存） ------------------------
@st.cache_resource(show_spinner=True)
def init_models(
    embed_path: str,
    llm_path: str,
):
    # 嵌入模型（bge 系列建议 cosine + 归一化；不同版本可能参数名不同，这里保持默认）
    embed_model = HuggingFaceEmbedding(
        model_name=embed_path,
        device="cuda",
    )
    Settings.embed_model = embed_model

    # 生成模型
    llm = HuggingFaceLLM(
        model_name=llm_path,
        tokenizer_name=llm_path,
        model_kwargs={
            "trust_remote_code": True,
            "device_map": "auto",
        },
        tokenizer_kwargs={"trust_remote_code": True},
        generate_kwargs={"temperature": 0.3, "max_new_tokens": 768},
    )
    Settings.llm = llm

    # 简要验证
    try:
        test_vec = embed_model.get_text_embedding("测试文本")
        st.caption(f"✅ 嵌入模型加载完成，维度：{len(test_vec)}")
    except Exception as e:
        st.error(f"嵌入模型验证失败：{e}")

    return True  # 仅占位

# ------------------------ 侧边栏配置 ------------------------
with st.sidebar:
    st.header("⚙️ 配置")
    embed_path = st.text_input("Embedding 模型路径", Config.EMBED_MODEL_PATH)
    llm_path = st.text_input("LLM 模型路径", Config.LLM_MODEL_PATH)
    data_dir = st.text_input("数据目录（*.json）", Config.DATA_DIR)
    chunk_size = st.number_input("分块长度（tokens）", 128, 2048, 512, 16)
    chunk_overlap = st.number_input("分块重叠", 0, 512, 64, 8)
    top_k = st.number_input("Top-K", 1, 20, Config.DEFAULT_TOP_K, 1)
    sim_cutoff = st.slider("相似度阈值（越高越严格）", 0.0, 0.99, Config.DEFAULT_CUTOFF, 0.01)
    force_rebuild = st.checkbox("强制重建索引（清空集合后重建）", value=False)
    init_btn = st.button("初始化 / 重新加载")

# ------------------------ 初始化流程 ------------------------
if "query_engine" not in st.session_state or init_btn:
    with st.spinner("正在加载模型与索引..."):
        # 1) 模型
        init_models(embed_path, llm_path)

        # 2) 数据（仅在需要重建或库为空时加载）
        raw_data = build_nodes(chunk_nodes(build_nodes(load_and_validate_json_files(data_dir)), chunk_size, chunk_overlap))  # 占位避免 mypy 警告
        # 上面写在一行可读性差，实际分步执行更清晰：
        try:
            raw = load_and_validate_json_files(data_dir)
            nodes = build_nodes(raw)
            nodes = chunk_nodes(nodes, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception as e:
            st.error(f"数据加载失败：{e}")
            nodes = None

        # 3) 索引
        try:
            st.session_state["query_engine"] = create_or_load_index(
                nodes=nodes, force_rebuild=force_rebuild, top_k=int(top_k), sim_cutoff=float(sim_cutoff)
            )
            st.success("索引就绪 ✅")
        except Exception as e:
            st.error(f"索引创建/加载失败：{e}")

# ------------------------ 对话框 & 功能 ------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是 AI 小聚。请提出你的法律问题（仅依据已导入法条作答）。"}
    ]

# 展示历史消息
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "聊天记录已清空。请提出你的法律问题。"}
    ]

st.sidebar.button("🧹 清空聊天记录", on_click=clear_chat_history)

# 处理输入
prompt = st.chat_input("请输入法律相关问题...")
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
                    placeholder = st.empty()
                    placeholder.markdown(answer_text)

                    # 溯源展示
                    if getattr(resp, "source_nodes", None):
                        st.markdown("**支持依据：**")
                        for i, node in enumerate(resp.source_nodes, 1):
                            meta = node.metadata or {}
                            with st.expander(f"[{i}] {meta.get('full_title','未知标题')}  | 相似度：{getattr(node,'score',None)}"):
                                st.write(f"来源文件：{meta.get('source_file','-')}")
                                st.write(f"法律名称：{meta.get('law_name','-')}")
                                st.write(f"条款：{meta.get('article','-')}")
                                st.write("---")
                                st.write(node.text)

                    st.caption(f"耗时：{time.time()-t0:.2f}s")
                except Exception as e:
                    st.error(f"检索或生成失败：{e}")
                    answer_text = f"检索或生成失败：{e}"

    st.session_state.messages.append({"role": "assistant", "content": answer_text})

# ------------------------ 运行提示 ------------------------
with st.sidebar.expander("📦 使用说明", expanded=False):
    st.markdown(
        """
**数据格式**：`./data/*.json`，每个文件是列表，列表元素为字典，形如：
```json
[
  {"劳动法 第三十六条": "……条文内容……"},
  {"劳动法 第四十一条": "……条文内容……"}
]
