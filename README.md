# TCM-Doctor-AI

"""中医AI Streamlit 完整 Demo 模板

功能：
- Streamlit 界面
- 本地文档构建知识库（RAG）
- Agent 调用检索工具
- 支持聊天式问诊演示

建议安装：
    pip install streamlit openai faiss-cpu numpy pypdf python-dotenv

可选安装（如果你想用本地 embeddings）：
    pip install sentence-transformers

运行：
    streamlit run tcm_streamlit_agent_rag_demo.py

准备数据：
- 把中医资料放到 ./data/ 目录下
- 支持 txt / md / pdf

环境变量：
- OPENAI_API_KEY
- OPENAI_MODEL（可选，默认 gpt-4.1-mini）
- OPENAI_EMBEDDING_MODEL（可选，默认 text-embedding-3-small）
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


APP_TITLE = "中医AI Agent + RAG Demo"
DATA_DIR = Path("data")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = 4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80


# ----------------------------
# Utilities
# ----------------------------

def safe_load_dotenv() -> None:
    if load_dotenv is not None:
        load_dotenv()


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("读取 PDF 需要安装 pypdf")
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)
    raise ValueError(f"不支持的文件类型: {path.suffix}")


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


@dataclass
class DocChunk:
    text: str
    source: str
    chunk_id: int


# ----------------------------
# Embeddings and vector store
# ----------------------------

class EmbeddingBackend:
    def __init__(self) -> None:
        self.client = None
        self.local_model = None
        self.mode = None

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI is not None:
            self.client = OpenAI(api_key=api_key)
            self.mode = "openai"
        else:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self.local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self.mode = "local"
            except Exception:
                self.mode = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.mode == "openai":
            assert self.client is not None
            resp = self.client.embeddings.create(model=DEFAULT_EMBEDDING_MODEL, input=texts)
            vectors = [item.embedding for item in resp.data]
            return np.array(vectors, dtype=np.float32)
        if self.mode == "local":
            assert self.local_model is not None
            vectors = self.local_model.encode(texts, normalize_embeddings=True)
            return np.array(vectors, dtype=np.float32)
        raise RuntimeError(
            "没有可用的 embedding 后端。请配置 OPENAI_API_KEY，或安装 sentence-transformers。"
        )

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.embed_texts([text])
        return vec[0]


class VectorStore:
    def __init__(self, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self.index = None
        self.chunks: List[DocChunk] = []
        self.dim = None

    def build(self, chunks: List[DocChunk]) -> None:
        if faiss is None:
            raise RuntimeError("需要安装 faiss-cpu")
        if not chunks:
            raise RuntimeError("没有可索引的文档内容")

        texts = [c.text for c in chunks]
        vectors = self.backend.embed_texts(texts)
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)

        if self.backend.mode == "openai":
            faiss.normalize_L2(vectors)
        elif self.backend.mode == "local":
            pass

        self.index.add(vectors)
        self.chunks = chunks

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[DocChunk, float]]:
        if self.index is None:
            raise RuntimeError("向量库尚未构建")
        qvec = self.backend.embed_query(query).astype(np.float32)[None, :]
        if self.backend.mode == "openai":
            faiss.normalize_L2(qvec)
        scores, idxs = self.index.search(qvec, top_k)
        results: List[Tuple[DocChunk, float]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results


# ----------------------------
# Document ingestion
# ----------------------------

def load_documents_from_dir(data_dir: Path) -> List[DocChunk]:
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        return []

    all_chunks: List[DocChunk] = []
    files = sorted([p for p in data_dir.rglob("*") if p.is_file()])
    for path in files:
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        try:
            text = read_text_file(path)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(DocChunk(text=chunk, source=str(path.relative_to(data_dir)), chunk_id=i))
        except Exception as e:
            st.warning(f"跳过文件 {path.name}，原因：{e}")
    return all_chunks


# ----------------------------
# Agent layer
# ----------------------------

SYSTEM_PROMPT = """你是一个中医AI演示助手，目标是帮助用户做知识检索和结构化分析。

要求：
1. 你必须优先依据检索到的资料回答。
2. 不要把输出写成绝对诊断，使用“可能”“倾向于”“建议进一步确认”等谨慎措辞。
3. 如果输入包含急症、胸痛、意识障碍、大出血、呼吸困难等危险信号，先提示立即就医。
4. 输出必须包含以下字段：
   - 主要判断
   - 可能证候
   - 证据依据
   - 建议下一步
   - 风险提示
5. 文字简洁，适合比赛演示。
"""


def format_retrieved_context(results: List[Tuple[DocChunk, float]]) -> str:
    if not results:
        return "未检索到相关资料。"
    lines = []
    for i, (chunk, score) in enumerate(results, 1):
        lines.append(
            f"[{i}] 来源: {chunk.source} | 片段: {chunk.chunk_id} | 相似度: {score:.3f}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def detect_red_flags(text: str) -> bool:
    flags = ["胸痛", "呼吸困难", "昏迷", "意识障碍", "大出血", "抽搐", "高热不退", "剧烈腹痛"]
    return any(flag in text for flag in flags)


class AgentRunner:
    def __init__(self, vectorstore: VectorStore) -> None:
        self.vectorstore = vectorstore
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key and OpenAI is not None else None
        self.model = DEFAULT_MODEL

    def retrieve(self, question: str) -> Dict[str, Any]:
        results = self.vectorstore.search(question, top_k=TOP_K)
        context = format_retrieved_context(results)
        return {"results": results, "context": context}

    def answer(self, user_input: str, chat_history: List[Dict[str, str]]) -> str:
        if self.client is None:
            return (
                "当前没有可用的 OpenAI 接口。请配置 OPENAI_API_KEY 后再运行完整 Agent。\n\n"
                "你也可以先用检索面板查看知识库内容。"
            )

        retrieval = self.retrieve(user_input)
        context = retrieval["context"]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-6:]])

        prompt = f"""{SYSTEM_PROMPT}

对话历史:
{history_text if history_text else '无'}

用户输入:
{user_input}

检索到的资料:
{context}

请输出结构化结果。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""


# ----------------------------
# Streamlit UI
# ----------------------------

@st.cache_resource(show_spinner=True)
def build_system() -> Tuple[EmbeddingBackend, Optional[VectorStore], List[DocChunk]]:
    backend = EmbeddingBackend()
    chunks = load_documents_from_dir(DATA_DIR)
    if not chunks:
        return backend, None, []
    store = VectorStore(backend)
    store.build(chunks)
    return backend, store, chunks


def sidebar_panel(backend: EmbeddingBackend, chunks: List[DocChunk]) -> None:
    st.sidebar.title("控制面板")
    st.sidebar.write(f"Embedding 后端: **{backend.mode or '不可用'}**")
    st.sidebar.write(f"已加载切片数: **{len(chunks)}**")
    st.sidebar.caption("把 txt / md / pdf 放进 data 目录后，刷新页面即可重建缓存。")

    with st.sidebar.expander("演示建议", expanded=True):
        st.write("1. 先展示知识库检索")
        st.write("2. 再展示 Agent 结构化输出")
        st.write("3. 最后强调安全提示和可追溯证据")


def render_retrieval_tab(vectorstore: Optional[VectorStore]) -> None:
    st.subheader("知识库检索")
    query = st.text_input("输入一个中医问题", placeholder="例如：头痛、发热、无汗，可能是什么证候？")
    if st.button("检索", type="primary"):
        if vectorstore is None:
            st.error("知识库还没有构建。请先在 data 目录放入资料。")
            return
        if not query.strip():
            st.warning("请输入检索问题。")
            return
        results = vectorstore.search(query)
        if not results:
            st.info("没有找到相关内容。")
            return
        for i, (chunk, score) in enumerate(results, 1):
            with st.expander(f"结果 {i} | {chunk.source} | 片段 {chunk.chunk_id} | score={score:.3f}"):
                st.write(chunk.text)


def render_agent_tab(vectorstore: Optional[VectorStore]) -> None:
    st.subheader("Agent 问诊 Demo")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("输入症状，例如：咽痛、发热、口渴、怕冷")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if detect_red_flags(user_input):
            warning = "输入中出现危险信号，建议立即线下就医或急诊评估。"
            st.session_state.chat_history.append({"role": "assistant", "content": warning})
            with st.chat_message("assistant"):
                st.error(warning)
            return

        if vectorstore is None:
            assistant_msg = "知识库尚未构建，请先在 data 目录放入中医资料。"
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
            with st.chat_message("assistant"):
                st.info(assistant_msg)
            return

        agent = AgentRunner(vectorstore)
        with st.chat_message("assistant"):
            with st.spinner("Agent 分析中..."):
                answer = agent.answer(user_input, st.session_state.chat_history)
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


def render_prompt_tab() -> None:
    st.subheader("可直接拿去改的提示词")
    st.code(
        SYSTEM_PROMPT,
        language="text",
    )


def main() -> None:
    safe_load_dotenv()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("单文件版，适合比赛 Demo 和快速原型。")

    backend, vectorstore, chunks = build_system()
    sidebar_panel(backend, chunks)

    tabs = st.tabs(["Agent 问诊", "知识库检索", "提示词"])
    with tabs[0]:
        render_agent_tab(vectorstore)
    with tabs[1]:
        render_retrieval_tab(vectorstore)
    with tabs[2]:
        render_prompt_tab()

    with st.expander("部署说明", expanded=False):
        st.markdown(
            """
            **最小部署流程**
            1. 把资料放到 `data/` 目录。
            2. 配置 `OPENAI_API_KEY`。
            3. 安装依赖。
            4. 运行 `streamlit run tcm_streamlit_agent_rag_demo.py`。

            **比赛展示顺序**
            先展示检索命中，再展示 Agent 输出，再展示风险提示。
            """
        )


if __name__ == "__main__":
    main()
