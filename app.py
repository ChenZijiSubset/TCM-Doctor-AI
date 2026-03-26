from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from src.agent.agent import Agent
from src.inference.ranker import rank_syndromes
from src.rag.embedder import Embedder
from src.rag.vectorstore import VectorStore
from src.safety.guard import has_risk

APP_TITLE = "中医AI Agent Demo"
DATA_FILE = Path("data/knowledge.json")


def load_knowledge():
    if not DATA_FILE.exists():
        raise FileNotFoundError("找不到 data/knowledge.json")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_user_input(text: str):
    symptom_keywords = [
        "恶寒重", "发热轻", "无汗", "头痛", "鼻塞", "流清涕", "咽痛", "黄痰", "口渴",
        "乏力", "自汗", "气短", "低热", "干咳", "口燥咽干", "盗汗", "胸闷", "恶心",
        "痰多", "食少", "便溏", "畏寒", "腹泻", "四肢不温", "腹胀", "便秘", "心悸",
        "失眠", "多梦", "健忘", "头晕", "面色苍白", "刺痛", "痛有定处", "舌暗",
        "口苦", "小便黄", "身重", "神志不清", "言语紊乱", "烦躁"
    ]

    tongue_keywords = [
        "舌苔薄白", "舌苔薄黄", "舌苔黄腻", "舌苔白腻", "舌红", "舌红少苔",
        "舌淡", "舌淡胖", "舌暗", "舌有瘀点", "舌尖红"
    ]

    pulse_keywords = [
        "浮紧", "浮数", "濡数", "弱脉", "虚脉", "细数", "滑脉",
        "弦脉", "沉迟", "沉缓", "细弱", "数脉", "涩脉", "弦涩"
    ]

    symptoms = [k for k in symptom_keywords if k in text]
    tongue = [k for k in tongue_keywords if k in text]
    pulse = [k for k in pulse_keywords if k in text]

    return {
        "symptoms": symptoms,
        "tongue": tongue,
        "pulse": pulse,
    }


def build_ranked_text(results):
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"""[{i}] {r.name}
得分: {r.score}
关键症状: {", ".join(r.matched_key_symptoms) if r.matched_key_symptoms else "无"}
一般症状: {", ".join(r.matched_symptoms) if r.matched_symptoms else "无"}
舌象: {", ".join(r.matched_tongue) if r.matched_tongue else "无"}
脉象: {", ".join(r.matched_pulse) if r.matched_pulse else "无"}
排除项: {", ".join(r.matched_exclusions) if r.matched_exclusions else "无"}
治法: {r.treatment}
方剂: {", ".join(r.formulas) if r.formulas else "无"}
来源: {r.source}
说明: {r.explain}
"""
        )
    return "\n".join(lines)


def build_context_text(results):
    lines = []
    for i, (item, score) in enumerate(results, 1):
        lines.append(f"[{i}] 相似度: {score:.3f} | {item.meta}\n{item.text}")
    return "\n\n".join(lines) if lines else "未检索到相关资料。"


@st.cache_resource(show_spinner=True)
def load_system():
    embedder = Embedder()
    knowledge = load_knowledge()

    texts = []
    metas = []
    for item in knowledge:
        text = (
            f"证型:{item['name']} "
            f"大类:{item.get('category', '')} "
            f"同义词:{','.join(item.get('synonyms', []))} "
            f"关键症状:{','.join(item.get('key_symptoms', []))} "
            f"症状:{','.join(item.get('symptoms', []))} "
            f"舌象:{','.join(item.get('tongue', []))} "
            f"脉象:{','.join(item.get('pulse', []))} "
            f"治法:{item.get('treatment', '')} "
            f"方剂:{','.join(item.get('formulas', []))} "
            f"来源:{item.get('source', '')} "
            f"解释:{item.get('explain', '')}"
        )
        texts.append(text)
        metas.append(item["name"])

    vs = VectorStore(embedder)
    vs.build(texts, metas)
    return knowledge, vs


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("症状 → 打分 → 证型排序 → RAG → Qwen3-32B")

    knowledge, vs = load_system()
    agent = Agent()

    with st.sidebar:
        st.header("运行状态")
        st.write(f"知识条目数: {len(knowledge)}")
        st.write(f"测试模式: {'是' if os.getenv('TEST_MODE', '0') == '1' else '否'}")
        st.caption("如果 TEST_MODE=1，会返回假回答，用于先测试整条流程。")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_area(
        "输入症状",
        placeholder="例如：恶寒重、发热轻、无汗、头痛、鼻塞、流清涕",
        height=120,
    )

    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("开始分析", type="primary")
    with col2:
        clear_btn = st.button("清空对话")

    if clear_btn:
        st.session_state.chat_history = []
        st.rerun()

    if run_btn:
        if not user_input.strip():
            st.warning("请输入症状。")
            return

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if has_risk(user_input):
            msg = "检测到危险症状，请尽快线下就医或急诊评估。"
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.error(msg)
            return

        user_features = parse_user_input(user_input)
        ranked = rank_syndromes(user_features, knowledge, top_k=3)
        ranked_text = build_ranked_text(ranked)

        retrieval_results = vs.search(user_input, top_k=4)
        context_text = build_context_text(retrieval_results)

        answer = agent.answer(
            question=user_input,
            ranked_text=ranked_text,
            context=context_text,
            history=st.session_state.chat_history,
        )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.subheader("证型排序结果")
    if user_input.strip():
        user_features = parse_user_input(user_input)
        ranked = rank_syndromes(user_features, knowledge, top_k=3)

        for i, r in enumerate(ranked, 1):
            with st.expander(f"{i}. {r.name} | 得分 {r.score}"):
                st.write("关键症状:", ", ".join(r.matched_key_symptoms) if r.matched_key_symptoms else "无")
                st.write("一般症状:", ", ".join(r.matched_symptoms) if r.matched_symptoms else "无")
                st.write("舌象:", ", ".join(r.matched_tongue) if r.matched_tongue else "无")
                st.write("脉象:", ", ".join(r.matched_pulse) if r.matched_pulse else "无")
                st.write("排除项:", ", ".join(r.matched_exclusions) if r.matched_exclusions else "无")
                st.write("治法:", r.treatment)
                st.write("方剂:", ", ".join(r.formulas) if r.formulas else "无")
                st.write("来源:", r.source)
                st.write("说明:", r.explain)

    st.subheader("模型输出")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


if __name__ == "__main__":
    main()