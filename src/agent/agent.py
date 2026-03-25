from __future__ import annotations

import os
from typing import Dict, List

from src.llm.client import get_client, get_model

TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

SYSTEM_PROMPT = """
你是一个中医AI助手。
你必须根据输入和候选证型进行结构化输出。

固定输出格式：
1. 主要判断
2. 可能证候排序
3. 证据依据
4. 建议下一步
5. 风险提示

要求：
- 优先依据候选证型排序结果
- 不要给绝对诊断
- 使用谨慎表达
- 如果存在危险症状，先提示线下就医
"""


class Agent:
    def __init__(self) -> None:
        self.client = get_client()
        self.model = get_model()

    def mock_answer(self, question: str, ranked_text: str, context: str) -> str:
        return f"""1. 主要判断
当前处于测试模式，尚未调用本地模型。

2. 可能证候排序
- 测试候选1
- 测试候选2
- 测试候选3

3. 证据依据
{ranked_text}

4. 建议下一步
先检查“症状 → 打分 → 证型排序”流程是否正常，再接入真实模型。

5. 风险提示
本结果仅用于功能测试，不可作为临床建议。
"""

    def answer(self, question: str, ranked_text: str, context: str, history: List[Dict[str, str]]) -> str:
        if TEST_MODE:
            return self.mock_answer(question, ranked_text, context)

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
        user_prompt = f"""
对话历史:
{history_text if history_text else "无"}

用户问题:
{question}

证型排序结果:
{ranked_text}

检索资料:
{context}

请按固定格式输出。
"""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=900,
        )
        return resp.choices[0].message.content or ""