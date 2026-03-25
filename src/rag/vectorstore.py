from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

try:
    import faiss
except Exception:
    faiss = None


@dataclass
class KnowledgeItem:
    text: str
    meta: str = ""


class VectorStore:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.index = None
        self.items: List[KnowledgeItem] = []

    def build(self, texts: List[str], metas: List[str] | None = None) -> None:
        if faiss is None:
            raise RuntimeError("请先安装 faiss-cpu")
        if not texts:
            raise RuntimeError("没有可用于建库的文本")

        vectors = self.embedder.embed(texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        if metas is None:
            metas = [""] * len(texts)

        self.items = [KnowledgeItem(text=t, meta=m) for t, m in zip(texts, metas)]

    def search(self, query: str, top_k: int = 4) -> List[Tuple[KnowledgeItem, float]]:
        if self.index is None:
            raise RuntimeError("知识库尚未构建")

        q = self.embedder.embed_one(query)[None, :]
        scores, idxs = self.index.search(q, top_k)

        results: List[Tuple[KnowledgeItem, float]] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx == -1:
                continue
            results.append((self.items[idx], float(score)))
        return results