"""
检索模块 (Retriever)

功能：根据用户问题，从向量存储中检索最相关的文档块
面试要点：
  - 检索的本质？→ 用问题的向量表示，在文档向量空间中找最邻近的点
  - 为什么检索比关键词搜索好？→ 理解语义：查"如何学习"能找到"学习方法"相关内容
  - 检索质量衡量？→ recall@k（前k个结果中包含正确答案的比例）
"""

from typing import List, Dict
import numpy as np
from .vector_store import VectorStore
from .embedder import embed_texts


def retrieve(
    query: str,
    vector_store: VectorStore,
    top_k: int = 4,
) -> List[Dict]:
    """
    检索与问题最相关的文档块

    流程：
    1. 将用户问题转为向量
    2. 在向量存储中搜索 top_k 个最相似的块
    3. 返回文本块及其相似度分数

    Args:
        query: 用户问题
        vector_store: 向量存储实例
        top_k: 返回结果数量

    Returns:
        [{text, metadata, score}, ...]
    """
    if vector_store.get_count() == 0:
        return []

    # 1. 生成查询向量
    query_embedding = embed_texts(query)
    # embed_texts 对于单个字符串返回 shape (dim,)

    # 2. 向量搜索
    results = vector_store.search(query_embedding, top_k=top_k)

    return results


def retrieve_with_threshold(
    query: str,
    vector_store: VectorStore,
    top_k: int = 4,
    score_threshold: float = None,
) -> List[Dict]:
    """
    带阈值的检索（过滤不够相关的结果）

    面试要点：阈值过滤 → 防止引入不相关内容污染 LLM 的回答
    ChromaDB 返回的是 L2 距离（越小越好），一般 < 1.0 认为是相关的
    """
    results = retrieve(query, vector_store, top_k=top_k)

    if score_threshold is not None:
        results = [r for r in results
                   if r["score"] is not None and r["score"] < score_threshold]

    return results


if __name__ == "__main__":
    # 测试检索（需要有已存储的数据）
    from .vector_store import VectorStore
    from .embedder import embed_texts

    store = VectorStore(persist_dir="d:/练习/rag-project/chroma_db")

    if store.get_count() > 0:
        results = retrieve("什么是机器学习", store, top_k=3)
        print(f"✅ 检索到 {len(results)} 个结果\n")
        for i, r in enumerate(results):
            print(f"  #{i+1} [score={r['score']:.4f}]: {r['text'][:80]}...")
    else:
        print("⚠️  知识库为空，请先运行 vector_store.py 添加测试数据")
