"""
混合检索 (Hybrid Retriever)

功能：BM25 关键词检索 + 向量语义检索 → RRF 融合
面试要点：
  - 为什么混合检索？→ 向量检索理解语义但可能漏关键词，
    BM25 精确匹配关键词但不懂同义词，两者互补
  - RRF (Reciprocal Rank Fusion)：不依赖原始分数大小，
    只用排名位置融合，简单有效，工业界常用
  - 为什么用 jieba？→ 中文分词，BM25 需要分词后的词列表
"""

import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import jieba

from .vector_store import VectorStore
from .embedder import embed_texts


class BM25Index:
    """
    BM25 关键词检索引擎

    面试要点：BM25 是 TF-IDF 的改进版，考虑了词频饱和度和文档长度归一化
    """

    def __init__(self):
        self.bm25 = None
        self.corpus: List[str] = []  # 原始文本
        self.tokenized_corpus: List[List[str]] = []  # 分词后的文本

    def build(self, documents: List[str]):
        """构建 BM25 索引"""
        self.corpus = documents
        self.tokenized_corpus = [_tokenize(doc) for doc in documents]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        BM25 检索

        Returns:
            [(doc_index, bm25_score), ...]  按分数降序
        """
        if self.bm25 is None or not self.corpus:
            return []

        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 按分数排序，返回 top_k
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:top_k] if score > 0]


def _tokenize(text: str) -> List[str]:
    """中文分词（jieba 精确模式）"""
    return list(jieba.cut(text))


def rrf_fusion(
    vector_results: List[Tuple[int, float, Dict]],
    bm25_results: List[Tuple[int, float]],
    k: int = 60,
    final_top_k: int = 4,
) -> List[int]:
    """
    RRF (Reciprocal Rank Fusion) 融合算法

    公式: score = sum(1 / (k + rank_i))  对每个来源的排名求和

    Args:
        vector_results: [(chunk_index, similarity, chunk_dict), ...]
        bm25_results: [(chunk_index, bm25_score), ...]
        k: 平滑参数，通常取 60
        final_top_k: 最终返回数量

    Returns:
        融合后的 chunk_index 列表（按 RRF 分数降序）

    面试要点：
    - 不需要原始分数，只依赖排名位置
    - k 的作用：防止排名1和排名2之间差距过大（k=60 让 1/(60+1) ≈ 0.016，1/(60+2) ≈ 0.016，差距很小）
    """
    rrf_scores = {}

    # 向量检索排名
    for rank, (idx, sim, _) in enumerate(vector_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)

    # BM25 检索排名
    for rank, (idx, bm25_score) in enumerate(bm25_results, start=1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)

    # 按 RRF 分数排序
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, score in sorted_items[:final_top_k]]


def hybrid_search(
    query: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    top_k: int = 4,
) -> List[Dict]:
    """
    混合检索：向量 + BM25 → RRF 融合

    Args:
        query: 用户问题
        vector_store: 向量存储
        bm25_index: BM25 索引
        top_k: 最终返回的文档块数量

    Returns:
        融合后的文档块列表 [{text, metadata, score, hybrid_score}, ...]
    """
    if vector_store.get_count() == 0:
        return []

    # 1. 向量检索（取 top_k * 2 做候选池）
    query_embedding = embed_texts(query)
    vector_results_raw = vector_store.search(query_embedding, top_k=top_k * 2)

    # 转为带索引的格式
    all_texts = _get_all_texts(vector_store)
    vector_results = []
    for vr in vector_results_raw:
        # 找到在语料中的索引
        text = vr["text"]
        if text in all_texts:
            chunk_idx = all_texts.index(text)
            vector_results.append((chunk_idx, 1.0 - (vr.get("score", 0) or 0), vr))

    # 2. BM25 检索（取 top_k * 2）
    bm25_raw = bm25_index.search(query, top_k=top_k * 2)

    # 3. RRF 融合
    fused_indices = rrf_fusion(vector_results, bm25_raw, final_top_k=top_k)

    # 4. 构建最终结果
    results = []
    all_chunks = _get_all_chunks(vector_store)
    for idx in fused_indices:
        if idx < len(all_chunks):
            chunk = all_chunks[idx]
            rrf_score = None
            combined_score = 1 / (60 + fused_indices.index(idx) + 1)
            results.append({
                "text": chunk["text"] if isinstance(chunk, dict) else chunk,
                "metadata": chunk.get("metadata", {}) if isinstance(chunk, dict) else {},
                "score": combined_score,
                "hybrid_score": combined_score,
            })

    return results


def _get_all_texts(store: VectorStore) -> List[str]:
    """获取向量库中所有文档文本"""
    if store.get_count() == 0:
        return []
    data = store.collection.get(include=["documents"])
    return data["documents"] or []


def _get_all_chunks(store: VectorStore) -> List[Dict]:
    """获取向量库中所有文档块"""
    if store.get_count() == 0:
        return []
    data = store.collection.get(include=["documents", "metadatas"])
    chunks = []
    if data["documents"]:
        for i, doc in enumerate(data["documents"]):
            meta = data["metadatas"][i] if data["metadatas"] else {}
            chunks.append({"text": doc, "metadata": meta})
    return chunks


if __name__ == "__main__":
    print("✅ Hybrid Retriever 模块加载正常")

    # 测试 BM25 分词和检索
    bm25 = BM25Index()
    test_docs = [
        "机器学习是人工智能的重要分支",
        "深度学习使用多层神经网络",
        "自然语言处理是AI的一个应用领域",
        "Python是常用的机器学习编程语言",
    ]
    bm25.build(test_docs)

    results = bm25.search("什么是机器学习", top_k=2)
    print(f"\nBM25 检索 '什么是机器学习':")
    for idx, score in results:
        print(f"  [{score:.4f}] {test_docs[idx]}")
