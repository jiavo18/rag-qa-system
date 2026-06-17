"""
Reranker 重排序模块

功能：用 Cross-Encoder 对检索结果做精细排序
面试要点：
  - Bi-Encoder vs Cross-Encoder：
    Bi-Encoder（向量检索）：query 和 doc 分别编码 → 快，适合海量初筛
    Cross-Encoder（重排序）：query 和 doc 一起编码 → 慢但准，适合精筛
  - 为什么二阶段？粗筛（向量/BM25）快但粗，精筛（Cross-Encoder）准但慢
  - 模型：ms-marco-MiniLM-L-6-v2，MS MARCO 数据集训练，专门做 passage ranking
"""

from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
import numpy as np

# 默认 rerank 模型
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"

# 全局单例
_reranker_instance = None  # None=未加载, False=加载失败
_reranker_load_attempted = False


def _get_reranker(model_name: str = DEFAULT_RERANK_MODEL) -> Optional[CrossEncoder]:
    """获取 Cross-Encoder 模型（单例模式，失败时返回 None 优雅降级）"""
    global _reranker_instance, _reranker_load_attempted
    if _reranker_load_attempted:
        return _reranker_instance  # 可能是 None(加载失败) 或模型实例
    _reranker_load_attempted = True
    try:
        _reranker_instance = CrossEncoder(model_name)
        print(f"✅ Rerank 模型加载完成: {model_name}")
        return _reranker_instance
    except Exception as e:
        print(f"⚠️  Rerank 模型加载失败（将跳过重排）: {e}")
        _reranker_instance = None
        return None


def rerank(
    query: str,
    candidates: List[Dict],
    top_k: int = 4,
    threshold: float = None,
) -> List[Dict]:
    """
    用 Cross-Encoder 对候选文档块重新排序

    Args:
        query: 用户问题
        candidates: 候选文档块列表 [{text, metadata, score}, ...]
        top_k: 返回数量
        threshold: 最低相关性阈值（低于此值的被过滤）

    Returns:
        重排序后的文档块列表（附带 rerank_score 字段）

    面试要点：Cross-Encoder 把每对 (query, doc) 拼成
    "[CLS] query [SEP] doc [SEP]" 输入 Transformer，输出一个相关性分数
    """
    if not candidates:
        return []

    model = _get_reranker()
    if model is None:
        # 模型不可用，直接返回候选（保留原排序）
        return candidates[:top_k]

    # 构建 (query, doc) 对
    pairs = [(query, c["text"]) for c in candidates]

    # 批量打分
    scores = model.predict(pairs)

    # 按分数排序
    ranked = list(zip(candidates, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    # 过滤和截断
    results = []
    for cand, score in ranked:
        if threshold is not None and score < threshold:
            continue
        cand["rerank_score"] = float(score)
        results.append(cand)
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    print("✅ Reranker 模块加载正常")

    # 测试 rerank（首次运行会下载模型）
    test_query = "什么是机器学习"
    test_candidates = [
        {"text": "机器学习是人工智能的一个分支，使计算机能从数据中学习。",
         "metadata": {"source": "doc1"}, "score": 0.8},
        {"text": "今天天气很好，适合出去玩。",
         "metadata": {"source": "doc2"}, "score": 0.3},
        {"text": "深度学习是机器学习的子集，使用多层神经网络。",
         "metadata": {"source": "doc3"}, "score": 0.7},
    ]

    results = rerank(test_query, test_candidates, top_k=2)
    print(f"\nRerank '什么是机器学习' (top 2):")
    for i, r in enumerate(results):
        score = r.get('rerank_score', r.get('score', 0))
        print(f"  #{i+1} [score={score:.4f}] {r['text'][:60]}...")
    print()
    if _reranker_instance is None:
        print("⚠️  Rerank 模型未加载，使用原始排序（系统可正常工作）")
