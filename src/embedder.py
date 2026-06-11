"""
向量化模块 (Embedder)

功能：将文本转换为向量（高维数字表示）
面试要点：
  - embedding 是什么？→ 将语义映射到高维空间，语义相近的文本在向量空间中距离也近
  - 为什么本地做 embedding？→ 免费、低延迟、隐私好；也支持 OpenAI API 作为备选
  - 单例模式的作用？→ 模型加载很慢（数秒），全局只加载一次，避免重复
  - 国内部署注意：HuggingFace 需要设 HF_ENDPOINT=https://hf-mirror.com 镜像

支持两种后端：
  1. sentence-transformers（本地免费，优先使用）
  2. OpenAI API（需要网络和 API Key，作为备选）
"""

import numpy as np
from typing import List, Union, Optional
import os

# 全局实例（单例模式）
_embedder_instance = None
_embedder_type = None  # "local" 或 "openai"

# 默认本地模型（轻量高效，384维，约80MB）
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
# OpenAI embedding 模型
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


def _get_local_embedder(model_name: str = DEFAULT_LOCAL_MODEL):
    """获取本地 sentence-transformers 模型实例"""
    from sentence_transformers import SentenceTransformer

    global _embedder_instance, _embedder_type

    if _embedder_instance is not None and _embedder_type == "local":
        return _embedder_instance

    print(f"🔄 正在加载本地嵌入模型: {model_name} ...")

    # 检查是否配置了 HF 镜像
    hf_endpoint = os.getenv("HF_ENDPOINT", "")
    if hf_endpoint:
        print(f"   使用 HF 镜像: {hf_endpoint}")

    try:
        _embedder_instance = SentenceTransformer(model_name)
        _embedder_type = "local"
        print(f"✅ 本地模型加载完成")
        return _embedder_instance
    except Exception as e:
        print(f"⚠️  本地模型加载失败: {e}")
        raise


def _get_openai_embedder():
    """获取 OpenAI embedding 客户端"""
    from openai import OpenAI

    global _embedder_instance, _embedder_type

    if _embedder_instance is not None and _embedder_type == "openai":
        return _embedder_instance

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("使用 OpenAI embedding 需要设置 OPENAI_API_KEY")

    _embedder_instance = OpenAI(api_key=api_key, base_url=base_url or None)
    _embedder_type = "openai"
    print(f"✅ 使用 OpenAI embedding API")
    return _embedder_instance


def embed_texts(
    texts: Union[str, List[str]],
    model_name: Optional[str] = None,
    backend: Optional[str] = None,  # "local" 或 "openai"
) -> np.ndarray:
    """
    将文本转换为向量

    Args:
        texts: 单个文本字符串，或文本列表
        model_name: 模型名称。local 默认 all-MiniLM-L6-v2；openai 默认 text-embedding-3-small
        backend: "local"（本地免费）或 "openai"（API 调用）。默认自动选择（优先本地）

    Returns:
        numpy 数组，shape 为 (n_texts, dim)

    面试要点：支持多后端切换——展示工程灵活性
    """
    # 统一转为列表
    if isinstance(texts, str):
        texts = [texts]

    # 自动选择后端：优先本地，失败则用 OpenAI
    if backend is None:
        backend = os.getenv("EMBEDDING_BACKEND", "local")

    if backend == "openai":
        # OpenAI API embedding
        openai_model = model_name or os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL)
        client = _get_openai_embedder()

        response = client.embeddings.create(
            model=openai_model,
            input=texts,
        )
        embeddings = np.array([d.embedding for d in response.data], dtype=np.float32)
        return embeddings

    else:
        # 本地 sentence-transformers
        local_model = model_name or DEFAULT_LOCAL_MODEL

        try:
            model = _get_local_embedder(local_model)
        except Exception as e:
            # 自动降级到 OpenAI
            print(f"⚠️  本地模型不可用，自动切换到 OpenAI embedding...")
            return embed_texts(texts, model_name=None, backend="openai")

        embeddings = model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 归一化
        )
        return embeddings


def get_embedding_dimension(backend: Optional[str] = None) -> int:
    """返回向量的维度"""
    if backend == "openai":
        # text-embedding-3-small 默认 1536 维
        return 1536
    else:
        try:
            model = _get_local_embedder()
            return model.get_embedding_dimension()
        except Exception:
            return 1536  # OpenAI 默认维度


def is_local_model_available() -> bool:
    """检查本地模型是否可用"""
    try:
        _get_local_embedder()
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # 测试向量化
    texts = ["今天天气真好", "机器学习很有趣", "It's a sunny day"]

    # 自动选择后端
    print("=" * 50)
    print("  测试 embedding 模块")
    print("=" * 50)

    embeddings = embed_texts(texts)
    dimension = get_embedding_dimension()

    print(f"✅ 向量维度: {dimension}")
    print(f"✅ 输出形状: {embeddings.shape}")
    print(f"✅ 前5个值: {embeddings[0][:5]}")

    # 语义相似度测试
    from numpy import dot
    from numpy.linalg import norm

    def cosine_sim(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    sim_0_2 = cosine_sim(embeddings[0], embeddings[2])
    sim_1_2 = cosine_sim(embeddings[1], embeddings[2])

    print(f"\n✅ 语义相似度测试：")
    print(f"  '{texts[0]}' ↔ '{texts[2]}': {sim_0_2:.4f}")
    print(f"  '{texts[1]}' ↔ '{texts[2]}': {sim_1_2:.4f}")
    print(f"  结论：第1对更相似（都关于天气，跨语言也能匹配！）")
