"""
向量存储 (Vector Store)

功能：存储文档块的向量表示，并提供相似度搜索
面试要点：
  - 为什么选 ChromaDB？→ 轻量（纯Python，pip install即可）、内置持久化、开源免费
  - 为什么不用 FAISS？→ FAISS 是纯向量索引，不存储原始文本；ChromaDB 自带元数据管理
  - 生产环境替代？→ Milvus、Pinecone、Weaviate、PostgreSQL + pgvector
  - 相似度计算？→ 默认用余弦相似度（向量已 L2 归一化，点积 = 余弦相似度）
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
import os


class VectorStore:
    """
    ChromaDB 向量存储封装

    面试要点：封装的意义 → 如果以后想换 Milvus/Pinecone，
    只需改这个类，其他代码不动（依赖倒置原则）
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_dir: str = "./chroma_db",
    ):
        """
        初始化向量存储

        Args:
            collection_name: 集合名称（相当于数据库的表）
            persist_dir: 持久化目录
        """
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        # 确保目录存在
        os.makedirs(persist_dir, exist_ok=True)

        # 创建 ChromaDB 客户端（持久化模式）
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG 知识库文档集合"},
        )

        print(f"✅ 向量存储已就绪: {persist_dir}/{collection_name} "
              f"({self.collection.count()} 条记录)")

    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: np.ndarray,
        source_file: str,
    ) -> int:
        """
        将文档块和向量添加到存储

        Args:
            chunks: 文本块列表 [{text, metadata}, ...]
            embeddings: 对应的向量，shape (n, dim)
            source_file: 来源文件名

        Returns:
            添加的文档数量

        面试要点：embedding 和 chunk 必须严格一一对应
        """
        if len(chunks) == 0:
            return 0

        n = len(chunks)

        # 构建 ChromaDB 所需的参数
        ids = [f"chunk_{i}_{hash(chunks[i]['text']) % 100000}"
               for i in range(n)]

        documents = [c["text"] for c in chunks]

        metadatas = [
            {
                **c["metadata"],
                "source": os.path.basename(source_file),
            }
            for c in chunks
        ]

        # 批量插入
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        print(f"✅ 已存储 {n} 个文档块（来源: {os.path.basename(source_file)}）")
        return n

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
    ) -> List[Dict]:
        """
        向量相似度搜索

        Args:
            query_embedding: 查询向量，shape (dim,) 或 (1, dim)
            top_k: 返回最相似的 top_k 个结果

        Returns:
            [{text, metadata, score}, ...]
            score 为距离值（越小越相关）

        面试要点：top_k 不是越大越好——
        - 太小：可能遗漏关键信息
        - 太大：引入噪声，增加 LLM 调用成本
        - 典型值：3-5
        """
        if self.collection.count() == 0:
            return []

        # 处理 2D 数组：embed_texts 对单个文本返回 shape (1, dim)
        emb = query_embedding
        if emb.ndim == 2:
            emb = emb[0]  # 取第一行

        results = self.collection.query(
            query_embeddings=[emb.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # 格式化结果
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": results["distances"][0][i] if results["distances"] else None,
                })

        return formatted

    def clear(self):
        """清空集合（重新上传文档时使用）"""
        # 删除旧集合并创建新集合
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "RAG 知识库文档集合"},
        )
        print(f"🗑️  集合已清空")

    def get_count(self) -> int:
        """返回当前存储的文档块数量"""
        return self.collection.count()

    def get_stats(self) -> Dict:
        """返回存储统计信息"""
        count = self.collection.count()
        # 获取一些元数据样本
        sources = set()
        if count > 0:
            sample = self.collection.get(limit=min(count, 100), include=["metadatas"])
            if sample["metadatas"]:
                for m in sample["metadatas"]:
                    if m and "source" in m:
                        sources.add(m["source"])

        return {
            "total_chunks": count,
            "source_files": list(sources),
            "persist_dir": self.persist_dir,
        }


if __name__ == "__main__":
    # 快速测试
    store = VectorStore(persist_dir="d:/练习/rag-project/chroma_db")

    # 造一些测试数据
    test_texts = ["人工智能基础知识", "机器学习算法概述", "深度学习框架介绍"]
    from src.embedder import embed_texts
    test_embeddings = embed_texts(test_texts)

    test_chunks = [
        {"text": t, "metadata": {"chunk_index": i, "chunk_count": 3}}
        for i, t in enumerate(test_texts)
    ]

    store.clear()
    store.add_documents(test_chunks, test_embeddings, "test.txt")

    # 搜索测试
    query_emb = embed_texts("什么是机器学习")
    results = store.search(query_emb, top_k=2)

    print(f"\n✅ 搜索 '什么是机器学习' 的结果:")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['text'][:50]}...")
