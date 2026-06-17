"""
RAG 管道 (RAG Pipeline)

功能：编排所有模块，提供完整的文档上传和问答流程
面试要点：
  - Pipeline 设计模式：将分散的模块组合成完整的业务流程
  - 职责分离：每个模块只做一件事（单一职责原则）
  - 流程可观测性：每步都有日志，方便调试和监控
  - 面试时能讲清楚数据流向：
    ingest: 文件 → 文本 → 块 → 向量 → 存储
    query:  问题 → 向量 → 搜索 → 上下文 → LLM → 答案
"""

import os
from typing import List, Dict, Generator, Optional
from pathlib import Path

from .loader import load_document
from .chunker import chunk_text
from .embedder import embed_texts
from .vector_store import VectorStore
from .retriever import retrieve
from .generator import generate_answer, generate_answer_stream
from .conversation import get_conversation_manager
from .hybrid_retriever import BM25Index, hybrid_search
from .reranker import rerank


class RAGPipeline:
    """
    RAG 完整管道

    面试要点：这个类的设计体现了什么？
    1. 依赖注入：VectorStore 可以替换为任何实现相同接口的存储
    2. 有状态的流程管理：维护当前处理的文件列表
    3. 错误处理：每一步都可能出错，需要给用户明确反馈
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_documents",
        user_id: Optional[int] = None,
    ):
        """
        初始化 RAG 管道

        Args:
            persist_dir: 向量数据库持久化目录
            collection_name: 集合名称
        """
        # 用户隔离：不同用户使用不同 collection
        self.user_id = user_id
        if user_id:
            collection_name = f"rag_user_{user_id}_{collection_name}"

        self.vector_store = VectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
        self.documents: List[str] = []  # 已处理的文件列表
        self.bm25_index = BM25Index()  # BM25 关键词索引
        # 从已有数据重建 BM25 索引
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self):
        """从向量库重建 BM25 索引"""
        if self.vector_store.get_count() > 0:
            data = self.vector_store.collection.get(include=["documents"])
            if data["documents"]:
                self.bm25_index.build(data["documents"])

    def ingest(self, file_path: str) -> Dict:
        """
        文档上传处理流程

        流程：加载 → 切分 → 向量化 → 存储

        Args:
            file_path: 文档路径

        Returns:
            处理结果摘要

        面试要点：能画出这个流程图
        ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
        │ PDF    │→  │ 纯文本  │→  │ 块     │→  │ 向量   │→ ChromaDB
        │ Word   │   │        │   │ 块+元  │   │ 向量   │
        │ TXT    │   │        │   │ 块+元  │   │ 向量   │
        └────────┘   └────────┘   └────────┘   └────────┘
        """
        file_name = os.path.basename(file_path)

        # Step 1: 加载文档
        print(f"\n📄 [1/4] 加载文档: {file_name}")
        text = load_document(file_path)
        if not text.strip():
            raise ValueError(f"文档内容为空: {file_name}")

        # Step 2: 文本切分
        print(f"✂️  [2/4] 文本切分...")
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError(f"文档切分后无有效内容: {file_name}")

        # Step 3: 向量化
        print(f"🧮 [3/4] 生成向量（共 {len(chunks)} 个块）...")
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)

        # Step 4: 存储
        print(f"💾 [4/4] 存储到向量数据库...")
        count = self.vector_store.add_documents(chunks, embeddings, file_name)

        # Step 5: 重建 BM25 索引
        self._rebuild_bm25_index()

        self.documents.append(file_name)

        result = {
            "file_name": file_name,
            "text_length": len(text),
            "chunk_count": len(chunks),
            "stored_count": count,
            "total_in_store": self.vector_store.get_count(),
        }

        print(f"\n✅ 处理完成！{file_name}: "
              f"{len(text)} 字符 → {len(chunks)} 块 → 已存储")
        return result

    def query(
        self,
        question: str,
        top_k: int = 4,
        stream: bool = True,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "deepseek-chat",
        session_id: Optional[str] = None,
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ) -> Generator:
        """
        问答流程

        流程：取历史 → embed 问题 → 检索 → 生成答案 → 存对话

        Args:
            question: 用户问题
            top_k: 检索返回的文档块数量
            stream: 是否流式输出
            api_key: OpenAI API Key
            base_url: API Base URL
            model: 使用的模型
            session_id: 对话 ID（多轮对话支持）

        Yields:
            生成的内容（流式或非流式）
            最后 yield 检索上下文信息
        """
        if self.vector_store.get_count() == 0:
            yield "⚠️ 知识库为空，请先上传文档。"
            return

        # Step 0: 获取对话历史（多轮对话）
        history = None
        cm = get_conversation_manager()
        if session_id:
            history = cm.get_history(session_id)

        # Step 1: 检索（向量 / 混合）
        method = "混合检索" if use_hybrid else "向量检索"
        print(f"🔍 检索中 [{method}]: \"{question}\"")
        if use_hybrid:
            context_chunks = hybrid_search(
                query=question,
                vector_store=self.vector_store,
                bm25_index=self.bm25_index,
                top_k=top_k * 2 if use_rerank else top_k,
            )
        else:
            context_chunks = retrieve(
                query=question,
                vector_store=self.vector_store,
                top_k=top_k * 2 if use_rerank else top_k,
            )

        # Step 1.5: Rerank（可选）
        if use_rerank and context_chunks:
            print(f"🔄 Rerank 重排序: {len(context_chunks)} → {top_k}")
            context_chunks = rerank(
                query=question,
                candidates=context_chunks,
                top_k=top_k,
            )
            if context_chunks:
                print(f"   重排后 top{len(context_chunks)}")

        if not context_chunks:
            yield "⚠️ 未找到相关内容，请尝试换个问法或上传更多相关文档。"
            return

        print(f"   找到 {len(context_chunks)} 个相关片段"
              + (f" (对话历史: {len(history)} 条)" if history else ""))

        # Step 2: 记录用户问题
        if session_id:
            cm.add_message(session_id, "user", question)

        # Step 3: 生成
        full_answer = ""
        if stream:
            for chunk in generate_answer_stream(
                query=question,
                context_chunks=context_chunks,
                api_key=api_key,
                base_url=base_url,
                model=model,
                history=history,
            ):
                if isinstance(chunk, str):
                    full_answer += chunk
                yield chunk
        else:
            full_answer = generate_answer(
                query=question,
                context_chunks=context_chunks,
                api_key=api_key,
                base_url=base_url,
                model=model,
                history=history,
            )
            yield full_answer

        # Step 4: 记录助手回答
        if session_id and full_answer:
            # 去掉来源引用脚注再存储
            clean_answer = full_answer.split("\n\n---\n")[0]
            cm.add_message(session_id, "assistant", clean_answer)

        # 附加上下文信息（供 UI 展示）
        yield ("__CONTEXT__", context_chunks)

    def get_stats(self) -> Dict:
        """获取知识库统计信息"""
        stats = self.vector_store.get_stats()
        stats["documents"] = self.documents
        return stats

    def clear_knowledge_base(self):
        """清空知识库"""
        self.vector_store.clear()
        self.documents = []
        print("🗑️  知识库已清空")


# ============================================
# 命令行测试（不需要 UI 就能验证流程）
# ============================================
if __name__ == "__main__":
    import sys

    # 创建测试数据
    test_dir = "d:/练习/rag-project/data"
    os.makedirs(test_dir, exist_ok=True)

    test_content = """# 人工智能简介

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个重要分支。
它致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。

## 机器学习

机器学习（Machine Learning）是人工智能的核心子领域。它使计算机能够通过经验自动改进性能。
机器学习算法可以分为三大类：
1. 监督学习（Supervised Learning）：使用带标签的数据进行训练
2. 无监督学习（Unsupervised Learning）：从无标签数据中发现模式
3. 强化学习（Reinforcement Learning）：通过与环境交互来学习最优策略

## 深度学习

深度学习（Deep Learning）是机器学习的一个子集，使用多层人工神经网络。
它在图像识别、自然语言处理、语音识别等领域取得了突破性成果。

## 大语言模型

大语言模型（Large Language Model，LLM）是基于深度学习的自然语言处理模型。
GPT（Generative Pre-trained Transformer）是目前最具代表性的LLM架构。
ChatGPT于2022年11月发布，标志着AI进入了一个新时代。"""

    test_file = f"{test_dir}/ai_intro.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)

    # 测试完整流程
    pipeline = RAGPipeline(
        persist_dir="d:/练习/rag-project/chroma_db",
    )

    # 清空旧数据
    pipeline.clear_knowledge_base()

    # 上传文档
    print("=" * 60)
    print("  📥 测试：文档上传")
    print("=" * 60)
    result = pipeline.ingest(test_file)
    print(f"\n结果摘要: {result}")

    # 提问测试（不调用 API）
    print("\n" + "=" * 60)
    print("  🔍 测试：检索（不调用 LLM）")
    print("=" * 60)
    from .retriever import retrieve as direct_retrieve

    results = direct_retrieve(
        "什么是机器学习",
        pipeline.vector_store,
        top_k=3,
    )

    for i, r in enumerate(results):
        print(f"\n  #{i+1} [相似度={r['score']:.4f}]")
        print(f"  {r['text'][:100]}...")

    print(f"\n✅ 管道测试完成！总共 {pipeline.vector_store.get_count()} 个文档块")
