"""
文本切分器 (Text Chunker)

功能：将长文本切分成适合向量化和检索的小块
面试要点：
  - 为什么需要切分？→ LLM 上下文窗口有限；小块检索更精准
  - chunk_size 怎么选？→ 500 适合中文（约300-500字/块）；太大降低检索精度，太小丢失上下文
  - chunk_overlap 为什么重要？→ 防止关键信息落在两个块的边界被切断
  - RecursiveCharacterTextSplitter 原理？→ 按优先级分隔符递归切分：段落→句子→词→字符
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    将文本切分成重叠的块

    Args:
        text: 原始文本
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块的重叠字符数

    Returns:
        文本块列表，每个块包含 {text, metadata}
    """
    # 中文友好的分隔符：段落 → 句子 → 中文标点 → 空格 → 字符
    separators = [
        "\n\n",     # 段落分隔
        "\n",       # 换行
        "。",       # 中文句号
        "；",       # 中文分号
        "，",       # 中文逗号
        ".",        # 英文句号
        " ",        # 空格
        "",         # 逐字符
    ]

    # 计算实际 token 数（面试加分点：不要只看字符数）
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        token_count = len(enc.encode(text))
    except Exception:
        token_count = len(text) // 2  # 中文粗略估算

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    # 构建带元数据的块
    chunks = []
    for i, chunk_text_value in enumerate(raw_chunks):
        # 计算该块在原文本中的位置
        start_pos = text.find(chunk_text_value) if i == 0 else text.find(chunk_text_value)
        chunks.append({
            "text": chunk_text_value,
            "metadata": {
                "chunk_index": i,
                "chunk_count": len(raw_chunks),
                "char_start": start_pos if start_pos >= 0 else 0,
                "token_estimate": len(chunk_text_value) // 2,
            }
        })

    return chunks


if __name__ == "__main__":
    # 测试切分
    test_text = """人工智能（AI）是计算机科学的一个分支。

它致力于创造能够模拟人类智能的系统。机器学习是AI的一个重要子领域。

深度学习则是机器学习的一种方法，使用多层神经网络来处理复杂的模式识别任务。

近年来，大语言模型（LLM）如GPT系列取得了突破性进展，展现出了强大的自然语言理解和生成能力。"""

    chunks = chunk_text(test_text, chunk_size=100, chunk_overlap=20)
    print(f"✅ 原文 {len(test_text)} 字符 → {len(chunks)} 个块\n")
    for c in chunks:
        print(f"  块 {c['metadata']['chunk_index']}: "
              f"[{len(c['text'])}字] {c['text'][:60]}...")
