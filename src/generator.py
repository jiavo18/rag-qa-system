"""
LLM 生成模块 (Generator)

功能：基于检索到的上下文，用 LLM 生成答案
面试要点：
  - Prompt Engineering：system prompt 定义角色，user prompt 包含上下文+问题
  - 上下文窗口管理：如果检索到的内容太长，需要截断
  - 流式输出：提升用户体验，面试时可展示
  - 来源引用：让用户知道答案来自哪里（可追溯性）
"""

from openai import OpenAI
from typing import List, Dict, Generator, Optional
import os
from dotenv import load_dotenv

# 加载 .env 文件（确保环境变量可用）
load_dotenv(override=True)

# Prompt 模板（面试常问：你怎么设计 prompt？）
SYSTEM_PROMPT = """你是一个基于文档的智能问答助手。你的回答必须基于提供的文档内容。

规则：
1. 优先使用文档中的信息回答问题
2. 如果文档中有明确答案，直接引用并注明来源
3. 如果文档信息不完整，可以结合常识补充，但必须说明哪些来自文档、哪些来自常识
4. 如果文档完全不相关，诚实告知用户
5. 回答要简洁、准确、结构化"""


def _build_user_prompt(
    query: str,
    context_chunks: List[Dict],
    history: List = None,
) -> str:
    """
    构建包含上下文的用户 prompt

    面试要点：prompt 结构设计
    - 有历史对话时先放历史，再放文档，最后当前问题
    - 每个上下文块标注来源
    - 明确指示基于上下文回答
    """
    prompt_parts = []

    # 1. 历史对话（多轮对话支持）
    if history:
        history_text = ""
        for role, content in history:
            label = "用户" if role == "user" else "助手"
            history_text += f"{label}：{content}\n"
        prompt_parts.append(f"## 历史对话\n{history_text}")

    # 2. 文档内容
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        source = chunk["metadata"].get("source", "未知来源")
        context_text += (
            f"\n--- 文档片段 {i + 1} [来源: {source}] ---\n"
            f"{chunk['text']}\n"
        )
    prompt_parts.append(f"## 文档内容\n{context_text}")

    # 3. 用户问题
    prompt_parts.append(f"## 用户问题\n{query}")

    prompt = "\n\n".join(prompt_parts)
    prompt += "\n\n请基于以上文档内容回答。如果文档内容不足，请明确指出。"
    return prompt


def generate_answer(
    query: str,
    context_chunks: List[Dict],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "deepseek-chat",
    history: List = None,
) -> str:
    """
    生成答案（非流式）

    Args:
        query: 用户问题
        context_chunks: 检索到的上下文块
        api_key: OpenAI API Key
        base_url: API Base URL（可选，使用代理或兼容 API）
        model: 使用的模型

    Returns:
        生成的答案文本

    面试要点：API 调用参数
    - temperature: 0.3 偏低，让回答更确定、更忠实于文档
    - max_tokens: 控制回复长度，避免浪费
    """
    if not context_chunks:
        return "⚠️ 知识库中没有找到相关内容。请先上传相关文档。"

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"

    if not api_key:
        return "❌ 请设置 OPENAI_API_KEY 环境变量或在界面中输入 API Key"

    import httpx
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(timeout=60.0),
    )

    user_prompt = _build_user_prompt(query, context_chunks, history)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content

        # 追加来源引用
        sources = set()
        for chunk in context_chunks:
            source = chunk["metadata"].get("source", "未知")
            sources.add(source)

        answer += f"\n\n---\n📚 **参考来源**: {', '.join(sources)}"

        return answer

    except Exception as e:
        return f"❌ LLM 调用失败: {str(e)}"


def generate_answer_stream(
    query: str,
    context_chunks: List[Dict],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "deepseek-chat",
    history: List = None,
) -> Generator[str, None, str]:
    """
    流式生成答案（面试加分项）

    Yields:
        增量文本内容
        最后 yield 来源引用字符串

    面试要点：流式输出提升用户体验
    - 用户不用等完整回答，逐字出现
    - 实现方式：yield 每个 token
    """
    if not context_chunks:
        yield "⚠️ 知识库中没有找到相关内容。请先上传相关文档。"
        return

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"

    if not api_key:
        yield "❌ 请设置 OPENAI_API_KEY 环境变量或在界面中输入 API Key"
        return

    import httpx
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(timeout=60.0),  # 加长超时
    )

    user_prompt = _build_user_prompt(query, context_chunks, history)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
            stream=True,  # ✅ 关键：启用流式
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        # 来源引用
        sources = set()
        for chunk in context_chunks:
            source = chunk["metadata"].get("source", "未知")
            sources.add(source)

        yield f"\n\n---\n📚 **参考来源**: {', '.join(sources)}"

    except Exception as e:
        yield f"\n\n❌ LLM 调用失败: {str(e)}"


if __name__ == "__main__":
    # 测试（不调用 API，只看 prompt 构建）
    test_chunks = [
        {
            "text": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
            "metadata": {"source": "ai_intro.pdf", "chunk_index": 0}
        },
        {
            "text": "深度学习使用多层神经网络，在图像识别和自然语言处理方面表现出色。",
            "metadata": {"source": "ai_intro.pdf", "chunk_index": 1}
        },
    ]

    prompt = _build_user_prompt("什么是机器学习？", test_chunks)
    print("✅ 构建的 Prompt 预览:")
    print(prompt[:500])
