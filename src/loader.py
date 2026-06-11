"""
文档加载器 (Document Loader)

功能：将不同格式的文档解析为纯文本
面试要点：
  - 为什么需要统一的文本接口？→ 下游模块只处理纯文本，解耦
  - 如何处理不同格式？→ 策略模式，按扩展名分发到不同解析器
  - 大文件怎么办？→ 流式读取，避免一次性加载到内存
"""

import os
from pathlib import Path


def load_document(file_path: str) -> str:
    """
    加载文档，根据扩展名选择解析器

    Args:
        file_path: 文档路径

    Returns:
        文档的纯文本内容

    Raises:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        return _load_txt(file_path)
    elif ext == ".md":
        return _load_txt(file_path)  # Markdown 本质是纯文本
    elif ext == ".pdf":
        return _load_pdf(file_path)
    elif ext == ".docx":
        return _load_docx(file_path)
    else:
        raise ValueError(
            f"不支持的文件格式: {ext}\n"
            f"当前支持: .txt, .md, .pdf, .docx"
        )


def _load_txt(file_path: str) -> str:
    """加载纯文本/Markdown 文件，使用 UTF-8 编码"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_pdf(file_path: str) -> str:
    """
    加载 PDF 文件

    面试要点：PyPDF2 按页读取，需要用空行连接各页
    """
    from PyPDF2 import PdfReader

    reader = PdfReader(file_path)
    text_parts = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    if not text_parts:
        raise ValueError(f"PDF 文件无法提取文本（可能是扫描版）: {file_path}")

    return "\n\n".join(text_parts)


def _load_docx(file_path: str) -> str:
    """
    加载 Word 文档

    面试要点：python-docx 按段落读取，每个 <w:p> 标签是一个段落
    """
    from docx import Document

    doc = Document(file_path)
    text_parts = []

    for para in doc.paragraphs:
        if para.text.strip():
            text_parts.append(para.text)

    return "\n\n".join(text_parts)


# ============================================
# 面试自测
# ============================================
if __name__ == "__main__":
    # 测试：创建一个测试文件然后加载
    test_file = "d:/练习/rag-project/data/test.txt"
    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    with open(test_file, "w", encoding="utf-8") as f:
        f.write("这是第一行测试文本。\n这是第二行测试文本。")

    text = load_document(test_file)
    print(f"✅ 加载成功，共 {len(text)} 个字符")
    print(f"内容预览: {text[:100]}...")
